// Package proxy is the HTTP side of the gateway: it routes OpenAI-style
// requests to kvserve backends, streams responses through, fails over when a
// backend is unreachable, and health-checks the pool.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/FrimpsManu/kvserve/gateway/internal/router"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

const maxBodyBytes = 16 << 20

var (
	requestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kvgateway_requests_total", Help: "Proxied requests by backend and status code.",
	}, []string{"backend", "code"})
	failoversTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "kvgateway_failovers_total", Help: "Requests retried on another backend after a connection failure.",
	}, []string{"backend"})
	firstByteSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "kvgateway_time_to_first_byte_seconds",
		Help:    "Gateway receipt to first response byte from the backend.",
		Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
	}, []string{"backend"})
	inflightGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "kvgateway_backend_inflight", Help: "Requests in flight per backend.",
	}, []string{"backend"})
	healthyGauge = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "kvgateway_backend_healthy", Help: "1 if the backend passed its last health check.",
	}, []string{"backend"})
)

type Gateway struct {
	Backends []*router.Backend
	Policy   router.Policy
	Keyer    router.PrefixKey
	Client   *http.Client
	Log      *slog.Logger
}

func New(backends []*router.Backend, policy router.Policy, keyer router.PrefixKey, log *slog.Logger) *Gateway {
	transport := &http.Transport{
		DialContext:         (&net.Dialer{Timeout: 2 * time.Second}).DialContext,
		MaxIdleConnsPerHost: 512, // a burst opens hundreds of concurrent streams per backend
		IdleConnTimeout:     90 * time.Second,
	}
	return &Gateway{Backends: backends, Policy: policy, Keyer: keyer, Client: &http.Client{Transport: transport}, Log: log}
}

func (g *Gateway) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/completions", g.proxy)
	mux.HandleFunc("POST /v1/chat/completions", g.proxy)
	mux.HandleFunc("GET /v1/models", g.proxy)
	mux.HandleFunc("GET /health", g.health)
	mux.HandleFunc("GET /gateway/backends", g.status)
	return mux
}

func (g *Gateway) proxy(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	if err != nil {
		http.Error(w, "request body too large or unreadable", http.StatusBadRequest)
		return
	}
	candidates := router.Healthy(g.Backends)
	if len(candidates) == 0 {
		writeError(w, http.StatusServiceUnavailable, "no healthy backends")
		return
	}
	key, _ := g.Keyer.Key(body) // key 0 for keyless requests: still a valid, stable key
	order := g.Policy.Order(key, candidates)

	// Fail over only on connection errors, before any response bytes are sent:
	// once a backend has started generating, retrying elsewhere would duplicate
	// work and could double-bill tokens.
	for i, b := range order {
		resp, err := g.send(r.Context(), b, r, body)
		if err != nil {
			if r.Context().Err() != nil {
				return // client went away
			}
			g.Log.Warn("backend unreachable, failing over", "backend", b.URL, "err", err)
			b.SetHealthy(false)
			healthyGauge.WithLabelValues(b.URL).Set(0)
			if i+1 < len(order) {
				failoversTotal.WithLabelValues(b.URL).Inc()
			}
			continue
		}
		g.relay(w, resp, b, start)
		return
	}
	writeError(w, http.StatusBadGateway, "all backends unreachable")
}

func (g *Gateway) send(ctx context.Context, b *router.Backend, r *http.Request, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, r.Method, b.URL+r.URL.Path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", r.Header.Get("Content-Type"))
	req.Header.Set("Accept", r.Header.Get("Accept"))
	b.Acquire()
	inflightGauge.WithLabelValues(b.URL).Set(float64(b.Inflight()))
	resp, err := g.Client.Do(req)
	if err != nil {
		b.Release()
		inflightGauge.WithLabelValues(b.URL).Set(float64(b.Inflight()))
	}
	return resp, err
}

// relay copies the backend response to the client, flushing after every read so
// server-sent events stream token by token instead of being buffered.
func (g *Gateway) relay(w http.ResponseWriter, resp *http.Response, b *router.Backend, start time.Time) {
	defer func() {
		resp.Body.Close()
		b.Release()
		inflightGauge.WithLabelValues(b.URL).Set(float64(b.Inflight()))
	}()
	for _, h := range []string{"Content-Type", "Cache-Control"} {
		if v := resp.Header.Get(h); v != "" {
			w.Header().Set(h, v)
		}
	}
	w.Header().Set("X-Kvserve-Backend", b.URL)
	w.WriteHeader(resp.StatusCode)
	requestsTotal.WithLabelValues(b.URL, strconv.Itoa(resp.StatusCode)).Inc()

	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	first := true
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if first {
				firstByteSeconds.WithLabelValues(b.URL).Observe(time.Since(start).Seconds())
				first = false
			}
			if _, werr := w.Write(buf[:n]); werr != nil {
				return // client disconnected; closing the backend body aborts generation there
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			if !errors.Is(err, io.EOF) {
				g.Log.Warn("backend stream ended with error", "backend", b.URL, "err", err)
			}
			return
		}
	}
}

func (g *Gateway) health(w http.ResponseWriter, _ *http.Request) {
	if len(router.Healthy(g.Backends)) == 0 {
		writeError(w, http.StatusServiceUnavailable, "no healthy backends")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok"}`))
}

func (g *Gateway) status(w http.ResponseWriter, _ *http.Request) {
	type row struct {
		URL      string `json:"url"`
		Healthy  bool   `json:"healthy"`
		Inflight int64  `json:"inflight"`
	}
	rows := make([]row, len(g.Backends))
	for i, b := range g.Backends {
		rows[i] = row{b.URL, b.Healthy(), b.Inflight()}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"policy": g.Policy.Name(), "backends": rows})
}

// HealthCheck polls each backend's /health until ctx is cancelled. A backend
// marked down by a failed request comes back as soon as a check passes.
func (g *Gateway) HealthCheck(ctx context.Context, every time.Duration) {
	client := &http.Client{Timeout: every}
	check := func() {
		for _, b := range g.Backends {
			ok := false
			if resp, err := client.Get(b.URL + "/health"); err == nil {
				ok = resp.StatusCode == http.StatusOK
				resp.Body.Close()
			}
			if ok != b.Healthy() {
				g.Log.Info("backend health changed", "backend", b.URL, "healthy", ok)
			}
			b.SetHealthy(ok)
			v := 0.0
			if ok {
				v = 1
			}
			healthyGauge.WithLabelValues(b.URL).Set(v)
		}
	}
	check()
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			check()
		}
	}
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": msg, "code": code}})
}
