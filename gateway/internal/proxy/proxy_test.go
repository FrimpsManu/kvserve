package proxy

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/FrimpsManu/kvserve/gateway/internal/router"
)

func quietLog() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

func newGateway(urls []string, policy router.Policy) (*Gateway, *httptest.Server) {
	bs := make([]*router.Backend, len(urls))
	for i, u := range urls {
		bs[i] = router.NewBackend(u)
	}
	gw := New(bs, policy, router.PrefixKey{PrefixChars: 64, PrefixTokens: 16}, quietLog())
	return gw, httptest.NewServer(gw.Routes())
}

// sseBackend streams n events, waiting for `release` before the last one, so a
// test can observe that earlier events arrived without buffering.
func sseBackend(name string, n int, release <-chan struct{}) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		for i := 0; i < n; i++ {
			if i == n-1 && release != nil {
				select {
				case <-release:
				case <-r.Context().Done():
					return
				}
			}
			fmt.Fprintf(w, "data: %s-%d\n\n", name, i)
			f.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
}

func TestStreamsIncrementally(t *testing.T) {
	release := make(chan struct{})
	backend := sseBackend("b0", 3, release)
	defer backend.Close()
	_, gw := newGateway([]string{backend.URL}, &router.RoundRobin{})
	defer gw.Close()

	resp, err := http.Post(gw.URL+"/v1/completions", "application/json", strings.NewReader(`{"prompt":"hi","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.Header.Get("X-Kvserve-Backend") != backend.URL {
		t.Fatalf("missing/wrong backend header: %q", resp.Header.Get("X-Kvserve-Backend"))
	}
	reader := bufio.NewReader(resp.Body)
	// The backend is blocked before its last event; the first two must already be here.
	for i := 0; i < 2; i++ {
		line, err := reader.ReadString('\n')
		if err != nil || !strings.HasPrefix(line, fmt.Sprintf("data: b0-%d", i)) {
			t.Fatalf("event %d not streamed through: %q %v", i, line, err)
		}
		reader.ReadString('\n') // blank separator line
	}
	close(release)
	rest, _ := io.ReadAll(reader)
	if !strings.Contains(string(rest), "b0-2") || !strings.Contains(string(rest), "[DONE]") {
		t.Fatalf("tail of stream missing: %q", rest)
	}
}

func TestFailsOverToHealthyBackend(t *testing.T) {
	dead := httptest.NewServer(http.NotFoundHandler())
	deadURL := dead.URL
	dead.Close() // connection refused from now on
	alive := sseBackend("alive", 1, nil)
	defer alive.Close()

	g, gw := newGateway([]string{deadURL, alive.URL}, &router.RoundRobin{})
	defer gw.Close()
	resp, err := http.Post(gw.URL+"/v1/completions", "application/json", strings.NewReader(`{"prompt":"x"}`))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(string(body), "alive-0") {
		t.Fatalf("expected failover to the live backend, got %d %q", resp.StatusCode, body)
	}
	if g.Backends[0].Healthy() {
		t.Fatal("unreachable backend should be marked unhealthy")
	}
	for _, b := range g.Backends {
		if b.Inflight() != 0 {
			t.Fatalf("%s inflight = %d after request, want 0", b, b.Inflight())
		}
	}
}

func TestAllBackendsDown(t *testing.T) {
	dead := httptest.NewServer(http.NotFoundHandler())
	url := dead.URL
	dead.Close()
	g, gw := newGateway([]string{url}, &router.RoundRobin{})
	defer gw.Close()

	resp, _ := http.Post(gw.URL+"/v1/completions", "application/json", strings.NewReader(`{"prompt":"x"}`))
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", resp.StatusCode)
	}
	// Now marked down: the next request is refused without trying, and health reports it.
	resp, _ = http.Post(gw.URL+"/v1/completions", "application/json", strings.NewReader(`{"prompt":"x"}`))
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 once no backend is healthy", resp.StatusCode)
	}
	resp, _ = http.Get(gw.URL + "/health")
	resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable || g.Backends[0].Healthy() {
		t.Fatal("gateway health should be 503 with no healthy backends")
	}
}

func TestClientDisconnectReleasesBackend(t *testing.T) {
	release := make(chan struct{}) // never closed: the backend would stream forever
	backend := sseBackend("slow", 2, release)
	defer backend.Close()
	g, gw := newGateway([]string{backend.URL}, &router.RoundRobin{})
	defer gw.Close()

	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "POST", gw.URL+"/v1/completions", strings.NewReader(`{"prompt":"x"}`))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	bufio.NewReader(resp.Body).ReadString('\n') // first event arrived; request is in flight
	if g.Backends[0].Inflight() != 1 {
		t.Fatalf("inflight = %d mid-stream, want 1", g.Backends[0].Inflight())
	}
	cancel()
	resp.Body.Close()
	deadline := time.Now().Add(2 * time.Second)
	for g.Backends[0].Inflight() != 0 {
		if time.Now().After(deadline) {
			t.Fatal("inflight never returned to 0 after the client disconnected")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestHealthCheckRestoresBackend(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"status":"ok"}`))
	}))
	defer backend.Close()
	g, gw := newGateway([]string{backend.URL}, &router.RoundRobin{})
	defer gw.Close()
	g.Backends[0].SetHealthy(false)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go g.HealthCheck(ctx, 20*time.Millisecond)
	deadline := time.Now().Add(2 * time.Second)
	for !g.Backends[0].Healthy() {
		if time.Now().After(deadline) {
			t.Fatal("health check never restored the backend")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestAffinityRoutesSamePrefixToSameBackend(t *testing.T) {
	var servers []*httptest.Server
	var urls []string
	for i := 0; i < 3; i++ {
		s := sseBackend(fmt.Sprintf("b%d", i), 1, nil)
		servers = append(servers, s)
		urls = append(urls, s.URL)
	}
	defer func() {
		for _, s := range servers {
			s.Close()
		}
	}()
	_, gw := newGateway(urls, &router.PrefixAffinity{Epsilon: 0.25})
	defer gw.Close()

	system := strings.Repeat("shared system prompt ", 10)
	seen := map[string]bool{}
	for i := 0; i < 10; i++ {
		body := fmt.Sprintf(`{"messages":[{"role":"system","content":%q},{"role":"user","content":"q%d"}]}`, system, i)
		resp, err := http.Post(gw.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		seen[resp.Header.Get("X-Kvserve-Backend")] = true
	}
	if len(seen) != 1 {
		t.Fatalf("requests sharing a prefix went to %d backends, want 1", len(seen))
	}
}
