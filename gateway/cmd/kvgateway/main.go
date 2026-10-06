// kvgateway routes OpenAI-compatible requests across kvserve instances.
//
//	kvgateway --backends http://localhost:8001,http://localhost:8002 --policy prefix_affinity
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/FrimpsManu/kvserve/gateway/internal/proxy"
	"github.com/FrimpsManu/kvserve/gateway/internal/router"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	listen := flag.String("listen", ":9000", "address to listen on")
	backendList := flag.String("backends", "", "comma-separated kvserve base URLs")
	policyName := flag.String("policy", "prefix_affinity", "round_robin | least_loaded | prefix_affinity")
	epsilon := flag.Float64("epsilon", 0.25, "prefix_affinity load slack: a backend takes new requests while in-flight < ceil((1+epsilon) * average)")
	prefixChars := flag.Int("prefix-chars", 512, "characters of a text prompt used as the routing key")
	prefixTokens := flag.Int("prefix-tokens", 128, "token ids of a pre-tokenized prompt used as the routing key")
	healthEvery := flag.Duration("health-interval", 2*time.Second, "backend health check interval")
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if *backendList == "" {
		log.Error("--backends is required")
		os.Exit(2)
	}
	var backends []*router.Backend
	for _, u := range strings.Split(*backendList, ",") {
		backends = append(backends, router.NewBackend(strings.TrimRight(strings.TrimSpace(u), "/")))
	}
	policy, err := router.NewPolicy(*policyName, *epsilon)
	if err != nil {
		log.Error("bad policy", "err", err)
		os.Exit(2)
	}

	gw := proxy.New(backends, policy, router.PrefixKey{PrefixChars: *prefixChars, PrefixTokens: *prefixTokens}, log)
	mux := gw.Routes()
	mux.Handle("GET /metrics", promhttp.Handler())

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go gw.HealthCheck(ctx, *healthEvery)

	// No write timeout: streaming responses can legitimately run for minutes.
	srv := &http.Server{Addr: *listen, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		srv.Shutdown(shutdown) // drain in-flight streams
	}()
	log.Info("kvgateway listening", "addr", *listen, "policy", policy.Name(), "backends", len(backends))
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Error("server failed", "err", err)
		os.Exit(1)
	}
}
