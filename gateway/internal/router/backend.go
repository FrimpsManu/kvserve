// Package router chooses which kvserve instance serves each request.
package router

import (
	"sync/atomic"
)

// Backend is one kvserve instance. Fields are updated concurrently by request
// handlers and the health checker, so they are atomics.
type Backend struct {
	URL string // e.g. http://10.0.0.5:8000

	healthy  atomic.Bool
	inflight atomic.Int64 // requests this gateway has open against the backend
}

func NewBackend(url string) *Backend {
	b := &Backend{URL: url}
	b.healthy.Store(true) // optimistic until the first health check says otherwise
	return b
}

func (b *Backend) Healthy() bool      { return b.healthy.Load() }
func (b *Backend) SetHealthy(ok bool) { b.healthy.Store(ok) }
func (b *Backend) Inflight() int64    { return b.inflight.Load() }
func (b *Backend) Acquire()           { b.inflight.Add(1) }
func (b *Backend) Release()           { b.inflight.Add(-1) }
func (b *Backend) String() string     { return b.URL }

// Healthy filters out backends that failed their last health check.
func Healthy(backends []*Backend) []*Backend {
	out := make([]*Backend, 0, len(backends))
	for _, b := range backends {
		if b.Healthy() {
			out = append(out, b)
		}
	}
	return out
}
