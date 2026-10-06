package router

import (
	"fmt"
	"hash/fnv"
	"math"
	"sort"
	"sync/atomic"
)

// Policy ranks backends for a request. Order returns candidates in preference
// order: the first is the pick, the rest are failover choices if connecting to
// it fails. key identifies the request's shared prefix (see PrefixKey); policies
// that ignore locality ignore it. Candidates are healthy and never empty.
type Policy interface {
	Name() string
	Order(key uint64, candidates []*Backend) []*Backend
}

// ---- round robin ---------------------------------------------------------------

// RoundRobin spreads requests evenly and ignores both load and locality. It is
// the baseline: every backend ends up computing and caching every shared prefix.
type RoundRobin struct{ next atomic.Uint64 }

func (*RoundRobin) Name() string { return "round_robin" }

func (p *RoundRobin) Order(_ uint64, c []*Backend) []*Backend {
	start := int(p.next.Add(1)-1) % len(c)
	return append(append([]*Backend{}, c[start:]...), c[:start]...)
}

// ---- least loaded --------------------------------------------------------------

// LeastLoaded picks the backend with the fewest requests in flight from this
// gateway. In-flight count is instant and exact for a single gateway, unlike
// polled server-side queue lengths, which lag behind bursts.
type LeastLoaded struct{ tiebreak atomic.Uint64 }

func (*LeastLoaded) Name() string { return "least_loaded" }

func (p *LeastLoaded) Order(_ uint64, c []*Backend) []*Backend {
	out := append([]*Backend{}, c...)
	// Rotate first so ties don't always favour the first backend.
	r := int(p.tiebreak.Add(1)) % len(out)
	out = append(out[r:], out[:r]...)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Inflight() < out[j].Inflight() })
	return out
}

// ---- prefix affinity -----------------------------------------------------------

// PrefixAffinity sends requests sharing a prefix to the same backend so that
// backend's prefix cache gets reused, using rendezvous (highest-random-weight)
// hashing: each backend gets a score hash(key, backend), highest wins. Adding or
// removing a backend only moves the keys that backend owned.
//
// Pure affinity overloads a backend when one prefix is hot, so loads are
// bounded (Mirrokni et al., "Consistent Hashing with Bounded Loads"): a backend
// is eligible only while its in-flight count is below ceil(c * average) where
// c = 1 + Epsilon. Requests walk the rendezvous order to the first eligible
// backend, so a hot prefix spills to its second-choice backend, and so on.
type PrefixAffinity struct {
	Epsilon float64 // load slack over the average, e.g. 0.25
}

func (*PrefixAffinity) Name() string { return "prefix_affinity" }

func (p *PrefixAffinity) Order(key uint64, c []*Backend) []*Backend {
	type scored struct {
		b     *Backend
		score uint64
	}
	ranked := make([]scored, len(c))
	for i, b := range c {
		ranked[i] = scored{b, rendezvousScore(key, b.URL)}
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })

	// Average load including the request being routed.
	var total int64 = 1
	for _, b := range c {
		total += b.Inflight()
	}
	limit := int64(math.Ceil((1 + p.Epsilon) * float64(total) / float64(len(c))))

	eligible := make([]*Backend, 0, len(c))
	overloaded := make([]*Backend, 0)
	for _, s := range ranked {
		if s.b.Inflight() < limit {
			eligible = append(eligible, s.b)
		} else {
			overloaded = append(overloaded, s.b)
		}
	}
	return append(eligible, overloaded...)
}

func rendezvousScore(key uint64, backend string) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for i := range buf {
		buf[i] = byte(key >> (8 * i))
	}
	h.Write(buf[:])
	h.Write([]byte(backend))
	return mix64(h.Sum64())
}

// mix64 is the splitmix64 finaliser; FNV alone distributes poorly over the
// small set of backend names.
func mix64(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

func NewPolicy(name string, epsilon float64) (Policy, error) {
	switch name {
	case "round_robin":
		return &RoundRobin{}, nil
	case "least_loaded":
		return &LeastLoaded{}, nil
	case "prefix_affinity":
		return &PrefixAffinity{Epsilon: epsilon}, nil
	}
	return nil, fmt.Errorf("unknown policy %q (round_robin | least_loaded | prefix_affinity)", name)
}
