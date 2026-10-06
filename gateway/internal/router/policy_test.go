package router

import (
	"fmt"
	"math"
	"testing"
)

func backends(n int) []*Backend {
	out := make([]*Backend, n)
	for i := range out {
		out[i] = NewBackend(fmt.Sprintf("http://10.0.0.%d:8000", i+1))
	}
	return out
}

func TestRoundRobinIsEven(t *testing.T) {
	bs := backends(3)
	p := &RoundRobin{}
	counts := map[*Backend]int{}
	for i := 0; i < 300; i++ {
		counts[p.Order(0, bs)[0]]++
	}
	for _, b := range bs {
		if counts[b] != 100 {
			t.Fatalf("%s got %d of 300 picks, want 100", b, counts[b])
		}
	}
}

func TestLeastLoadedPicksIdlest(t *testing.T) {
	bs := backends(3)
	bs[0].inflight.Store(5)
	bs[1].inflight.Store(1)
	bs[2].inflight.Store(3)
	order := (&LeastLoaded{}).Order(0, bs)
	if order[0] != bs[1] || order[1] != bs[2] || order[2] != bs[0] {
		t.Fatalf("order = %v, want [b1 b2 b0]", order)
	}
}

func TestAffinityIsStable(t *testing.T) {
	bs := backends(4)
	p := &PrefixAffinity{Epsilon: 0.25}
	for key := uint64(0); key < 1000; key++ {
		first := p.Order(key, bs)[0]
		for i := 0; i < 5; i++ {
			if got := p.Order(key, bs)[0]; got != first {
				t.Fatalf("key %d moved from %s to %s with no load change", key, first, got)
			}
		}
	}
}

func TestAffinitySpreadsKeysEvenly(t *testing.T) {
	bs := backends(4)
	p := &PrefixAffinity{Epsilon: 0.25}
	counts := map[*Backend]int{}
	const n = 20000
	for key := uint64(0); key < n; key++ {
		counts[p.Order(mix64(key), bs)[0]]++
	}
	for _, b := range bs {
		share := float64(counts[b]) / n
		if math.Abs(share-0.25) > 0.03 {
			t.Fatalf("%s owns %.1f%% of keys, want ~25%%", b, 100*share)
		}
	}
}

func TestAffinityRemovingBackendOnlyMovesItsKeys(t *testing.T) {
	bs := backends(4)
	p := &PrefixAffinity{Epsilon: 0.25}
	before := map[uint64]*Backend{}
	for key := uint64(0); key < 5000; key++ {
		before[key] = p.Order(mix64(key), bs)[0]
	}
	removed := bs[2]
	remaining := []*Backend{bs[0], bs[1], bs[3]}
	for key, owner := range before {
		after := p.Order(mix64(key), remaining)[0]
		if owner != removed && after != owner {
			t.Fatalf("key %d moved from %s to %s although its owner is still up", key, owner, after)
		}
	}
}

func TestAffinityBoundedLoadSpillsHotPrefix(t *testing.T) {
	bs := backends(3)
	p := &PrefixAffinity{Epsilon: 0.25}
	const key = 42
	preferred := p.Order(key, bs)
	preferred[0].inflight.Store(10) // the owner of this prefix is swamped
	order := p.Order(key, bs)
	if order[0] != preferred[1] {
		t.Fatalf("hot prefix went to %s, want its second choice %s", order[0], preferred[1])
	}
	if order[len(order)-1] != preferred[0] {
		t.Fatalf("overloaded owner should be the last-resort failover, got order %v", order)
	}
}

func TestAffinityUsesOwnerWhenBalanced(t *testing.T) {
	bs := backends(3)
	p := &PrefixAffinity{Epsilon: 0.25}
	for _, b := range bs {
		b.inflight.Store(4)
	}
	owner := p.Order(7, []*Backend{bs[0], bs[1], bs[2]})[0]
	// limit = ceil(1.25 * 13 / 3) = 6 > 4: everyone is eligible, so affinity wins.
	if got := p.Order(7, bs)[0]; got != owner {
		t.Fatalf("balanced load should keep affinity: got %s want %s", got, owner)
	}
}

func TestNewPolicy(t *testing.T) {
	for _, name := range []string{"round_robin", "least_loaded", "prefix_affinity"} {
		p, err := NewPolicy(name, 0.25)
		if err != nil || p.Name() != name {
			t.Fatalf("NewPolicy(%q) = %v, %v", name, p, err)
		}
	}
	if _, err := NewPolicy("random", 0); err == nil {
		t.Fatal("expected error for unknown policy")
	}
}
