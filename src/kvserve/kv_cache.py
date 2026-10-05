"""Paged KV cache: block allocation, block tables and prefix caching.

The physical cache is a pool of fixed-size blocks (`block_size` tokens each). A
sequence owns a block table mapping logical block i (tokens [i*bs, (i+1)*bs)) to a
physical block id, so a sequence never needs contiguous memory and only wastes at
most one partially filled block.

Prefix caching: every *full* block is identified by a hash chained over all tokens
up to and including that block. Blocks with the same hash hold identical K/V, so a
new request whose prompt shares a prefix with earlier requests reuses those blocks
instead of recomputing them. Blocks are reference counted; when the count drops to
zero a block goes to the free queue but keeps its contents and hash, so it can still
be revived by a later hit until it is evicted (LRU) for reuse.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict

from kvserve.sequence import Sequence


def hash_block(parent: bytes | None, token_ids: list[int]) -> bytes:
    h = hashlib.sha256(parent or b"")
    h.update(array("q", token_ids).tobytes())
    return h.digest()


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        self.ref_counts = [0] * num_blocks
        self.block_hash: list[bytes | None] = [None] * num_blocks
        # Blocks with ref_count == 0, oldest first. Insertion order is the LRU order.
        self.free_queue: OrderedDict[int, None] = OrderedDict.fromkeys(range(num_blocks))
        self.cached: dict[bytes, int] = {}  # block hash -> physical block

        self.prefix_queries = 0  # tokens looked up in the prefix cache
        self.prefix_hits = 0  # tokens served from it

    # ---- capacity ------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_queue)

    @property
    def usage(self) -> float:
        return 1.0 - self.num_free_blocks / self.num_blocks

    def num_blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    # ---- prefix cache --------------------------------------------------------------

    def find_cached_prefix(self, seq: Sequence) -> list[int]:
        """Physical blocks holding the longest cached prefix of `seq`'s tokens.

        At least one token is always left uncached: the forward pass must run on the
        last token to produce logits for sampling.
        """
        if not self.enable_prefix_caching:
            return []
        tokens = seq.token_ids
        max_blocks = (len(tokens) - 1) // self.block_size
        self._extend_hashes(seq, max_blocks)
        hits: list[int] = []
        for h in seq.block_hashes[:max_blocks]:
            block = self.cached.get(h)
            if block is None:
                break
            hits.append(block)
        self.prefix_queries += len(tokens)
        self.prefix_hits += len(hits) * self.block_size
        return hits

    def _extend_hashes(self, seq: Sequence, num_full_blocks: int) -> None:
        tokens = seq.token_ids
        bs = self.block_size
        while len(seq.block_hashes) < num_full_blocks:
            i = len(seq.block_hashes)
            parent = seq.block_hashes[-1] if seq.block_hashes else None
            seq.block_hashes.append(hash_block(parent, tokens[i * bs : (i + 1) * bs]))

    def cache_full_blocks(self, seq: Sequence) -> None:
        """Register blocks that became full after a forward pass."""
        if not self.enable_prefix_caching:
            return
        num_full = seq.num_computed_tokens // self.block_size
        self._extend_hashes(seq, num_full)
        for i in range(num_full):
            block = seq.block_table[i]
            if self.block_hash[block] is not None:
                continue
            h = seq.block_hashes[i]
            if h not in self.cached:  # an identical block may already be cached elsewhere
                self.cached[h] = block
                self.block_hash[block] = h

    # ---- allocation ----------------------------------------------------------------

    def allocate_slots(self, seq: Sequence, num_new_tokens: int, cached_blocks: list[int] | None = None) -> bool:
        """Grow `seq`'s block table to hold `num_new_tokens` more tokens.

        `cached_blocks` (prefix-cache hits, only at admission) are attached first and
        count as computed. Returns False, changing nothing, if the pool is too small.
        """
        cached_blocks = cached_blocks or []
        num_cached_tokens = len(cached_blocks) * self.block_size
        total = seq.num_computed_tokens + num_cached_tokens + num_new_tokens
        num_needed = self.num_blocks_for(total) - len(seq.block_table) - len(cached_blocks)
        # Reviving a cached block that sits in the free queue also consumes capacity.
        num_revived = sum(1 for b in cached_blocks if self.ref_counts[b] == 0)
        if num_needed + num_revived > self.num_free_blocks:
            return False

        for block in cached_blocks:
            if self.ref_counts[block] == 0:
                del self.free_queue[block]
            self.ref_counts[block] += 1
            seq.block_table.append(block)
        seq.num_computed_tokens += num_cached_tokens
        seq.num_cached_prompt_tokens += num_cached_tokens

        for _ in range(max(num_needed, 0)):
            seq.block_table.append(self._pop_free_block())
        return True

    def _pop_free_block(self) -> int:
        block, _ = self.free_queue.popitem(last=False)  # least recently freed
        h = self.block_hash[block]
        if h is not None:  # evict from the prefix cache
            if self.cached.get(h) == block:
                del self.cached[h]
            self.block_hash[block] = None
        self.ref_counts[block] = 1
        return block

    def free(self, seq: Sequence) -> None:
        # Free tail blocks first so they are evicted before the shared prefix blocks.
        for block in reversed(seq.block_table):
            self.ref_counts[block] -= 1
            assert self.ref_counts[block] >= 0, f"double free of block {block}"
            if self.ref_counts[block] == 0:
                self.free_queue[block] = None
        seq.block_table = []

    def slot(self, seq: Sequence, position: int) -> int:
        """Flat slot index in the KV pool for token `position` of `seq`."""
        return seq.block_table[position // self.block_size] * self.block_size + position % self.block_size

    @property
    def prefix_hit_rate(self) -> float:
        return self.prefix_hits / self.prefix_queries if self.prefix_queries else 0.0
