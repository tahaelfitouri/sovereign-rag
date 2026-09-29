# ADR 0001: Publication protocol for concurrent HNSW inserts

- **Status:** Proposed
- **Date:** 2026-09-29
- **Issue:** [#2](https://github.com/tahaelfitouri/sovereign-rag/issues/2) (root-cause analysis in
  the issue thread)
- **Code:** `crates/sovereign-index/src/hnsw/build.rs` (`Builder::insert`)

## Context

The parallel HNSW builder inserts nodes from rayon workers. Each `(node, layer)` adjacency list
sits behind its own `parking_lot::Mutex`, and a thread holds at most one adjacency lock at a time.
That makes every individual list read or write atomic, but not a whole insert.

Before this change an insert of `q` handled one layer at a time, top-down, and published each
layer before starting the next. Per layer it searched, selected neighbors, overwrote `q`'s own list
(`clear()` + `extend()`), then added the back-links `nb → q`. So `q` became reachable at its top
layer while its lower lists, including layer 0, were still empty. This caused two failure modes:

- **M1: dead-end start.** Another insert `r` descends into `q` through an upper layer and starts
  its layer-0 search there. `q`'s layer-0 list is still empty, so the beam cannot expand and `r`
  is linked to `q` alone (out-degree 1).
- **M2: clobbered back-link.** `r` adds `r` to `q`'s layer-0 list. Then `q`'s own insert reaches
  layer 0 and overwrites that list with its search result, erasing the link. If that was `r`'s
  only in-edge, `r` is unreachable from the entry point and no query can find it, whatever `ef` is.

Measured on `main` (3000 × 32, cosine, top-1, ef = 64, 10 builds per row): self-recall was 3000
with one thread, 2987–2994 with 4, 2927–2983 with 16 and 2788–2927 with 64. Every miss was a node
that was unreachable from the entry point. An instrumented build attributed nearly all of those
nodes to M2. Sequential builds recorded zero clobbers.

## Decision

Split every insert into two phases:

1. **Link out.** For every layer from `min(level, top)` down to 0: search, select neighbors and set
   `q`'s own list. Record the back-links `(layer, nb)` without applying them.
2. **Publish.** Apply every recorded back-link `nb → q` with the existing `connect` (which prunes
   `nb` with the selection heuristic when `nb` is full).

Before phase 2, no list contains `q`, because `q` can only enter a list through its own `connect`
calls. So no other insert can reach `q` while its lists are incomplete, which rules out M1. After
phase 1, `q` never rewrites its own lists again, so it cannot erase a back-link another insert
added, which rules out M2.

**Sequential equivalence.** The layer-`L` search reads only layer-`L` lists. Deferring the
layer-`L` back-links only delays writes to layer-`L` lists, and the searches of the layers below
never read those. `connect` never reads `q`'s own lists, and phase 2 applies the back-links in the
same order as before (top layer first, each in selection order). So a sequential build produces
exactly the same graph; `single_thread_parallel_build_matches_sequential` checks this.

Lock discipline is unchanged. A thread still holds at most one adjacency lock at a time, and the
entry-point lock is still taken first. The deadlock-freedom argument in the module docs still
holds. The change adds one scratch `Vec<(usize, u32)>` per rayon worker, reused across inserts.

## Alternatives considered

- **Merge instead of overwrite in phase 1** (union `q`'s current list with the new selection, then
  re-prune). This addresses M2 but not M1. In the instrumented A/B it was still insufficient at 64
  threads: self-recall 2923–2979 / 3000. Rejected.
- **Hold `q`'s node lock for the whole insert and make readers take it** (hnswlib's per-element
  lock). This gives the same guarantee, since nobody reads an in-flight node, but it means a thread
  holds its own lock while acquiring others'. That breaks the "at most one adjacency lock" rule,
  needs a new deadlock-freedom argument, blocks readers for the whole insert instead of a
  ~64-element copy, and needs a per-node lock rather than per-`(node, layer)`. Deferred
  publication gets the guarantee without holding any lock longer. Rejected.
- **Post-build repair** (BFS from the entry point, then re-link unreachable nodes). This treats
  the symptom. It adds a whole-graph pass and does nothing for the out-degree-1 nodes from M1 that
  are still reachable. Rejected as the fix. It could still be added later as a defensive check.
- **Fewer threads or a sequential build.** Sequential builds are about 3.7–4× slower (20k × 128:
  7.4–7.7 s vs 1.9–2.0 s). Issue #2 explicitly rules out disabling parallelism to hide the defect.
  Rejected.

## Consequences

Measured with the uninstrumented `main` against this change, runs interleaved on the same 4-vCPU
host:

| | `main` | this change |
|---|---|---|
| sequential graph, all layers (3000 × 32) | baseline | byte-identical (3190 adjacency rows) |
| self-recall, 4 threads (5 builds) | 2988–2993 | 3000 in every build |
| self-recall, 16 threads (5 builds) | 2867–2973 | 3000 in every build |
| self-recall, 64 threads (5 builds) | 2828–2878 | 3000 in every build |
| nodes unreachable from the entry point, 4/16/64 threads | 7–165 | 0 in every build |
| recall@10, 64 threads | 0.9305–0.9485 | 0.9915–0.9940 |
| build time, 20k × 128, 4 threads (4 builds each) | 1849–2376 ms | 1833–2063 ms |
| self-recall, 20k × 128 | 19980–19987 | 20000 in every build |

Build time is within run-to-run noise on this host, so the change has no measurable throughput
cost.

Regression coverage:

- `hnsw::build::tests::parallel_build_has_no_unreachable_nodes` checks reachability from the
  entry point across all layers after 16-thread builds. It failed 20/20 runs before the fix and
  passed 50/50 runs (150 builds) after it.
- `every_vector_finds_itself_with_oversubscribed_parallel_build` (integration) builds with 16
  threads and requires self-recall of at least 2997 / 3000. Before the fix it failed 10/10 runs
  (for example 2961); after it, 20/20 runs returned 3000.
- `single_thread_parallel_build_matches_sequential` pins sequential equivalence.
- The existing thresholds are unchanged.

### Remaining limitations

- **Parallel builds are still non-deterministic.** Inserts that run at the same time cannot see
  each other, so the edges chosen depend on scheduling. This is the normal trade-off for
  concurrent HNSW. Use `parallel = false` when you need a reproducible graph.
- **The first inserts see an almost empty graph.** Early in a parallel build, up to one insert per
  worker links against the few nodes already published, so those nodes start with short out-lists.
  They gain links as later inserts pick them as neighbors. No measured build left such a node
  unreachable.
- **Heuristic pruning can still drop a node's last in-edge.** `connect` re-runs the selection
  heuristic on a full list and can evict an edge that was some node's only in-edge. This is a
  property of HNSW itself, sequential builds included, not of concurrency. The reachability test
  covers the tested configurations but does not guarantee connectivity for every dataset.
- **The evidence is synthetic.** It comes from isotropic Gaussian data (3000 × 32, 20k × 128) and
  the existing clustered recall test. No real-embedding corpus was used.
