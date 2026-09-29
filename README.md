# sovereign-rag

**A zero-copy, SIMD-accelerated retrieval engine in async Rust.**
Memory-mapped vector segments, hand-written AVX-512 / AVX2+FMA / NEON distance kernels with
runtime dispatch, a concurrent HNSW index that is searched *directly out of the page cache*,
lock-free RCU snapshot swaps, and a backpressure-aware ingestion pipeline built on custom
lock-free SPSC rings.

> [!NOTE]
> The initial implementation was developed with assistance from Claude Code under my direction.
> I'm now reviewing, testing, benchmarking, and extending the codebase module by module to understand
> and own the system at the engineering level.

```text
$ sovereign bench --dim 1536 --rows 20000 --queries 200        # median of 4 runs (min–max)
Kernels · 1536-d · single core           ns/op             speedup vs naive scalar
  cosine · avx512                        110.6             52.3x   (52.0–60.8x)
  dot    · avx512                         61.1             31.4x   (28.3–32.8x)
Search · 20,000 × 1536-d · k=10
  exact scan (rayon, 4 threads)          p50 3.89 ms       (3.38–5.11 ms)
  hnsw ef=64 (1 thread per query)        p50 286 µs        (277–387 µs)   recall@10 0.991–0.998
  open (mmap + validate)                 45 µs             (42–48 µs, 120 MiB segment)
```

<sub>Measured on one shared 4-vCPU cloud VM (Intel Xeon @ 2.8 GHz, AVX-512); run-to-run variation on this host is up to ~1.5× for latency. See [Benchmarks](#benchmarks) for methodology.</sub>

---

## Table of contents

1. [Architecture](#architecture)
2. [Quick start](#quick-start)
3. [Distance metrics: the math](#distance-metrics-the-math)
4. [SIMD kernels](#simd-kernels)
5. [Memory layout](#memory-layout)
6. [Concurrency model](#concurrency-model)
7. [Durability and integrity](#durability-and-integrity)
8. [Benchmarks](#benchmarks)
9. [Profiling and flamegraphs](#profiling-and-flamegraphs)
10. [Design decisions and trade-offs](#design-decisions-and-trade-offs)
11. [Testing](#testing)
12. [Roadmap](#roadmap)

---

## Architecture

```text
                              ┌──────────────────────────── sovereign-cli ────────────────────────────┐
                              │  ingest · query · info · compact · bench · simd   (clap + indicatif)  │
                              └──────────────┬───────────────────────────────────┬────────────────────┘
                                             │                                   │
            ┌────────────────────── sovereign-pipeline ──────────────┐           │
            │                                                        │           │
 files ───► │ discover ─► reader ─► chunker ─┬─► embed #1 ─┐         │           │
            │ (walk)      (tokio   (zero-    ├─► embed #2 ─┼─► writer│           │
            │             fs)      copy)     └─► embed #N ─┘    │    │           │
            │        └──── bounded lock-free SPSC rings ────┘   │    │           │
            └─────────────────────────────────────────────────── │ ──┘           │
                                                                 ▼               ▼
            ┌────────────────────── sovereign-index ────────────────────────────────────────────┐
            │  SegmentWriter ──(tmp+fsync+rename)──► seg-N.srag ──mmap──► Segment               │
            │  HNSW builder (rayon, per-node locks)                         ├─ flat SIMD scan    │
            │                                                               └─ HNSW (zero-copy)  │
            │  IndexStore: ArcSwap<Snapshot{segments}> · MANIFEST · compaction · writer LOCK    │
            └──────────────────────────────────────────┬────────────────────────────────────────┘
                                                       │
            ┌────────────────────── sovereign-core ────▼────────────────────────────────────────┐
            │  simd: AVX-512 │ AVX2+FMA │ NEON │ scalar  ── one-time cpuid dispatch (&'static)   │
            │  AlignedVec (64 B) · MatrixRef (padded rows) · Arena · TopK · hint · xoshiro256++  │
            └───────────────────────────────────────────────────────────────────────────────────┘
```

| crate | responsibility | key techniques |
|---|---|---|
| [`sovereign-core`](crates/sovereign-core) | math & memory primitives | `#[target_feature]` kernels, masked AVX-512 tails, 1×4 register blocking, 64-byte aligned allocator, bump arena, deterministic top-k |
| [`sovereign-index`](crates/sovereign-index) | storage & search | `#[repr(C, align(64))]` on-disk header, `bytemuck` zero-copy casts, `memmap2` + `madvise`, parallel HNSW build, RCU via `arc-swap`, CRC32 integrity |
| [`sovereign-pipeline`](crates/sovereign-pipeline) | ingestion | custom async SPSC ring (`AtomicWaker`, cache-padded indices), `memchr` chunker over `&str`, sharded fan-out/fan-in, `DashSet` dedup, `spawn_blocking` + arena per batch |
| [`sovereign-cli`](crates/sovereign-cli) | UX | live throughput / RSS progress, latency percentiles, synthetic benchmark |

---

## Quick start

```bash
# Build (release profile: fat LTO, codegen-units=1)
cargo build --release
alias sovereign=./target/release/sovereign

# What SIMD does this machine have?
sovereign simd

# Index a directory (Markdown, text and ~30 source-code extensions)
sovereign ingest ./docs ./src --index .sovereign

# Ask it something
sovereign query -i .sovereign -k 5 "how are wakeups handled when the ring is full?"
sovereign query -i .sovereign --exact --repeat 200 "cache line alignment"   # exact + p50/p99

# Add more data (new segment, published atomically), inspect, compact
sovereign ingest ./more-docs -i .sovereign
sovereign info -i .sovereign
sovereign compact -i .sovereign

# Whole-system synthetic benchmark
sovereign bench --dim 1536 --rows 50000 --queries 500
```

Force a backend with `--simd avx2` or `SOVEREIGN_SIMD=scalar` (useful for A/B comparisons).

> **About the embedder.** The bundled `HashEmbedder` is a deterministic *feature-hashing* model
> (lowercased unigrams + bigrams hashed into signed buckets). It needs no GPU or download, which
> makes the engine demonstrable end to end — but it measures **lexical** overlap, not meaning, and
> hash collisions add noise. Real deployments implement the four-method
> [`Embedder`](crates/sovereign-pipeline/src/embed.rs) trait for ONNX Runtime, candle,
> llama.cpp or an HTTP API. The model's fingerprint is stored in every segment header and checked
> at query time, so an index can never be queried with the wrong model. Pipeline throughput
> numbers below are with the hash embedder; with a transformer, embedding dominates.

### Library usage

```rust
use sovereign_core::Metric;
use sovereign_index::{IndexStore, SearchParams, SegmentConfig, WriteOptions};

let store = IndexStore::open_or_create("./idx", SegmentConfig {
    dim: 1536, metric: Metric::Cosine, fingerprint: my_model.fingerprint(),
})?;

let mut w = store.writer()?;
w.push(42, &embedding, Some("chunk text / metadata"))?;
store.add_segment(&w, &WriteOptions::default())?;      // builds HNSW, fsyncs, swaps atomically

let res = store.search(&query_embedding, &SearchParams::top(10).ef(64))?;
for hit in &res.hits {
    println!("{} {:.3} {:?}", hit.id, hit.score, res.payload(hit)?); // payload borrowed from mmap
}
```

---

## Distance metrics: the math

For vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$:

$$
\text{dot}(\mathbf{a},\mathbf{b}) = \sum_{i=1}^{d} a_i b_i
\qquad
\text{cos}(\mathbf{a},\mathbf{b}) = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2}\,\sqrt{\sum_i b_i^2}}
\qquad
\text{L2}^2(\mathbf{a},\mathbf{b}) = \sum_{i=1}^{d} (a_i - b_i)^2
$$

**Unified scoring.** Every search *maximizes* a score $s$; HNSW *minimizes* a distance
$\delta = -s$, so both views agree by construction:

| metric | score $s(\mathbf{q},\mathbf{x})$ | distance $\delta$ | stored vectors |
|---|---|---|---|
| cosine | $\hat{\mathbf{q}} \cdot \hat{\mathbf{x}}$ | $-\hat{\mathbf{q}} \cdot \hat{\mathbf{x}}$ | $\hat{\mathbf{x}} = \mathbf{x}/\lVert\mathbf{x}\rVert$ |
| dot | $\mathbf{q}\cdot\mathbf{x}$ | $-\mathbf{q}\cdot\mathbf{x}$ | as-is |
| L2 | $-\lVert\mathbf{q}-\mathbf{x}\rVert^2$ | $\lVert\mathbf{q}-\mathbf{x}\rVert^2$ | as-is |

**Normalize once, dot forever.** Since
$\cos(\mathbf{q},\mathbf{x}) = \hat{\mathbf{q}}\cdot\hat{\mathbf{x}}$, vectors are L2-normalized at
ingest and queries once per search. The hot loop then runs a *single-accumulator* dot product
(2 loads + 1 FMA per 16 lanes) instead of the fused cosine (2 loads + 3 FMAs + a division). In the
Criterion suite this is the `prenormalized_dot` vs `fused` comparison.

**Fused cosine** (used when vectors are not pre-normalized) makes one pass with three accumulator
sets, and finishes with a single square root:

$$
\cos = \frac{D}{\sqrt{N_a N_b}}, \quad D=\sum a_ib_i,\; N_a=\sum a_i^2,\; N_b=\sum b_i^2,
\qquad \cos := 0 \text{ if } N_aN_b \le \texttt{f32::MIN\_POSITIVE}
$$

**Zero padding is exact.** Rows are padded from $d$ to $d' = \lceil d/16 \rceil \cdot 16$ with
zeros. For all three metrics the padding contributes $0\cdot0 = 0$ or $(0-0)^2 = 0$, so the kernels
can run branch-free over $d'$ lanes with no scalar tail and no masks.

**Floating-point accuracy.** A naive left-to-right sum has a worst-case error bound of
$\gamma_{d}\sum|a_ib_i|$ with $\gamma_n \approx n\varepsilon$. With $k$ independent SIMD
accumulators of $L$ lanes, each partial sum only sees $d/(kL)$ terms before the final tree
reduction, tightening the bound to roughly $(d/(kL) + \log_2(kL))\,\varepsilon\sum|a_ib_i|$ —
for $d=1536$, AVX-512 with $k=4$: 24 + 6 = 30 terms of error growth instead of 1536. The kernel
tests check every backend against an `f64` reference with tolerance $10^{-5}\sum|a_ib_i| + 10^{-6}$.

**HNSW levels.** A node's top layer is drawn as $\ell = \lfloor -\ln(U)\cdot m_L \rfloor$ with
$U \sim \mathcal{U}(0,1]$ and $m_L = 1/\ln M$, giving $P(\ell \ge l) = M^{-l}$. Expected search
cost is $O(\log n)$ distance evaluations versus $O(n)$ for a flat scan — the only way to get
sub-millisecond latency at scale, because a flat scan is bound by memory bandwidth
(1M × 1536-d × 4 B = 6.1 GB per query).

---

## SIMD kernels

Source: [`crates/sovereign-core/src/simd`](crates/sovereign-core/src/simd).

| backend | ISA | registers | tail handling | loop body (dot) |
|---|---|---|---|---|
| `avx512` | AVX-512F | 32 × 512-bit | masked load `vmovups zmm{k}{z}` | 4 acc × 16 lanes = 64 floats/iter |
| `avx2` | AVX2 + FMA | 16 × 256-bit | scalar (≤ 7 elements) | 4 acc × 8 lanes = 32 floats/iter |
| `neon` | ASIMD | 32 × 128-bit | scalar (≤ 3 elements) | 4 acc × 4 lanes = 16 floats/iter |
| `scalar` | baseline | — | — | 8 independent accumulators (LLVM packs them into SSE2 `mulps`/`addps`; checked by disassembling the release binary) |

**Why four accumulators?** `vfmadd231ps` has ~4 cycles latency and 2 issue ports, so one
accumulator chain would leave the FMA units ~87% idle. But a dot product needs 2 loads per FMA and
cores sustain 2 loads/cycle, so the loop is *load-bound* at 1 FMA/cycle: 4 chains are enough to
cover the latency. Fused cosine (3 FMAs per 2 loads) is FMA-bound instead, so it uses a 2× unroll
with 6 accumulators.

**1×4 register blocking.** `dot_x4` scores one query against four rows: each query load feeds four
FMAs (5 loads per 4 FMAs instead of 8), which lifts the load-port ceiling when rows are cache
resident. The flat scan uses it for cosine/dot.

**Masked tails (AVX-512).** For `n mod 16 = r ≠ 0` the last iteration uses `k = (1 << r) − 1`
with `_mm512_maskz_loadu_ps`; masked-off lanes are neither read nor faulted on, so there is no
out-of-bounds access even when the slice ends at a page boundary.

**Dispatch.** Binaries are compiled for the baseline ISA. The first `kernels()` call runs `cpuid`
and installs a `&'static Kernels` table in a `OnceLock`; hot loops fetch it once and call through
a function pointer. That indirect call is expected to be negligible next to a ~60–100 ns 1536-d
kernel, but the dispatch overhead itself is not separately benchmarked. A
`&'static Kernels` can only be obtained for a backend the CPU supports (private fields + checked
constructors), which is the invariant that makes the `unsafe` calls behind the safe API sound.

**Stable-Rust branch hints.** `core::intrinsics::likely` is permanently unstable, so
[`hint.rs`](crates/sovereign-core/src/hint.rs) uses the `hashbrown` technique: the unlikely side of
a branch calls a `#[cold]` function, which LLVM uses to lay the block out of line. Error/panic
formatting paths are `#[cold] #[inline(never)]` so they never pollute the hot loop's I-cache.

**Unsafe policy.** `unsafe_op_in_unsafe_fn` is denied workspace-wide and every `unsafe` block has
a `// SAFETY:` comment (enforced by `clippy::undocumented_unsafe_blocks`). Since Rust 1.87 the
arithmetic intrinsics are safe inside `#[target_feature]` functions, so only the pointer loads
sit inside `unsafe` blocks.

---

## Memory layout

### Segment file (`.srag`, format v1)

```text
 offset 0      ┌────────────────────────────────────────────┐
               │ SegmentHeader  (192 B = 3 cache lines)     │  #[repr(C, align(64))], Pod
 4096          ├────────────────────────────────────────────┤  ◄── every section is 4 KiB aligned:
               │ vectors   count × stride × f32             │      madvise-able (per 4 KiB page),
               │           (rows padded to 64 B, L2-normed) │      naturally aligned casts
 align 4096    ├────────────────────────────────────────────┤
               │ ids       count × u64                      │
 align 4096    ├────────────────────────────────────────────┤
               │ payload offsets  (count + 1) × u64         │  optional
 align 4096    ├────────────────────────────────────────────┤
               │ payload data     UTF-8                     │  optional
 align 4096    ├────────────────────────────────────────────┤
               │ graph: GraphHeader (128 B) + adjacency     │  optional
               └────────────────────────────────────────────┘
```

### Header (192 bytes, no implicit padding — verified by `bytemuck::Pod` at compile time)

```text
 byte  0        8      12     16    20      24      32      40            48
       ├ magic ─┼ ver ─┼ met ─┼ dim ┼ stride┼ flags ┼ count ┼ fingerprint ─┤
       48 ─ vectors{off,len} ─ 64 ─ ids ─ 80 ─ pay_off ─ 96 ─ pay_data ─ 112 ─ graph ─ 128
      128 ─ body_crc32 ─ 132 ─ header_crc32 ─ 136 ─────────── reserved (zero) ─────────── 192
```

### Padded rows and cache lines

```text
 dim = 1536 → stride 1536 → 6144 B = 96 cache lines = 96 ZMM loads, zero waste
 dim =  100 → stride  112 →  448 B =  7 cache lines (89.3% lane efficiency)

 row i (64-byte aligned):
 |◄──────── line 0 ────────►|◄──────── line 1 ────────►|         |◄──── line 6 ────►|
 [ x0 … x15   (one zmm)     ][ x16 … x31  (one zmm)    ]  …      [ x96 … x99 0 … 0  ]
```

Every full-width load touches exactly one cache line. The `alignment_1536` Criterion group
measures what happens when you break this (a 4-byte offset makes every ZMM load a split load).

### HNSW adjacency (fixed-width rows, zero-copy from mmap)

```text
 levels      : [u8; count]                           top layer per node
 layer0      : [u32; count × (1 + m0)]               [len, n0, n1, …, n(m0−1)] per node
 upper_index : [u32; count]                          first upper row of node (u32::MAX = none)
 upper       : [u32; rows × (1 + m)]                 layers 1..=level, node-contiguous
```

A neighbor lookup is two multiplies and one bounds-checked slice — no pointers, no per-node
allocations, and layer 0 (the only layer searched with the full `ef` beam) is a single dense array. Every id
read from disk is bounds-checked before use, so a corrupt file yields wrong results, never UB.

### Ring buffer (pipeline transport)

```text
  head: CachePadded<AtomicUsize>   ◄─ consumer core owns this line (128 B padded)
  tail: CachePadded<AtomicUsize>   ◄─ producer core owns this line
  buf:  [UnsafeCell<MaybeUninit<T>>; 2^k]    slot = index & (cap − 1)
  + per-side cached copy of the other index (re-read only when the cache says full/empty)
```

128-byte padding (not 64) because Intel's L2 spatial prefetcher pulls cache lines in adjacent
pairs; 64-byte padding still false-shares on those cores.

---

## Concurrency model

### Readers vs. writers: RCU with `ArcSwap`

```text
      readers (any thread, never block)                    writer (serialized)
 ┌───────────────────────────────────────┐   ┌──────────────────────────────────────────┐
 │ let snap = store.load();   // ArcSwap │   │ build HNSW + write seg-N.srag  (no lock) │
 │ snap.search(q)             // no lock │   │ mmap + validate                          │
 │ drop(snap)                            │   │ ── lock publish mutex ──                 │
 └───────────────────────────────────────┘   │ write MANIFEST (tmp + fsync + rename)    │
                     │                        │ current.store(Arc::new(next_snapshot))   │◄ atomic
                     ▼                        └──────────────────────────────────────────┘
  ArcSwap<Snapshot { generation, segments: Vec<Arc<Segment>> }>
```

* `load()` is a few atomic operations with no locks; readers are never blocked by writers and
  writers never wait for readers.
* A `Segment` is unmapped only when the last `Arc` to it drops. A reader holding an old snapshot
  keeps searching it safely even after compaction *deletes* the file — on POSIX an unlinked file's
  inode lives until the last `munmap`.
* The expensive parts of a write (graph build, file I/O) happen outside the publish lock, so
  ingestion and compaction overlap; compaction merges only the segments it started from and keeps
  anything published concurrently.
* An advisory `flock` on `LOCK` enforces a single writer *process*; any number of reader processes
  can map the same store and share one page-cache copy.

The test `readers_never_block_or_fail_during_swaps` runs 4 reader threads in a tight search loop
while the writer publishes 6 segments and compacts 3 times, asserting monotonic generations and
zero errors.

### Parallel HNSW construction

* One `parking_lot::Mutex<Vec<u32>>` per `(node, layer)`; readers copy the list (≤ 64 ids) and
  release the lock *before* computing any distance.
* A thread holds at most one adjacency lock at a time, and the entry-point lock is always taken
  first → no lock-order cycle, no deadlock.
* Inserts that raise the graph's top layer hold the entry-point lock for their duration (rare:
  probability ≈ 1/M per level) — the same scheme as hnswlib.
* Two-phase insert: a node's own lists on every layer are set *before* any back-link to it is
  published, so no concurrent insert can reach a half-built node or have its link to it erased
  ([ADR 0001](docs/adr/0001-hnsw-concurrent-insert-publication.md), issue #2).
* `parallel: false` gives a bit-for-bit deterministic graph for a given seed (tested).

### Pipeline backpressure

Every stage edge is a bounded SPSC ring. When embedding is the bottleneck, worker rings fill, the
chunker parks on `send().await`, the reader parks, and file reads stop — memory stays bounded no
matter how large the corpus. Wakeups use `AtomicWaker` with register-then-recheck, so a wakeup
can never be lost between a failed `try_send` and parking. Memory ordering: slot write →
`tail.store(Release)`; `tail.load(Acquire)` → slot read (and symmetrically for `head`).

---

## Durability and integrity

| mechanism | protects against |
|---|---|
| write to `.tmp` → `fsync` → `rename` → `fsync(dir)` | torn segments / manifests after a crash |
| header CRC32, checked on every open (O(1)) | corrupted or foreign headers |
| body CRC32, `Segment::verify()` / `OpenOptions::verify_checksum` | bit rot in vectors/graph/payloads |
| manifest CRC32 + strict parser | tampered or truncated manifests |
| embedder fingerprint in header + manifest | querying with a different model |
| bounds checks on every file-derived index | UB from malicious or corrupt files |

**The `mmap` safety contract.** `Mmap::map` is `unsafe` because a mapped file that is modified or
truncated underneath the process changes "immutable" memory or raises `SIGBUS`. Segments are
immutable by protocol: written once under a temp name, published by atomic rename, never opened
for writing again. Replacing a segment creates a new inode, so existing mappings are unaffected.
External tampering with published files is outside the contract — the same stance as LMDB.

---

## Benchmarks

### Criterion micro-benchmarks

```bash
cargo bench -p sovereign-core --bench simd_bench                  # all groups
cargo bench -p sovereign-core --bench simd_bench -- cosine_1536   # one group
open target/criterion/report/index.html                           # HTML report
```

Criterion point estimates (50 samples per benchmark, 2 s measurement, 0.5 s warm-up) from **one run**
on a 4-vCPU cloud VM (Intel Xeon @ 2.8 GHz, AVX-512, shared host). Criterion flagged 6–16% of
samples as outliers and the run was not repeated, so treat differences below ~10% as noise.
Throughput counts both operand vectors (`2 × 1536 × 4 B` per call). Reproduce with
`cargo bench -p sovereign-core --bench simd_bench`; `sovereign simd` prints the detected backend.
These tables were measured with rustc 1.94.1. The scalar kernels were later changed from
`chunks_exact` to `as_chunks` (a new clippy lint in 1.98); a same-compiler A/B on 1.98.1 showed no
consistent effect (−4.7% … +7.0%; the same `scalar::dot` function measured +0.7% and +7.0% in two
groups, i.e. within this host's noise). The SIMD kernels were not changed.

**Cosine similarity, 1536-d** — the headline comparison:

| implementation | time / call | throughput | vs naive |
|---|---:|---:|---:|
| scalar naive, 3 passes (`cosine_naive`) | 5.590 µs | 2.0 GiB/s | 1.0× |
| scalar, fused, 8 accumulators | 1.280 µs | 8.9 GiB/s | 4.4× |
| AVX2 + FMA, fused | 159.1 ns | 72 GiB/s | 35.1× |
| **AVX-512, fused** | **105.6 ns** | **108 GiB/s** | **52.9×** |
| AVX2, pre-normalized dot | 87.5 ns | 131 GiB/s | 63.9× |
| **AVX-512, pre-normalized dot** *(what the index runs)* | **56.9 ns** | **201 GiB/s** | **98.3×** |

**Dot product and squared L2, 1536-d:**

| backend | dot | speedup | L2² | speedup |
|---|---:|---:|---:|---:|
| scalar naive | 1.874 µs | 1.0× | 1.895 µs | 1.0× |
| scalar (8 acc) | 285.5 ns | 6.6× | 335.0 ns | 5.7× |
| AVX2 + FMA | 87.9 ns | 21.3× | 109.2 ns | 17.4× |
| AVX-512 | 59.2 ns | 31.6× | 73.8 ns | 25.7× |

**Dimension sweep (cosine, naive vs AVX-512):** the speedup grows with `d` as fixed per-call costs
(horizontal reduction, final `sqrt`/division) are amortized:

| d | 128 | 384 | 768 | 1024 | 1536 | 3072 |
|---|---:|---:|---:|---:|---:|---:|
| naive | 400 ns | 1.31 µs | 2.80 µs | 3.68 µs | 5.54 µs | 11.3 µs |
| AVX-512 | 21.5 ns | 42.4 ns | 66.4 ns | 82.9 ns | 104.4 ns | 192.0 ns |
| speedup | 18.6× | 30.9× | 42.2× | 44.5× | 53.1× | 58.9× |

**Why alignment matters** (`alignment_1536`, AVX-512 dot): 64-byte aligned operands **65.0 ns**
vs 4-byte misaligned **91.6 ns** — every ZMM load becomes a cache-line split, costing **41%**.
This is the measured justification for `AlignedVec` and 64-byte padded rows.

**Register blocking** (`scan_1536`, one query vs N rows):

| rows | working set | row-at-a-time | 1×4 blocked | gain |
|---|---:|---:|---:|---:|
| 256 | 1.5 MiB (L2) | 48.5 µs (30 GiB/s) | 40.9 µs (36 GiB/s) | 1.19× |
| 4096 | 24 MiB (LLC/DRAM) | 2.24 ms (10.5 GiB/s) | 1.64 ms (14.3 GiB/s) | 1.36× |

A plausible explanation for the larger gain on the DRAM-bound case (not verified with PMU counters)
is memory-level parallelism: four independent row
streams keep more cache misses in flight than one.

**Top-k selection:** 1M candidates through `TopK` in ~1.01 ms for k = 10 and k = 100 —
**≈1 ns per candidate**, since almost every candidate is rejected by one predictable comparison.

### End-to-end (`sovereign bench`)

Command: `sovereign bench --dim 1536 --rows 20000 --queries 200` (default seed 42, release
profile), run **4 times**; values are median (min–max). Queries are issued serially from one
thread: the exact scan parallelizes internally with rayon (4 threads), each HNSW query uses one
core. Before timing, one untimed exact scan and one untimed graph search warm the page cache.
Recall@10 is measured against the exact scan of the same segment; it varies between runs because
the parallel graph build is not deterministic.

| metric | median | min – max |
|---|---:|---:|
| HNSW build (20,000 × 1536-d, m=16, ef_construction=100, 4 threads) | 2.04 s | 2.00 – 2.13 s |
| open (mmap + validate, 120 MiB segment) | 45.3 µs | 42.1 – 47.9 µs |
| exact scan p50 / p99 | 3.89 / 6.16 ms | 3.38–5.11 / 5.71–7.73 ms |
| HNSW ef=32 p50 / p99 | 250 / 411 µs | 244–325 / 315–533 µs |
| HNSW ef=32 recall@10 | 0.976 | 0.972 – 0.978 |
| HNSW ef=64 p50 / p99 | 286 / 431 µs | 277–387 / 372–551 µs |
| HNSW ef=64 recall@10 | 0.997 | 0.991 – 0.998 |
| HNSW ef=128 p50 / p99 | 370 / 540 µs | 345–455 / 460–744 µs |
| HNSW ef=128 recall@10 | 0.998 | 0.995 – 0.999 |

* **Opening does O(1) work by construction:** only the header (and the graph header, when
  present) is read and validated; everything else is paged in lazily. Only one file size
  (120 MiB) was measured.
* **HNSW vs exact:** ~13.6× lower median p50 at recall ≈ 0.997, on a *single* core per query,
  while the exact scan uses all 4 cores.
* Recall isolates index quality from embedding quality. The data is synthetic (64 Gaussian
  clusters); recall on real embeddings has not been measured here.

**Ingestion stress test** — the local cargo registry sources. The corpus is machine-specific (it
depends on which crates happen to be cached), so these numbers are illustrative, not reproducible
byte-for-byte. Files were already in the page cache. Single run.

```bash
sovereign ingest ~/.cargo/registry/src --index /tmp/idx --no-sync
sovereign query -i /tmp/idx -k 3 --repeat 500 "memory mapped file advise random access"
sovereign query -i /tmp/idx -k 3 --repeat 20 --exact "memory mapped file advise random access"
```

| metric | value |
|---|---|
| input | 9,452 files · 126.0 MiB of Rust/TOML/Markdown |
| chunks | 94,511 (11,997 exact duplicates dropped before embedding) |
| pipeline wall time | 1.60 s → 78.9 MiB/s, ~59k chunks/s (hash embedder, 4 workers) |
| HNSW build | 82,514 × 384-d in 14.3 s (m = 16, ef_construction = 200, 4 threads) |
| query latency (500 repeats of **one** query) | p50 144 µs / p99 368 µs (HNSW) |
| exact scan (20 repeats of the same query) | p50 4.57 ms |
| peak RSS (`VmHWM`) | 303.6 MiB |

With a transformer embedder the pipeline is bound by the model, not by I/O, chunking or
indexing; this run only shows that the rest of the engine adds little overhead around the model.

---

## Profiling and flamegraphs

The `profiling` Cargo profile is `release` + full debug info:

```bash
# 1. Install tooling (Linux)
cargo install flamegraph            # wraps `perf record` + inferno
sudo sysctl kernel.perf_event_paranoid=1

# 2. Flamegraph of the end-to-end benchmark (frame pointers give cleaner stacks)
RUSTFLAGS="-C force-frame-pointers=yes" \
  cargo flamegraph --profile profiling -p sovereign-cli --bin sovereign -o flamegraph.svg -- \
  bench --rows 50000 --queries 2000

# 3. Flamegraph of one Criterion group
RUSTFLAGS="-C force-frame-pointers=yes" \
  cargo flamegraph --bench simd_bench -p sovereign-core -o simd.svg -- --bench cosine_1536

# 4. Ingestion pipeline
cargo flamegraph --profile profiling -p sovereign-cli --bin sovereign -o ingest.svg -- \
  ingest ~/src --index /tmp/idx --no-sync
```

Without `cargo-flamegraph`:

```bash
perf record -F 999 -g --call-graph dwarf ./target/profiling/sovereign bench --rows 50000
perf script | inferno-collapse-perf | inferno-flamegraph > flamegraph.svg
```

**Measuring SIMD utilization for real** (Intel PMU counters — names vary by generation, see
`perf list | grep fp_arith`). These commands were **not** run for this README: PMU counters are
usually not exposed inside cloud VMs, including the one used for the numbers above.

```bash
perf stat -e fp_arith_inst_retired.scalar_single,\
fp_arith_inst_retired.256b_packed_single,\
fp_arith_inst_retired.512b_packed_single \
  ./target/release/sovereign bench --rows 20000 --no-hnsw
```

The ratio of packed to scalar FP instructions is the actual vectorization rate; with
`--simd scalar` you should see the 512-bit counter drop to ~0. `perf stat -e cycles,instructions,
L1-dcache-load-misses,LLC-load-misses` shows the flat scan shifting from compute-bound (cache
resident) to memory-bound (DRAM) as `--rows` grows.

**Tips.** Pin with `taskset -c 2`, disable turbo for stable numbers, and look for
`hsum256`/`_mm512_reduce_add_ps` in the flamegraph: if horizontal reductions show up, the
per-call overhead dominates and batching (`dot_x4`) pays off.

---

## Design decisions and trade-offs

**`bytemuck` + `#[repr(C)]` instead of `rkyv`.** Everything on disk is a flat array of fixed-width
scalars. A hand-specified layout with `bytemuck` casts is *truly* zero-copy with an O(1) validation
cost (one 192-byte header, plus the 128-byte graph header when present), no archive format coupling
and no relative-pointer resolution — opening touches a constant number of pages regardless of file
size (measured: 42–48 µs for a 120 MiB segment over 4 runs). rkyv is the right tool for pointer-rich object graphs; this
isn't one.

**Function-pointer dispatch instead of `-C target-cpu=native`.** One portable binary that uses
AVX-512 where available and still runs on a 2013 CPU. The indirect call is noise next to a 1536-d
kernel. (If you only ship to one machine, `RUSTFLAGS="-C target-cpu=native"` also lets LLVM
vectorize everything *else*.)

**SPSC rings instead of `tokio::sync::mpsc`.** The fan-out/fan-in topology only ever needs one
producer and one consumer per edge, so paying for MPSC generality (linked blocks, CAS on the tail)
buys nothing. Sharding into N rings avoids contention by construction.

**`DashMap`/`DashSet` only where threads actually race.** Dedup runs inside N concurrent embed
workers, so it needs a concurrent set; the single chunker stage uses plain locals.

**HNSW heuristic selection (Algorithm 4).** Keeps "highway" edges between clusters instead of
wiring each node only to its densest neighborhood, which the HNSW paper (Malkov & Yashunin, §4)
reports improves recall on clustered data. This repository does not include an ablation.

**Small segments are scanned exactly.** In `Auto` mode, segments under 1024 rows skip the graph and
get 100% recall. The 1024-row threshold is a heuristic; it has not been tuned by benchmark.

**What's intentionally *not* here (yet):** deletes/tombstones, filtered search, product
quantization, a GPU searcher, a tree-sitter chunker and a `ratatui` TUI. The `VectorIndex` trait
and `Embedder` trait are the seams those plug into. The NEON kernels are compile- and
clippy-checked for `aarch64-unknown-linux-gnu` but have **not been executed** by the author; the CI
workflow has a `macos-14` (Apple Silicon) job that runs them. All published numbers are x86_64.

---

## Testing

```bash
cargo test --workspace                          # unit + integration + doc tests
SOVEREIGN_SIMD=scalar cargo test --workspace    # pin dispatch to one backend
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p sovereign-core --lib --target aarch64-unknown-linux-gnu   # NEON path (compile only)
cargo +nightly miri test -p sovereign-core --lib                          # UB checks (no mmap tests)
```

The pre-merge review — unsafe inventory with per-block coverage, Miri results, storage/ring/recall
analysis, and which checks ran locally vs only in CI — is in [`docs/REVIEW.md`](docs/REVIEW.md).

Highlights:

* every SIMD backend vs an `f64` reference for **every tail length 0–200** plus common embedding
  sizes, and `dot_x4` vs single-row kernels;
* parallel flat scan returns *bit-identical* results to a sequential scan (deterministic top-k);
* exact search reproduces an independent `f64` ranking for all metrics; HNSW recall@10 ≥ 0.95 on
  clustered data against that `f64` truth; self-recall > 99%; every node reachable from the entry
  point after oversubscribed (16-thread) parallel builds; deterministic sequential builds;
* file format round-trip; header / body / truncation / magic / manifest corruption; forged headers
  with a *valid* CRC for every structural check; corrupt bodies degrade without panicking; empty
  segments and stores; crash leftovers ignored on recovery;
* RCU stress test: concurrent readers during publishes and compactions;
* SPSC ring: FIFO under wraparound, 200k-message cross-thread stream with backpressure, exactly-once
  drop of undelivered items, close semantics;
* chunker: zero-copy (pointer identity), UTF-8 boundary safety, fence atomicity, section headings;
* end-to-end: files → pipeline → store → query → decoded payload.

---

## Roadmap

- [ ] Tombstone bitmap + filtered search (pre-filter during graph traversal)
- [ ] Scalar (int8) and product quantization with AVX-512 VNNI / `vpdpbusd` kernels
- [ ] ONNX Runtime embedder (`ort`) with dynamic batching
- [ ] tree-sitter chunker for exact AST boundaries
- [ ] `ratatui` dashboard for live pipeline metrics
- [ ] io_uring-based reader stage
- [ ] Hybrid retrieval: BM25 + vector fusion (RRF)

## License

MIT — see [LICENSE](LICENSE).
