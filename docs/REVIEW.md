# Pre-merge review: sovereign-rag

This document records the pre-merge review of the initial `sovereign-rag` workspace: what was
checked, how, what was found and fixed, and what risks remain. It is a review artifact, **not** a
claim of production readiness.

- Reviewed commit range: the full initial import (single root commit) plus the fixes listed in
  [§3](#3-findings-and-fixes-in-this-pr).
- Host for all local runs: 4-vCPU cloud VM, Intel Xeon @ 2.8 GHz, AVX-512F/AVX2/FMA, Linux 6.18,
  x86_64. Toolchains: `rustc 1.94.1` for the review runs, then **all gates re-run on `rustc 1.98.1`**
  (the CI `stable`) after finding F8; MSRV `1.89.0`; nightly `2026-09-23` (Miri only).

## Contents

1. [Verification matrix: local vs cross-compiled vs CI-only](#1-verification-matrix)
2. [Commands and results](#2-commands-and-results)
3. [Findings and fixes in this PR](#3-findings-and-fixes-in-this-pr)
4. [Unsafe code inventory](#4-unsafe-code-inventory)
5. [Storage: mmap, validation, CRC, atomic rename, recovery](#5-storage-review)
6. [SPSC ring: ordering, ownership, shutdown](#6-spsc-ring-review)
7. [HNSW recall evaluation and exact ground truth](#7-hnsw-recall-and-ground-truth)
8. [Races, UB, overflow, malformed files, empty inputs](#8-races-ub-overflow-malformed-files-empty-inputs)
9. [README performance claims](#9-readme-performance-claims)
10. [Remaining risks](#10-remaining-risks)

---

## 1. Verification matrix

| check | where it ran | status |
|---|---|---|
| `cargo fmt --all --check` | local | pass |
| `cargo check --workspace --all-targets` (stable 1.94.1) | local | pass, 0 warnings |
| `cargo clippy --workspace --all-targets -- -D warnings` | local | pass on 1.98.1 and 1.94.1 |
| `cargo test --workspace` (dispatch: auto → AVX-512) | local | 81 passed, 0 failed |
| `cargo test --workspace` with `SOVEREIGN_SIMD=avx2` / `scalar` / `avx512` | local | pass (each) |
| `cargo doc --workspace --no-deps` | local | pass, 0 warnings |
| `cargo +1.89 check --workspace --all-targets` (MSRV) | local | pass |
| `cargo llvm-cov --workspace` (line coverage per unsafe block) | local | see [§4](#4-unsafe-code-inventory) |
| Miri: `sovereign-core` lib tests (scalar dispatch) | local | pass — 30/30 in one run, plus the 2 tests added/changed during review in a second run; no UB |
| Miri: `sovereign-core` SIMD tests with `+avx2,+fma` / `+avx512f` | local | pass — 6/6 kernel tests per build; a probe confirmed Miri really dispatched the AVX2 and AVX-512 kernels (incl. the masked tail) |
| Miri: SPSC ring tests, `-Zmiri-many-seeds=0..16` | local | pass, no UB / data race reported |
| Miri: async ring shutdown test (tokio), 4 seeds | local | pass |
| Miri: `sovereign-index` HNSW unit tests (no mmap) | local | pass with caveats — see note below |
| Criterion `simd_bench`, `sovereign bench`, ingest stress run | local | numbers in README |
| NEON kernels: `cargo clippy -p sovereign-core --lib --target aarch64-unknown-linux-gnu` | **cross-compiled only** | pass (compile/lint only — never executed) |
| NEON kernels executed on Apple Silicon (`macos-14` job) | **CI-only** | configured in `.github/workflows/ci.yml`; not observed at time of writing |
| Ubuntu CI matrix, MSRV job, bench-compile job | **CI-only** (mirrors the local runs above) | not observed at time of writing |

**Miri note for `sovereign-index`.** Under the default Stacked Borrows model: the sequential HNSW
build tests, the in-memory graph encode → parse → search test and the small flat-scan test pass
with no UB. The *parallel* HNSW build test aborts with a Stacked Borrows violation reported
**inside `crossbeam-epoch` 0.9.21** (`Local::element_of`, reached from rayon's work-stealing
deque), not in this workspace; the same test passes under Tree Borrows
(`-Zmiri-tree-borrows -Zmiri-ignore-leaks`, 4 seeds, no UB; `ignore-leaks` because rayon's global
pool threads are never joined). The large parallel flat-scan test (≥ 4M floats) exceeded the Miri
time budget and was not completed. Anything touching `mmap` cannot run under Miri at all.

Not done at all: fuzzing of the file parser, loom model checking, sanitizers (ASan/TSan), runs on
32-bit targets, Windows or big-endian hosts (the latter is rejected at compile time).

## 2. Commands and results

```bash
cargo fmt --all --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
for s in avx2 scalar avx512; do SOVEREIGN_SIMD=$s cargo test --workspace; done
cargo doc --workspace --no-deps
cargo +1.89 check --workspace --all-targets
cargo clippy -p sovereign-core --lib --target aarch64-unknown-linux-gnu

# coverage (requires cargo-llvm-cov + llvm-tools-preview)
cargo llvm-cov --workspace --text --output-path cov.txt

# Miri (nightly + miri component). mmap-backed tests cannot run under Miri.
cargo +nightly miri test -p sovereign-core --lib
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo +nightly miri test -p sovereign-core --lib -- simd::
RUSTFLAGS="-C target-feature=+avx512f"   cargo +nightly miri test -p sovereign-core --lib -- simd::
MIRIFLAGS="-Zmiri-many-seeds=0..16" cargo +nightly miri test -p sovereign-pipeline --lib -- \
  ring::tests::std_threads ring::tests::fifo ring::tests::close ring::tests::undelivered ring::tests::capacity
MIRIFLAGS="-Zmiri-many-seeds=0..4" cargo +nightly miri test -p sovereign-pipeline --lib -- ring::tests::send_fails
cargo +nightly miri test -p sovereign-index --lib -- hnsw:: --skip parallel_build
cargo +nightly miri test -p sovereign-index --lib -- flat::tests::small
MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-ignore-leaks -Zmiri-many-seeds=0..4" \
  cargo +nightly miri test -p sovereign-index --lib -- hnsw::build::tests::parallel_build
```

## 3. Findings and fixes in this PR

Only confirmed defects were changed; each fix is minimal and has a regression test.

| # | finding | severity | fix | regression test |
|---|---|---|---|---|
| F1 | A caller-controlled `k` sized allocations directly: `SearchParams::top(usize::MAX)` panicked in `Vec::with_capacity`, and `k = 1<<40` would attempt a ~16 TB allocation and abort | medium (DoS on untrusted `k`) | clamp `k` to the row count in `Segment::search`, `Segment::search_into`, `Snapshot::search_with` | `malformed_tests::empty_store_searches_and_huge_k_is_clamped` |
| F2 | `sovereign ingest` printed "peak rss" from `VmRSS` (current), not `VmHWM` (peak); the README inherited the mislabel | low (misreported metric) | read `VmHWM` for the peak | manual (`/proc` read); README number re-measured |
| F3 | `sovereign bench --queries 0` panicked on `queries[0]` | low | reject zero `--dim/--rows/--queries/-k` with an error | manual |
| F4 | Invalid HNSW params (e.g. `--m 1`) were only rejected at graph-build time, after the whole pipeline ran (and silently never on empty input) | low | validate before running the pipeline | manual |
| F5 | Bench warm-up only touched vector pages; the first timed HNSW configuration paid graph page faults | low (biased p99) | one untimed graph query before timing | re-measured numbers |
| F6 | **Parallel HNSW build could create self-loops**: a concurrent insert can link `q` into a list reachable from `q`'s own layer search (beams from the layer above are reused as entry points, and distances are layer-independent), so `q` was selected as its own neighbor. Not a memory-safety issue (search skips visited nodes), but each self-loop wastes an adjacency slot and lowers the effective degree | medium (graph quality / recall) | filter `q` out of its own candidates; refuse `connect(q, q)` | `hnsw::build::tests::parallel_build_links_are_valid` (failed 14/300 runs before the fix, 0/1000 after) |
| F7 | `sovereign ingest --no-sync` labelled its write time "write + fsync" | low (misreported metric) | label reflects the actual mode | manual |
| F8 | **CI `fmt + clippy` job failed** on the PR: CI's floating `stable` is rustc 1.98, whose clippy added `chunks_exact_to_as_chunks`; the local review toolchain (1.94.1) lacked it, and `-D warnings` made it fatal (6 sites in `simd/scalar.rs`, 1 in the Criterion bench) | CI break | switch to `slice::as_chunks` / `as_chunks_mut` (stable since 1.88, within MSRV). Clippy now clean on 1.98.1 **and** 1.94.1; scalar loops still compile to packed SSE (`mulps`/`addps`, re-checked by disassembly); same-compiler Criterion A/B shows no consistent performance change | CI re-run |
| T1 | Structural header validation and graph parsing were never exercised: existing corruption tests flip header bytes, which the header CRC rejects first | test gap | forged-header tests with recomputed valid CRC (12 header cases, 8 graph cases) | `malformed_tests::forged_*` |
| T2 | Exact-search / recall ground truth used the same SIMD kernels as the system under test | test gap | ground truth recomputed in plain `f64` on raw vectors | `index_tests::brute_force` |
| T3 | Uncovered unsafe paths: `Arena::alloc_layout` zero-size branch; `Kernels::try_dot`/`try_cosine` success paths | test gap | targeted unit tests | `arena::tests::zero_sized_*`, `simd::tests::normalize_and_errors` |
| T4 | No runtime-free concurrent ring test that Miri can explore | test gap | `std_threads_handoff_preserves_order_and_ownership` (boxed values, capacity 4) | itself, under Miri ×16 seeds |
| T5 | Empty segment / empty store / crash-leftover behavior untested | test gap | tests | `malformed_tests::empty_*`, `recovery_*` |
| T6 | Frozen-graph search and flat scan were only reachable through mmap (not Miri-checkable) | test gap | in-memory graph roundtrip; small scan-vs-brute-force test; Miri-sized HNSW build tests via `cfg!(miri)` | `hnsw::graph::tests::*`, `flat::tests::small_*` |
| D1 | README/doc claims without methodology or overstated (see [§9](#9-readme-performance-claims)) | doc | corrected or qualified | — |

## 4. Unsafe code inventory

Totals: **97 `unsafe` blocks (95 in library code, 2 in test modules), 26 `unsafe fn`, 5 `unsafe impl`** across `sovereign-core`,
`sovereign-index` and `sovereign-pipeline` (`sovereign-cli` has none). Every block has a
`// SAFETY:` comment and every `unsafe fn` a `# Safety` section; this is enforced by
`clippy::undocumented_unsafe_blocks` and `clippy::missing_safety_doc`, and
`unsafe_op_in_unsafe_fn` is denied workspace-wide.

How to read the table: "executed by tests" is the llvm-cov execution count of the block's first
counted line during `cargo test --workspace` (dispatch auto → AVX-512; AVX2 and scalar kernels are
reached through `Kernels::for_backend` in the kernel tests). Groups share one invariant, listed
after the table.

| # | location | kind | fn | group | executed by tests (count) | Miri |
|---:|---|---|---|---|---|---|
| 1 | `sovereign-core/src/aligned.rs:116` | impl | `Send for AlignedVec` | aligned.rs:impl | n/a (compile-time) | n/a (trait impl) |
| 2 | `sovereign-core/src/aligned.rs:118` | impl | `Sync for AlignedVec` | aligned.rs:impl | n/a (compile-time) | n/a (trait impl) |
| 3 | `sovereign-core/src/aligned.rs:131` | block | `dangling` | aligned.rs:dangling | yes (3.18k) | pass (SB) |
| 4 | `sovereign-core/src/aligned.rs:171` | block | `zeroed` | aligned.rs:zeroed | yes (1) | pass (SB) |
| 5 | `sovereign-core/src/aligned.rs:226` | block | `as_slice` | aligned.rs:as_slice | yes (81.3k) | pass (SB) |
| 6 | `sovereign-core/src/aligned.rs:234` | block | `as_mut_slice` | aligned.rs:as_mut_slice | yes (61.4k) | pass (SB) |
| 7 | `sovereign-core/src/aligned.rs:294` | block | `grow_to` | aligned.rs:grow_to | yes (3.16k) | pass (SB) |
| 8 | `sovereign-core/src/aligned.rs:299` | block | `grow_to` | aligned.rs:grow_to | yes (355) | pass (SB) |
| 9 | `sovereign-core/src/aligned.rs:314` | block | `push` | aligned.rs:push | yes (10.1k) | pass (SB) |
| 10 | `sovereign-core/src/aligned.rs:324` | block | `extend_from_slice` | aligned.rs:extend_from_slice | yes (142k) | pass (SB) |
| 11 | `sovereign-core/src/aligned.rs:343` | block | `resize` | aligned.rs:resize | yes (126k) | pass (SB) |
| 12 | `sovereign-core/src/aligned.rs:373` | block | `drop` | aligned.rs:drop | yes (3.16k) | pass (SB) |
| 13 | `sovereign-core/src/arena.rs:65` | impl | `Send for Arena` | arena.rs:impl | n/a (compile-time) | n/a (trait impl) |
| 14 | `sovereign-core/src/arena.rs:106` | block | `alloc` | arena.rs:alloc | yes (10.0k) | pass (SB) |
| 15 | `sovereign-core/src/arena.rs:123` | block | `alloc_slice_copy` | arena.rs:alloc_slice_copy | yes (1.00k) | pass (SB) |
| 16 | `sovereign-core/src/arena.rs:134` | block | `alloc_str` | arena.rs:alloc_str | yes (1.00k) | pass (SB) |
| 17 | `sovereign-core/src/arena.rs:151` | block | `alloc_str_concat` | arena.rs:alloc_str_concat | yes (19) | pass (SB) |
| 18 | `sovereign-core/src/arena.rs:156` | block | `alloc_str_concat` | arena.rs:alloc_str_concat | yes (6) | pass (SB) |
| 19 | `sovereign-core/src/arena.rs:168` | block | `alloc_layout` | arena.rs:alloc_layout | yes (2) | pass (SB) |
| 20 | `sovereign-core/src/arena.rs:197` | block | `try_bump` | arena.rs:try_bump | yes (11.0k) | pass (SB) |
| 21 | `sovereign-core/src/arena.rs:208` | block | `alloc_slow` | arena.rs:alloc_slow | yes (26) | pass (SB) |
| 22 | `sovereign-core/src/arena.rs:231` | block | `reset` | arena.rs:reset | yes (9) | pass (SB) |
| 23 | `sovereign-core/src/arena.rs:252` | block | `drop` | arena.rs:drop | yes (17) | pass (SB) |
| 24 | `sovereign-core/src/hint.rs:46` | block | `prefetch_read` | hint.rs:prefetch_read | yes (163M) | pass (SB) |
| 25 | `sovereign-core/src/matrix.rs:141` | block | `row_padded` | matrix.rs:row_padded | yes (42.9k) | pass (SB) |
| 26 | `sovereign-core/src/matrix.rs:150` | fn | `row_padded_unchecked` | matrix.rs:row_padded_unchecked | yes (113M) | pass (SB) |
| 27 | `sovereign-core/src/matrix.rs:154` | block | `row_padded_unchecked` | matrix.rs:row_padded_unchecked | yes (113M) | pass (SB) |
| 28 | `sovereign-core/src/simd/mod.rs:262` | block | `dot` | mod.rs:* | yes (670) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 29 | `sovereign-core/src/simd/mod.rs:275` | block | `l2_sq` | mod.rs:* | yes (668) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 30 | `sovereign-core/src/simd/mod.rs:288` | block | `cosine` | mod.rs:* | yes (648) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 31 | `sovereign-core/src/simd/mod.rs:296` | block | `norm_sq` | mod.rs:* | yes (65.9k) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 32 | `sovereign-core/src/simd/mod.rs:311` | block | `dot_x4` | mod.rs:* | yes (39) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 33 | `sovereign-core/src/simd/mod.rs:324` | block | `try_dot` | mod.rs:* | yes (1) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 34 | `sovereign-core/src/simd/mod.rs:337` | block | `try_cosine` | mod.rs:* | yes (1) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 35 | `sovereign-core/src/simd/mod.rs:349` | fn | `dot_unchecked` | mod.rs:*_unchecked | yes (64.8M) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 36 | `sovereign-core/src/simd/mod.rs:352` | block | `dot_unchecked` | mod.rs:*_unchecked | yes (64.8M) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 37 | `sovereign-core/src/simd/mod.rs:361` | fn | `l2_sq_unchecked` | mod.rs:*_unchecked | yes (147k) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 38 | `sovereign-core/src/simd/mod.rs:364` | block | `l2_sq_unchecked` | mod.rs:*_unchecked | yes (147k) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 39 | `sovereign-core/src/simd/mod.rs:373` | fn | `dot_x4_unchecked` | mod.rs:*_unchecked | yes (6.58M) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 40 | `sovereign-core/src/simd/mod.rs:375` | block | `dot_x4_unchecked` | mod.rs:*_unchecked | yes (6.58M) | pass (SB; scalar, +avx2,+fma, +avx512f builds) |
| 41 | `sovereign-core/src/simd/neon.rs:20` | fn | `dot_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 42 | `sovereign-core/src/simd/neon.rs:28` | block | `dot_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 43 | `sovereign-core/src/simd/neon.rs:38` | block | `dot_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 44 | `sovereign-core/src/simd/neon.rs:44` | block | `dot_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 45 | `sovereign-core/src/simd/neon.rs:55` | fn | `l2_sq_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 46 | `sovereign-core/src/simd/neon.rs:63` | block | `l2_sq_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 47 | `sovereign-core/src/simd/neon.rs:77` | block | `l2_sq_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 48 | `sovereign-core/src/simd/neon.rs:86` | block | `l2_sq_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 49 | `sovereign-core/src/simd/neon.rs:98` | fn | `cosine_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 50 | `sovereign-core/src/simd/neon.rs:106` | block | `cosine_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 51 | `sovereign-core/src/simd/neon.rs:125` | block | `cosine_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 52 | `sovereign-core/src/simd/neon.rs:139` | fn | `norm_sq_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 53 | `sovereign-core/src/simd/neon.rs:141` | block | `norm_sq_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 54 | `sovereign-core/src/simd/neon.rs:149` | fn | `dot_x4_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 55 | `sovereign-core/src/simd/neon.rs:159` | block | `dot_x4_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 56 | `sovereign-core/src/simd/neon.rs:181` | block | `dot_x4_neon` | neon.rs:* | not compiled on x86_64 — CI-only (macos-14) | not run (not compiled on x86_64) |
| 57 | `sovereign-core/src/simd/x86.rs:51` | fn | `dot_avx2` | x86.rs:avx2 | yes (420) | pass (SB, +avx2,+fma build) |
| 58 | `sovereign-core/src/simd/x86.rs:59` | block | `dot_avx2` | x86.rs:avx2 | yes (1.56k) | pass (SB, +avx2,+fma build) |
| 59 | `sovereign-core/src/simd/x86.rs:79` | block | `dot_avx2` | x86.rs:avx2 | yes (590) | pass (SB, +avx2,+fma build) |
| 60 | `sovereign-core/src/simd/x86.rs:86` | block | `dot_avx2` | x86.rs:avx2 | yes (1.43k) | pass (SB, +avx2,+fma build) |
| 61 | `sovereign-core/src/simd/x86.rs:97` | fn | `l2_sq_avx2` | x86.rs:avx2 | yes (210) | pass (SB, +avx2,+fma build) |
| 62 | `sovereign-core/src/simd/x86.rs:105` | block | `l2_sq_avx2` | x86.rs:avx2 | yes (780) | pass (SB, +avx2,+fma build) |
| 63 | `sovereign-core/src/simd/x86.rs:121` | block | `l2_sq_avx2` | x86.rs:avx2 | yes (295) | pass (SB, +avx2,+fma build) |
| 64 | `sovereign-core/src/simd/x86.rs:128` | block | `l2_sq_avx2` | x86.rs:avx2 | yes (715) | pass (SB, +avx2,+fma build) |
| 65 | `sovereign-core/src/simd/x86.rs:143` | fn | `cosine_avx2` | x86.rs:avx2 | yes (213) | pass (SB, +avx2,+fma build) |
| 66 | `sovereign-core/src/simd/x86.rs:151` | block | `cosine_avx2` | x86.rs:avx2 | yes (1.66k) | pass (SB, +avx2,+fma build) |
| 67 | `sovereign-core/src/simd/x86.rs:169` | block | `cosine_avx2` | x86.rs:avx2 | yes (99) | pass (SB, +avx2,+fma build) |
| 68 | `sovereign-core/src/simd/x86.rs:180` | block | `cosine_avx2` | x86.rs:avx2 | yes (730) | pass (SB, +avx2,+fma build) |
| 69 | `sovereign-core/src/simd/x86.rs:194` | fn | `norm_sq_avx2` | x86.rs:avx2 | yes (210) | pass (SB, +avx2,+fma build) |
| 70 | `sovereign-core/src/simd/x86.rs:196` | block | `norm_sq_avx2` | x86.rs:avx2 | yes (210) | pass (SB, +avx2,+fma build) |
| 71 | `sovereign-core/src/simd/x86.rs:204` | fn | `dot_x4_avx2` | x86.rs:avx2 | yes (13) | pass (SB, +avx2,+fma build) |
| 72 | `sovereign-core/src/simd/x86.rs:214` | block | `dot_x4_avx2` | x86.rs:avx2 | yes (227) | pass (SB, +avx2,+fma build) |
| 73 | `sovereign-core/src/simd/x86.rs:230` | block | `dot_x4_avx2` | x86.rs:avx2 | yes (3) | pass (SB, +avx2,+fma build) |
| 74 | `sovereign-core/src/simd/x86.rs:247` | block | `dot_x4_avx2` | x86.rs:avx2 | yes (29) | pass (SB, +avx2,+fma build) |
| 75 | `sovereign-core/src/simd/x86.rs:274` | fn | `dot_avx512` | x86.rs:avx512 | yes (64.9M) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 76 | `sovereign-core/src/simd/x86.rs:282` | block | `dot_avx512` | x86.rs:avx512 | yes (148k) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 77 | `sovereign-core/src/simd/x86.rs:302` | block | `dot_avx512` | x86.rs:avx512 | yes (137M) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 78 | `sovereign-core/src/simd/x86.rs:311` | block | `dot_avx512` | x86.rs:avx512 | yes (887) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 79 | `sovereign-core/src/simd/x86.rs:322` | fn | `l2_sq_avx512` | x86.rs:avx512 | yes (147k) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 80 | `sovereign-core/src/simd/x86.rs:330` | block | `l2_sq_avx512` | x86.rs:avx512 | yes (133k) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 81 | `sovereign-core/src/simd/x86.rs:346` | block | `l2_sq_avx512` | x86.rs:avx512 | yes (42.4k) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 82 | `sovereign-core/src/simd/x86.rs:353` | block | `l2_sq_avx512` | x86.rs:avx512 | yes (192) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 83 | `sovereign-core/src/simd/x86.rs:366` | fn | `cosine_avx512` | x86.rs:avx512 | yes (223) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 84 | `sovereign-core/src/simd/x86.rs:374` | block | `cosine_avx512` | x86.rs:avx512 | yes (810) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 85 | `sovereign-core/src/simd/x86.rs:394` | block | `cosine_avx512` | x86.rs:avx512 | yes (98) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 86 | `sovereign-core/src/simd/x86.rs:398` | block | `cosine_avx512` | x86.rs:avx512 | yes (204) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 87 | `sovereign-core/src/simd/x86.rs:417` | fn | `norm_sq_avx512` | x86.rs:avx512 | yes (65.4k) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 88 | `sovereign-core/src/simd/x86.rs:419` | block | `norm_sq_avx512` | x86.rs:avx512 | yes (65.4k) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 89 | `sovereign-core/src/simd/x86.rs:427` | fn | `dot_x4_avx512` | x86.rs:avx512 | yes (6.58M) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 90 | `sovereign-core/src/simd/x86.rs:437` | block | `dot_x4_avx512` | x86.rs:avx512 | yes (6.62M) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 91 | `sovereign-core/src/simd/x86.rs:456` | block | `dot_x4_avx512` | x86.rs:avx512 | yes (7.01k) | pass (SB, +avx512f build; masked tail at n=17 confirmed) |
| 92 | `sovereign-index/src/flat.rs:32` | fn | `scan_range` | flat.rs:* | yes (60.3k) | pass (SB: sequential path incl. 1x4, remainder, L2); parallel branch not run under Miri (needs ≥4M floats) |
| 93 | `sovereign-index/src/flat.rs:38` | block | `scan_range` | flat.rs:* | yes (147k) | pass (SB: sequential path incl. 1x4, remainder, L2); parallel branch not run under Miri (needs ≥4M floats) |
| 94 | `sovereign-index/src/flat.rs:46` | block | `scan_range` | flat.rs:* | yes (6.58M) | pass (SB: sequential path incl. 1x4, remainder, L2); parallel branch not run under Miri (needs ≥4M floats) |
| 95 | `sovereign-index/src/flat.rs:64` | block | `scan_range` | flat.rs:* | yes (15) | pass (SB: sequential path incl. 1x4, remainder, L2); parallel branch not run under Miri (needs ≥4M floats) |
| 96 | `sovereign-index/src/flat.rs:86` | block | `scan` | flat.rs:* | yes (60.3k) | pass (SB: sequential path incl. 1x4, remainder, L2); parallel branch not run under Miri (needs ≥4M floats) |
| 97 | `sovereign-index/src/flat.rs:99` | block | `scan` | flat.rs:* | yes (66) | pass (SB: sequential path incl. 1x4, remainder, L2); parallel branch not run under Miri (needs ≥4M floats) |
| 98 | `sovereign-index/src/flat.rs:177` | block | `parallel_scan_matches_sequential_exactly` | flat.rs:* | yes (2) | pass (SB: sequential path incl. 1x4, remainder, L2); parallel branch not run under Miri (needs ≥4M floats) |
| 99 | `sovereign-index/src/hnsw/build.rs:99` | block | `insert` | build.rs:* | yes (32.5k) | pass (SB sequential; TB parallel ×4 seeds) |
| 100 | `sovereign-index/src/hnsw/build.rs:102` | block | `insert` | build.rs:* | yes (57.7k) | pass (SB sequential; TB parallel ×4 seeds) |
| 101 | `sovereign-index/src/hnsw/build.rs:110` | block | `insert` | build.rs:* | yes (35.1k) | pass (SB sequential; TB parallel ×4 seeds) |
| 102 | `sovereign-index/src/hnsw/build.rs:128` | block | `insert` | build.rs:* | yes (35.1k) | pass (SB sequential; TB parallel ×4 seeds) |
| 103 | `sovereign-index/src/hnsw/build.rs:167` | block | `connect` | build.rs:* | yes (18.5k) | pass (SB sequential; TB parallel ×4 seeds) |
| 104 | `sovereign-index/src/hnsw/build.rs:175` | block | `connect` | build.rs:* | yes (18.5k) | pass (SB sequential; TB parallel ×4 seeds) |
| 105 | `sovereign-index/src/hnsw/graph.rs:135` | fn | `search` | graph.rs:* | yes (21.9k) | pass (SB, in-memory encode→parse→search) |
| 106 | `sovereign-index/src/hnsw/graph.rs:150` | block | `search` | graph.rs:* | yes (21.9k) | pass (SB, in-memory encode→parse→search) |
| 107 | `sovereign-index/src/hnsw/graph.rs:153` | block | `search` | graph.rs:* | yes (44.1k) | pass (SB, in-memory encode→parse→search) |
| 108 | `sovereign-index/src/hnsw/graph.rs:156` | block | `search` | graph.rs:* | yes (21.9k) | pass (SB, in-memory encode→parse→search) |
| 109 | `sovereign-index/src/hnsw/graph.rs:242` | block | `encoded_graph_roundtrips_and_finds_stored_vectors` | graph.rs:* | yes (1.00k) | pass (SB, in-memory encode→parse→search) |
| 110 | `sovereign-index/src/hnsw/search.rs:128` | fn | `dist` | search.rs:* | yes (64.8M) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 111 | `sovereign-index/src/hnsw/search.rs:130` | block | `dist` | search.rs:* | yes (64.8M) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 112 | `sovereign-index/src/hnsw/search.rs:132` | block | `dist` | search.rs:* | yes (64.8M) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 113 | `sovereign-index/src/hnsw/search.rs:146` | fn | `dist_nodes` | search.rs:* | yes (22.5M) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 114 | `sovereign-index/src/hnsw/search.rs:148` | block | `dist_nodes` | search.rs:* | yes (22.5M) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 115 | `sovereign-index/src/hnsw/search.rs:150` | block | `dist_nodes` | search.rs:* | yes (22.5M) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 116 | `sovereign-index/src/hnsw/search.rs:164` | fn | `greedy_step` | search.rs:* | yes (101k) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 117 | `sovereign-index/src/hnsw/search.rs:181` | block | `greedy_step` | search.rs:* | yes (1.42M) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 118 | `sovereign-index/src/hnsw/search.rs:198` | fn | `search_layer` | search.rs:* | yes (57.0k) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 119 | `sovereign-index/src/hnsw/search.rs:244` | block | `search_layer` | search.rs:* | yes (40.8M) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 120 | `sovereign-index/src/hnsw/search.rs:267` | fn | `select_neighbors` | search.rs:* | yes (53.6k) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 121 | `sovereign-index/src/hnsw/search.rs:279` | block | `select_neighbors` | search.rs:* | yes (22.0M) | pass (SB sequential build + graph search; TB parallel build ×4 seeds) |
| 122 | `sovereign-index/src/segment.rs:113` | block | `open_with` | segment.rs:open_with | yes (63) | n/a (Miri cannot execute mmap) |
| 123 | `sovereign-index/src/segment.rs:364` | block | `search_prepared` | segment.rs:search_prepared | yes (20.9k) | n/a (needs an mmapped segment; callee paths checked via graph/flat tests) |
| 124 | `sovereign-pipeline/src/ring.rs:64` | impl | `Send for Shared` | ring.rs:impl | n/a (compile-time) | n/a (trait impl) |
| 125 | `sovereign-pipeline/src/ring.rs:66` | impl | `Sync for Shared` | ring.rs:impl | n/a (compile-time) | n/a (trait impl) |
| 126 | `sovereign-pipeline/src/ring.rs:76` | block | `drop` | ring.rs:drop | yes (6) | pass (SB, 16 seeds; tokio shutdown test 4 seeds) |
| 127 | `sovereign-pipeline/src/ring.rs:186` | block | `try_send` | ring.rs:try_send | yes (220k) | pass (SB, 16 seeds; tokio shutdown test 4 seeds) |
| 128 | `sovereign-pipeline/src/ring.rs:254` | block | `try_recv` | ring.rs:try_recv | yes (220k) | pass (SB, 16 seeds; tokio shutdown test 4 seeds) |

#### Group invariants

| group | safety invariant | alignment & bounds | CPU features |
|---|---|---|---|
| aligned.rs:impl | Unique ownership of the allocation, like `Vec<T>`: `Send` iff `T: Send`, `Sync` iff `T: Sync` | n/a | none |
| aligned.rs:dangling | Non-null constant 64 with no provenance; never dereferenced while `cap == 0`; valid base for empty slices | 64 ≥ `align_of::<T>()` (compile-time assert) | none |
| aligned.rs:zeroed | `alloc_zeroed` with a non-zero-size layout; null → `handle_alloc_error`; all-zero bytes are valid for `T: Pod` | layout align 64, size `len·size_of::<T>()` (checked) | none |
| aligned.rs:as_slice | `ptr` valid for `len` initialized elements (type invariant) | `len ≤ cap`; base 64-aligned or dangling-aligned | none |
| aligned.rs:as_mut_slice | As `as_slice`, uniqueness from `&mut self` | `len ≤ cap` | none |
| aligned.rs:grow_to | `alloc` (cap 0) or `realloc` with the exact old layout; new size non-zero and ≤ `isize::MAX`; `realloc` keeps the 64-byte alignment | align 64; size checked | none |
| aligned.rs:push | Slot `len` is inside the allocation after `reserve` | `len < cap` | none |
| aligned.rs:extend_from_slice | `reserve` guarantees `cap − len ≥ src.len()`; `&mut self` rules out aliasing with `src` | `len + src.len() ≤ cap` | none |
| aligned.rs:resize | Writes indices `len..new_len`, all `< cap` after `reserve` | `new_len ≤ cap` | none |
| aligned.rs:drop | `dealloc` with the identical layout, only when `cap > 0` | align 64 | none |
| arena.rs:impl | Arena owns its chunks; outstanding borrows are tied to `&self`, and `Arena: !Sync`, so moving it between threads is sound | n/a | none |
| arena.rs:alloc | Fresh, disjoint region sized/aligned for `T`, valid for `&self`; `T: Copy` so no drop glue | `Layout::new::<T>()` | none |
| arena.rs:alloc_slice_copy | Fresh region from `Layout::array`; cannot overlap `src` | `Layout::array::<T>(len)` (checked) | none |
| arena.rs:alloc_str | Byte-exact copy of valid UTF-8 | align 1 | none |
| arena.rs:alloc_str_concat | Writes stay within `total` (checked sum); concatenation of valid UTF-8 is valid UTF-8 | `at + p.len() ≤ total` | none |
| arena.rs:alloc_layout | Zero-size request: non-null pointer equal to `align`, valid for zero-size access | aligned to `layout.align()` | none |
| arena.rs:try_bump | `start ≤ end ≤ cap` with checked arithmetic; `base` non-null | `addr` rounded up to `layout.align()` | none |
| arena.rs:alloc_slow | Chunk size ≥ `size + align` > 0; the fresh chunk always satisfies the request | chunk align `max(align, 64)` | none |
| arena.rs:reset | Each freed chunk is deallocated with its recorded layout; `&mut self` proves no live borrows | n/a | none |
| arena.rs:drop | As `reset`, on drop | n/a | none |
| hint.rs:prefetch_read | `prefetcht0` never faults and never dereferences in the abstract machine; any address is allowed | none | SSE (x86_64 baseline) |
| matrix.rs:row_padded | Bounds checked (`assert!(i < rows)`) immediately before | `[i·stride, (i+1)·stride) ⊆ data` | none |
| matrix.rs:row_padded_unchecked | Caller guarantees `i < rows`; constructor proved `data.len() == rows·stride` without overflow | `[i·stride, (i+1)·stride) ⊆ data` | none |
| mod.rs:* | `Kernels` type invariant: a `&'static Kernels` exists only for a backend `is_supported()` confirmed (private fields, checked constructors). Kernels clamp to the shortest operand, so memory safety does not depend on lengths; `check_len` enforces logical correctness | none (kernels use unaligned loads) | backend's ISA, verified at table construction |
| mod.rs:*_unchecked | Same CPU invariant; equal lengths are a *logical* precondition only (a mismatch computes a truncated result, never reads out of bounds) | none | backend's ISA, verified at table construction |
| neon.rs:* | NEON is mandatory on AArch64. `vld1q_f32` at `p.add(i)` with `i + W ≤ n`, `n = min(len)`; 4-byte alignment (f32) suffices; scalar tail `get_unchecked(i)`, `i < n` | 4 B | `neon` |
| x86.rs:avx2 | Caller guarantees AVX2+FMA (only reachable through a verified `Kernels` table). `_mm256_loadu_ps`/`_mm512_loadu_ps` at `p.add(i)` with `i + W ≤ n`, `n = min(len)` of all operands; unaligned loads (no alignment requirement); scalar tail uses `get_unchecked(i)` with `i < n` | unaligned OK; 64 B alignment avoids split loads | `avx2`, `fma` |
| x86.rs:avx512 | Caller guarantees AVX-512F. `_mm256_loadu_ps`/`_mm512_loadu_ps` at `p.add(i)` with `i + W ≤ n`, `n = min(len)` of all operands; unaligned loads (no alignment requirement); tail: masked `_mm512_maskz_loadu_ps(k, p.add(i))`, `i < n`, `k = (1 << (n−i)) − 1`: only in-bounds lanes are accessed; masked lanes are neither read nor faulted on | unaligned OK; 64 B alignment avoids split loads | `avx512f` |
| flat.rs:* | `scan` asserts `q.len() == stride` before any call; ranges are `start < end ≤ rows`; rows via `row_padded_unchecked(r)`, `r < end`; kernels see equal lengths (`stride`) | row starts 64 B aligned when the base is (mmap page / `AlignedVec`) | via verified `Kernels` |
| build.rs:* | All ids are `q ∈ 1..n`, the entry point (an inserted node) or ids read from adjacency lists, which only ever contain inserted ids; `qv` is a padded row (`len == stride`) | as `Space` | via verified `Kernels` |
| graph.rs:* | `search` returns early unless `space.len() == self.len()`; `entry_point < node_count` validated in `parse`; `q.len() == stride` asserted by `Segment::search_prepared` | as `Space` | via verified `Kernels` |
| search.rs:* | `Space::dist*` require `id < rows` and `q.len() == stride`. Traversal checks every neighbor id `nb < n` before calling; entry ids are validated (entry point at parse / inserted nodes at build) | `row_padded_unchecked(id)` within `rows·stride` | via verified `Kernels` |
| segment.rs:open_with | `Mmap::map`: the file must not be modified or truncated while mapped. Upheld by protocol (write tmp → fsync → rename; published segments are never opened for writing). External modification is **not** defended against (UB / SIGBUS) — see remaining risks | mmap base is page aligned; sections 4 KiB aligned ⇒ all typed casts aligned | none |
| segment.rs:search_prepared | Graph search over a `Space` built from the same segment; `q.len() == stride` asserted on entry | as `Space` | via verified `Kernels` |
| ring.rs:impl | Values move between exactly one producer and one consumer via the acquire/release protocol; no `&T` is ever shared, so `T: Send` suffices for both `Send` and `Sync` | n/a | none |
| ring.rs:drop | Last `Arc` gone ⇒ exclusive access; slots `[head, tail)` are initialized and dropped exactly once | slot index `i & mask` | none |
| ring.rs:try_send | `tail − head < cap` with `head` loaded `Acquire` ⇒ slot is free and not accessed by the consumer until the `Release` store of `tail` | slot index `tail & mask` | none |
| ring.rs:try_recv | `head < tail` with `tail` loaded `Acquire` ⇒ slot fully written; producer won't reuse it until the `Release` store of `head` | slot index `head & mask` | none |

**Design-level invariant behind the SIMD blocks.** Kernel functions are `unsafe fn` with
`#[target_feature]`. The only way to call them from safe code is through a `&'static Kernels`,
which can only be obtained via `Kernels::for_backend`, `kernels()` or `init_backend`, all of which
check `Backend::is_supported()` (runtime `cpuid` on x86_64). Fields are private, so no table for an
unsupported ISA can be constructed. Kernels clamp every loop to `min(len)` of all operands, so a
length mismatch can only produce a wrong (truncated) value, never an out-of-bounds read; the safe
wrappers additionally panic on mismatch (`check_len`).

## 5. Storage review

**mmap.** `Segment::open_with` maps the file read-only (`Mmap::map`, the one `unsafe` in
`segment.rs`). Soundness relies on the file never being modified or truncated while mapped. The
crate upholds this by protocol: segments are written to a temp file, fsynced, renamed into place,
and never opened for writing again; replacement creates a new inode. This does **not** defend
against external tampering or truncation (UB / `SIGBUS`), and I/O errors on mapped pages surface
as `SIGBUS` rather than `Result` — both listed in [§10](#10-remaining-risks).
`madvise` ranges: sections are 4 KiB aligned; memmap2 rounds advice offsets down to the real page
size (checked in memmap2 0.9.11 `unix.rs`), so 16 KiB-page hosts do not get `EINVAL`.

**Validation at open (O(1)).** Order: length ≥ header → magic → version → header CRC32 → flags ⊆
known → metric tag → `1 ≤ dim ≤ 65536` → `stride == padded_stride(dim)` → `count ≤ u32::MAX−1` →
for every section: `offset + len` computed with `checked_add` and `≤ file length`, non-empty
sections 4 KiB aligned, exact expected sizes (`count·stride·4` with checked multiplication,
`count·8`, `(count+1)·8`) → payload/graph sections present iff their flag is set → graph header
parse (magic, `2 ≤ m ≤ m0 ≤ 1024`, `max_level < 64`, `node_count == count`,
`entry_point < count`, sub-sections in bounds with exact sizes, entry point on the top layer).
All of these branches are now exercised by `malformed_tests` with forged, CRC-valid headers.

**Lazy (per-access) validation.** Payload offsets are bounds-checked per access and payload bytes
UTF-8 validated per access; neighbor ids are checked `< count` before use; per-node levels and
upper-layer row indices use checked slicing. `corrupt_body_degrades_without_panicking` points every
layer-0 neighbor at `u32::MAX` and breaks payload offsets: search still returns only valid rows,
payload access returns `Corrupt`, `verify()` fails.

**CRC32.** Header CRC checked on every open; body CRC (all bytes after the header, including
inter-section padding) only via `Segment::verify()` / `OpenOptions::verify_checksum`, by design (so
open stays O(1)). Consequence: without `verify`, silent body corruption yields wrong results, not
errors. CRC32 detects accidental corruption only; it is not an integrity guarantee against
tampering. The manifest carries its own CRC32 and a strict line parser that rejects path
separators and dot-files in segment names.

**Atomic publish.** Segment: temp file in the same directory → write → header patched → `fsync`
→ `rename` → `fsync(dir)`; a drop guard deletes the temp file on every error path. Manifest: temp →
`fsync` → `rename` → `fsync(dir)`. Publication order is segment durable → manifest durable →
in-memory `ArcSwap::store`. With `sync: false` / `--no-sync` the segment is **not** fsynced but
the manifest still is, so after power loss a manifest can reference an unreadable segment.

**Recovery** (`recovery_ignores_crash_leftovers_and_reports_lost_segments`): the manifest is the
source of truth. Unpublished segments and temp files are ignored; an orphaned segment id is reused
by the next publish and atomically replaced; a manifest referencing a missing segment fails open
with an `Io` error naming the file. There is no garbage collection of leftovers and no repair tool.

## 6. SPSC ring review

- **Ordering.** Producer: slot write → `tail.store(Release)`; consumer: `tail.load(Acquire)` → slot
  read. Slot reuse: consumer read → `head.store(Release)`; producer `head.load(Acquire)` before
  overwriting. Cached indices are only refreshed through these acquiring loads.
- **Wakeups.** Both sides use `AtomicWaker` with register-then-recheck. The two RMWs on the
  waker state are totally ordered: if `register` follows `wake`, it acquires the waker's release
  and the recheck observes the new index; if it precedes, `wake` finds the waker.
- **Close.** `Producer::drop` stores `tx_closed` (`Release`) after its last `tail` store; the
  consumer's final `try_recv` after acquiring `tx_closed` therefore sees every item. Symmetric for
  `rx_closed`, which fails pending and future sends with the value returned.
- **Ownership.** Values are moved in/out through `MaybeUninit` slots; `Shared::drop` (exclusive
  after the last `Arc`) drops exactly the items in `[head, tail)`. `Send`/`Sync` require only
  `T: Send` because no `&T` is ever shared. `Producer`/`Consumer` are not `Clone`, and all
  mutating methods take `&mut self`.
- **Cancellation.** Dropping a pending `send()` future drops the value it carries (the usual
  cancellation semantics); no slot is left half-initialized because writes only happen in
  `try_send` after the capacity check.
- **Evidence.** Unit tests (FIFO under wraparound, 200k-item tokio cross-thread stream, exactly-once
  drop, close semantics, drop-while-parked) plus Miri with 16 schedule seeds on a std-thread handoff
  (boxed values) and 4 seeds on the tokio shutdown test: no UB or data race reported. Miri explores
  a subset of interleavings and weak-memory behaviors; it is not an exhaustive model check (loom
  was not used).

## 7. HNSW recall and ground truth

- **Tests.** `hnsw_recall_is_high`: 6,000 clustered Gaussian vectors (40 clusters, dim 48),
  200 held-out queries from the same distribution (different seed), k = 10, ef = 128,
  `SearchMode::Approximate` (always uses the graph). Ground truth: full sort of `f64` cosine on raw
  vectors, ties broken by id (independent of kernels, normalization and index code). Recall =
  |HNSW ∩ truth| / (queries · k). Measured 0.974–0.982 over 5 runs (the parallel build is non-deterministic); threshold 0.95.
  `every_vector_finds_itself` (query = stored vector; an easy case): 2987–2998 / 3000 over 5 runs.
- **Exact search.** `exact_search_matches_brute_force_for_all_metrics` requires the exact scan to
  reproduce the `f64` reference ranking *in order* for cosine, dot and L2. Near-ties between `f32`
  and `f64` could in principle reorder results; with the fixed seeds none occur.
- **CLI bench** (`sovereign bench`): truth = exact scan of the same segment (validated against
  `f64` by the test above); queries are fresh points from the cluster distribution; recall
  isolates index quality from embedding quality.
- **Limits.** Only synthetic clustered data was evaluated (test: n = 6,000, dim 48; bench:
  n = 20,000, dim 1536). No real-embedding benchmark, no graph-connectivity check, and parallel
  builds are non-deterministic (sequential builds are deterministic and tested).

## 8. Races, UB, overflow, malformed files, empty inputs

- **Data races.** No `static mut`. Shared mutable state: `ArcSwap`, atomics, `parking_lot::Mutex`,
  `OnceLock`, and the ring's `UnsafeCell` slots (protocol above). The HNSW builder's only `unsafe`
  is read-only access to the immutable matrix. `Arena` is `!Sync`.
- **UB risks.** Beyond the mmap contract, all unsafe code is listed in [§4](#4-unsafe-code-inventory).
  Masked AVX-512 loads rely on the documented fault-suppression semantics of
  `_mm512_maskz_loadu_ps`.
- **Integer overflow.** Size computations use `checked_mul`/`checked_add` (`AlignedVec::layout`,
  `padded_stride`, `MatrixRef::new`, section ends, vector section size, arena bumps,
  `node_count·(m0+1)`); ring indices wrap intentionally (`wrapping_*`). Residual: on 32-bit targets
  some index arithmetic (`node · (m0+1)`, `row · (m+1)`) could wrap for corrupt ids — slicing is
  still checked (no UB), but 32-bit is untested. `ef_construction` is stored truncated to `u32` in
  an informational header field.
- **Malformed files.** See [§5](#5-storage-review); no panic paths were found for body corruption.
  Not fuzzed.
- **Empty inputs** (all tested or checked manually): empty segment (open/verify/search in all
  modes), empty store search, `k = 0`, empty/whitespace/token-free text in chunker and embedder,
  empty directory ingest ("nothing to index"), compact with 0/1 segments, query on missing store,
  missing ingest path, dimension mismatch with an existing store.

## 9. README performance claims

| claim | methodology | status |
|---|---|---|
| Kernel speedups, dim sweep, alignment, 1×4 blocking, top-k (Criterion tables) | `cargo bench -p sovereign-core --bench simd_bench`; single run, one VM | kept; single-run/noise caveat added |
| `sovereign bench` block (build, open, exact/HNSW latency, recall) | `sovereign bench --dim 1536 --rows 20000 --queries 200` (seed 42) | re-measured after F5 |
| Ingestion stress test | `sovereign ingest ~/.cargo/registry/src --index <dir> --no-sync`, `sovereign query ... --repeat N` | re-measured; corpus is machine-specific (not reproducible byte-for-byte), page cache warm, query latency is repeats of **one** query |
| "~1 ns" indirect-call dispatch | none | removed (stated as not benchmarked) |
| "layer 0 … >90% of search time" | none | removed |
| "opening a 10 GB segment touches one page" | none (and wrong: graph segments also read the graph header) | corrected |
| "sections are page aligned" | wrong on 16 KiB-page hosts | corrected to 4 KiB |
| NEON "executed on CI" | CI not observed | corrected: compile-checked only |
| heuristic selection is "the main reason" recall is high | no ablation | attributed to the paper, ablation absent |
| 1024-row exact-scan threshold is "faster" | not benchmarked | stated as untuned heuristic |
| DRAM-bound 1×4 gain from memory-level parallelism | not measured | marked as plausible explanation |
| scalar backend auto-vectorizes | disassembly of release binary (`mulps`/`addps` in `scalar::dot`) | verified |
| PMU `perf stat` recipes | not runnable in this VM | marked as not run |
| microarchitecture facts (FMA latency/ports, load ports, L2 adjacent-line prefetch) | vendor documentation | background, not measured here |

## 10. Remaining risks

1. **mmap contract.** External modification/truncation of a published segment is UB / `SIGBUS`;
   I/O errors on mapped pages crash the process instead of returning errors.
2. **Cross-process readers during compaction.** A reader process running `IndexStore::open` can
   read a manifest and then fail with `ENOENT` because compaction deleted a listed segment; there is
   no retry.
3. **No garbage collection** of temp files or orphaned segments after crashes or compaction
   crashes.
4. **`--no-sync`** can leave a durable manifest pointing at a non-durable segment; the store then
   fails to open and there is no repair tool.
5. **Silent body corruption** is only detected when `verify()` / `verify_checksum` is used; CRC32 is
   not tamper-evident.
6. **Dedup** uses a 64-bit non-cryptographic hash: a collision drops a distinct chunk; the dedup set
   grows with the number of chunks in a run.
7. **Recall** validated only on synthetic data; parallel HNSW builds are non-deterministic; no
   connectivity guarantees after heuristic pruning.
8. **NEON** never executed by the author; **32-bit, Windows, macOS** untested locally.
9. **Advisory `flock`** writer lock may not work on network filesystems.
10. **Absurd inputs** (`SegmentWriter::with_capacity(huge)`, `bench --rows` near `usize::MAX`)
    panic on capacity overflow instead of returning an error.
11. **Demo embedder** is lexical feature hashing, with measurable hash-collision noise.
12. **Performance numbers** come from single runs on one shared cloud VM.
13. **No fuzzing, loom, or sanitizer runs.**
14. **Toolchain drift.** CI tracks floating `stable` with `-D warnings`, so a new clippy lint can
    turn CI red without any code change (this happened once, F8). Pinning the CI toolchain would
    trade that for staleness; left as is.
