# Public release audit for v0.1.0

Audit performed on 2026-11-27 on `release/v0.1.0-preparation`.

**GO for public visibility based on the reviewed Git contents.** The release CI gate
passed, and the repository owner approved public exposure of their existing commit email.
The intended release is an early local retrieval engine for trusted index directories and
document trees. The bundled embedder uses lexical feature hashing.

## Findings and changes

| Severity | Finding | Change |
| --- | --- | --- |
| P1 | A store handle opened before another writer published could subsequently acquire the writer lock with stale state, reuse segment IDs, and replace the newer manifest. Racing initializers could also overwrite a newly created manifest. | Serialize lazy lock acquisition within each handle. Under the OS lock, require a fresh manifest for existing handles and an absent manifest for creation. Reject stale writers with an instruction to reopen. Added a stale-handle regression. |
| P1 | If directory sync failed after manifest rename, the error path deleted newly referenced segments and left the in-memory snapshot behind the on-disk manifest. | Treat rename as the publication boundary. Update the snapshot and retain new segments when post-publication sync fails. Compaction also retains the old files in that case. Added a directory-sync fault regression. |
| P1 | The CLI benchmark used a predictable temporary directory and armed recursive cleanup before ensuring exclusive ownership. | Use `tempfile` for an exclusively created, automatically cleaned benchmark directory. The crate was already pinned in the lockfile for workspace tests. |
| Hardening | Temporary output files could be opened for truncation if their paths already existed. Malformed manifests could repeat an ID, attach trailing fields, or point at a noncanonical file. | Create temp files exclusively and clean up only owned paths. Require canonical segment names and unique IDs, with a valid-checksum duplicate-entry regression. Check ID and generation increments. |
| Hardening | Graph row addressing could overflow `usize` on 32-bit targets. Segment opening indexed the mapping directly using a prior file-length check. | Use checked graph row access and validate the mapped header length directly. Require a regular segment file. |

No P0 defect was confirmed in this review.

## Architecture and safety review

- **Core.** Reviewed alignment, allocation provenance, arena reset lifetimes, matrix strides,
  and SIMD dispatch. AVX2 requires AVX2 and FMA, AVX-512 requires AVX-512F, and AArch64 checks
  NEON. The safe kernel wrappers validate lengths, and unchecked search callers derive lengths
  from the prepared query and validated matrix. AVX-512 tail loads mask inactive lanes.
- **Index.** Segment open checks header CRC, version, section sizes and bounds. Full body CRC
  is opt-in. HNSW checks graph shape and entry point before search, then checks neighbor IDs
  before unchecked distance access. The builder publishes back-links only after initializing
  the new node's lists. Per-node locks protect adjacency edits. Snapshot guards and owned
  results pin mappings across same-process publication and compaction.
- **Pipeline.** Reviewed the SPSC ring's single-owner endpoints, acquire/release slot handoff,
  wake registration and state recheck, and teardown of undelivered values. Reviewed the staging
  path from directory discovery and bounded queues to embedding and publication. No change to
  the ring or SIMD kernels was warranted by this review.

## Verification

Local checks used Rust 1.100.0 on x86-64 Windows with LLVM-MinGW (`windows-gnullvm`).
The initial Windows GNU attempts failed before workspace verification because the host lacked
the native C compiler and GNU linker support. The final run used the portable LLVM linker.

| Check | Result |
| --- | --- |
| `cargo fmt --all` and `cargo fmt --all --check` | Passed |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | Passed |
| `cargo test --workspace --locked --offline` | 87 passed, including two doc tests. Zero failed or ignored. Cargo reported an existing unused `tokio-stream` dependency in the pipeline manifest. |
| `git diff --check` | Passed |

GitHub Actions passed **7 of 7 jobs** for release commit `c279f89`, whose tree matches
`origin/main` at `75eea96`. The run covered Ubuntu with automatic, AVX2, and scalar dispatch,
macOS 15 on Apple Silicon ARM64 with automatic dispatch and native NEON coverage,
Rust 1.89 MSRV, formatting and Clippy, and benchmark compilation.
The audit did not execute benchmarks or check 32-bit targets.
[GitHub Actions run](https://github.com/tahaelfitouri/sovereign-rag/actions/runs/46022805204)

## Performance and licensing

The README retains the historical Criterion single-run results, four-run synthetic CLI
medians and ranges, and single-run warm-cache ingestion measurements. It records the
hardware, commands, sample or run counts, and limitations. The release audit did not reproduce
those performance measurements. Real-corpus recall, transformer pipeline performance,
and performance on other hosts remain unverified. The AI-assisted development disclosure
remains in the README.

The repository has a root MIT license and each workspace package inherits the MIT SPDX
identifier. The source tree carries the license. The inspected `sovereign-core` Cargo
package file list omits the root license text. Include the license text in standalone crate
archives before any crates.io release. A full transitive dependency license and advisory
review was outside this audit.

## Secrets and privacy

The final lightweight review scanned 56 tracked files and 14 commits reachable from local
refs, including the four fetched origin branches.
Filename and text rules covered common credential filenames, private-key headers,
provider-token formats, credential-bearing URLs, and common quoted secret assignments.
No candidate credentials or sensitive filenames matched. This pattern-based scan cannot
rule out arbitrary, encoded, binary, or revoked secrets. GitHub issues, PR discussions,
Actions logs and artifacts, and local configuration outside Git were outside the scan.

The follow-up review identified one personal Outlook address, which appears only in author
and committer metadata. On 2026-11-27, the repository owner approved public exposure of this
existing address and directed that Git history and AI-assisted development attribution be
preserved. The other flagged address is an automated Anthropic noreply address that appears
in commit metadata and seven co-author trailers. This audit omits the address values.
Generated indexes can embed document text and source paths. Keep them out of release artifacts.

## Remaining risks and release gates

- Use a trusted local index directory. External writes or truncation violate mmap invariants
  and can terminate the process or invalidate Rust references. Checksums detect accidental
  corruption. They provide no authentication. Indexes should not be accepted from untrusted
  sources or kept in directories writable by untrusted users.
- Coordinate new opens across processes with compaction. Compaction can unlink a segment
  between another process reading its manifest and mapping that segment. Existing handles
  retain their snapshots. Network filesystem lock and rename semantics are unverified.
- Directory-sync failure can leave durability uncertain after publication. Inspect the store
  before retrying. `--no-sync` can leave an unrecoverable manifest after power loss. Automatic
  crash-orphan cleanup and repair remain absent. Directory syncing is implemented on Unix.
- Parallel HNSW builds are nondeterministic, and approximate recall varies with data and
  parameters. Some invalid or extreme inputs can exhaust memory. Source trees should stay
  trusted and stable during ingestion. Discovery's size check precedes the file read.
- Fuzzing, Miri, sanitizers and concurrency model checking were not rerun for this audit.
