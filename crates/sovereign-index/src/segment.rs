//! Read-only, memory-mapped segments.
//!
//! Opening a segment maps the file and validates the 192-byte header in O(1). Vectors, ids,
//! payloads and the HNSW graph are then served as borrowed slices straight out of the page cache
//! — no deserialization, no copies, and memory usage is whatever the OS decides to keep resident.
//! Many processes mapping the same segment share one physical copy.
//!
//! ```text
//!   Segment ──owns──► Mmap (PROT_READ, MAP_SHARED) ──► page cache ──► file on disk
//!      │
//!      ├─ vectors()  → MatrixRef<'_>       (&[f32] over the vectors section)
//!      ├─ ids()      → &[u64]
//!      ├─ payload(r) → Option<&str>        (UTF-8 validated per access)
//!      └─ graph()    → HnswGraphRef<'_>    (&[u32] adjacency)
//! ```

use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};
use sovereign_core::{padded_stride, Kernels, MatrixRef, Metric, TopK};

use crate::error::{IndexError, Result};
use crate::flat;
use crate::format::{
    flags, Section, SegmentHeader, FORMAT_VERSION, HEADER_SIZE, MAGIC, MAX_DIM, MAX_ROWS,
    SECTION_ALIGN,
};
use crate::hnsw::search::{LayerScratch, Space};
use crate::hnsw::HnswGraphRef;
use crate::query::{Hit, SearchMode, SearchParams, SearchScratch};

/// In `Auto` mode, segments smaller than this are scanned exactly (faster than a graph walk and
/// 100% recall).
const AUTO_EXACT_BELOW: usize = 1024;

/// Kernel page-cache hint for the vector section.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessPattern {
    /// No hint.
    Normal,
    /// Graph search: disable readahead (`MADV_RANDOM`) so each fault pulls only the page needed.
    Random,
    /// Flat scans: aggressive readahead (`MADV_SEQUENTIAL`).
    Sequential,
    /// Start paging everything in asynchronously (`MADV_WILLNEED`).
    WillNeed,
}

/// Options controlling how a segment is mapped.
#[derive(Clone, Debug, Default)]
pub struct OpenOptions {
    /// Verify the body CRC32 at open (reads the whole file once).
    pub verify_checksum: bool,
    /// Pre-fault all pages at map time (`MAP_POPULATE`) — trades open latency for no first-query
    /// page faults.
    pub populate: bool,
    /// `mlock` the mapping so it can never be swapped out (requires `RLIMIT_MEMLOCK`).
    pub lock_memory: bool,
    /// Access hint for the vector section; `None` picks `Random` for graph segments and
    /// `Sequential` otherwise.
    pub access: Option<AccessPattern>,
}

/// A validated, memory-mapped segment. Cheap to share behind an `Arc`.
#[derive(Debug)]
pub struct Segment {
    mmap: Mmap,
    header: SegmentHeader,
    metric: Metric,
    path: PathBuf,
    kernels: &'static Kernels,
}

fn section_bytes(mmap: &[u8], s: Section) -> &[u8] {
    // Bounds were validated at open; `get` keeps this panic-free regardless.
    let start = s.offset as usize;
    start.checked_add(s.len as usize).and_then(|end| mmap.get(start..end)).unwrap_or(&[])
}

impl Segment {
    /// Opens and validates a segment with default options.
    ///
    /// # Errors
    /// I/O errors, bad magic/version, header checksum mismatch or structural corruption.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, &OpenOptions::default())
    }

    /// Opens and validates a segment.
    ///
    /// # Errors
    /// As [`open`](Self::open), plus body checksum mismatch when `verify_checksum` is set and
    /// `mlock`/`madvise` failures when requested.
    pub fn open_with(path: impl AsRef<Path>, opts: &OpenOptions) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| IndexError::io(&path, e))?;
        let len = file.metadata().map_err(|e| IndexError::io(&path, e))?.len();
        if len < HEADER_SIZE as u64 {
            return Err(IndexError::corrupt(&path, "file shorter than header"));
        }

        let mut mo = MmapOptions::new();
        if opts.populate {
            mo.populate();
        }
        // SAFETY: mapping a file is only sound if nobody mutates or truncates it while mapped
        // (the mapped `&[u8]` would change underneath us, or accesses would SIGBUS). Segments are
        // immutable by protocol: they are written to a temp file and atomically renamed into
        // place, and no code path in this crate ever opens a published segment for writing.
        // Replacing a segment creates a *new* inode; existing mappings keep the old one alive.
        // External tampering with published files is outside this contract (as for LMDB et al.).
        let mmap = unsafe { mo.map(&file) }.map_err(|e| IndexError::io(&path, e))?;

        let header: SegmentHeader = bytemuck::pod_read_unaligned(&mmap[..HEADER_SIZE]);
        let metric = validate(&path, &header, mmap.len() as u64)?;

        let seg = Self { mmap, header, metric, path, kernels: sovereign_core::kernels() };
        if header.has(flags::HAS_GRAPH) {
            HnswGraphRef::parse(seg.section(header.graph), seg.len())
                .map_err(|r| IndexError::corrupt(&seg.path, r))?;
        }
        if opts.verify_checksum {
            seg.verify()?;
        }
        seg.apply_options(opts)?;
        Ok(seg)
    }

    fn apply_options(&self, opts: &OpenOptions) -> Result<()> {
        #[cfg(unix)]
        {
            use memmap2::Advice;
            let access = opts.access.unwrap_or(if self.has_graph() {
                AccessPattern::Random
            } else {
                AccessPattern::Sequential
            });
            let advice = match access {
                AccessPattern::Normal => None,
                AccessPattern::Random => Some(Advice::Random),
                AccessPattern::Sequential => Some(Advice::Sequential),
                AccessPattern::WillNeed => Some(Advice::WillNeed),
            };
            let v = self.header.vectors;
            if let (Some(advice), false) = (advice, v.is_empty()) {
                self.mmap
                    .advise_range(advice, v.offset as usize, v.len as usize)
                    .map_err(|e| IndexError::io(&self.path, e))?;
            }
            if opts.lock_memory {
                self.mmap.lock().map_err(|e| IndexError::io(&self.path, e))?;
            }
        }
        #[cfg(not(unix))]
        let _ = opts;
        Ok(())
    }

    #[inline]
    fn section(&self, s: Section) -> &[u8] {
        section_bytes(&self.mmap, s)
    }

    /// Recomputes the body CRC32 and compares it with the header (reads the whole file).
    ///
    /// # Errors
    /// [`IndexError::Checksum`] on mismatch.
    pub fn verify(&self) -> Result<()> {
        let computed = crc32fast::hash(&self.mmap[HEADER_SIZE..]);
        if computed != self.header.body_crc32 {
            return Err(IndexError::Checksum {
                path: self.path.clone(),
                what: "body",
                stored: self.header.body_crc32,
                computed,
            });
        }
        Ok(())
    }

    /// File path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Raw header.
    #[must_use]
    pub fn header(&self) -> &SegmentHeader {
        &self.header
    }

    /// Similarity metric.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Logical dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.header.dim as usize
    }

    /// Physical row length.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.header.stride as usize
    }

    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.header.count as usize
    }

    /// `true` if the segment has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.header.count == 0
    }

    /// Embedder fingerprint.
    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        self.header.fingerprint
    }

    /// Whether an HNSW graph is present.
    #[must_use]
    pub fn has_graph(&self) -> bool {
        self.header.has(flags::HAS_GRAPH)
    }

    /// Mapped size in bytes.
    #[must_use]
    pub fn mapped_bytes(&self) -> usize {
        self.mmap.len()
    }

    /// Zero-copy view of the (padded) vectors.
    #[must_use]
    pub fn vectors(&self) -> MatrixRef<'_> {
        MatrixRef::from_bytes(
            self.section(self.header.vectors),
            self.len(),
            self.dim(),
            self.stride(),
        )
        .expect("vector section shape validated at open")
    }

    /// Zero-copy view of the external ids.
    #[must_use]
    pub fn ids(&self) -> &[u64] {
        bytemuck::try_cast_slice(self.section(self.header.ids))
            .expect("id section validated at open")
    }

    /// External id of `row`.
    #[must_use]
    pub fn id(&self, row: usize) -> Option<u64> {
        self.ids().get(row).copied()
    }

    /// Payload text of `row` (`None` if the segment stores no payloads).
    ///
    /// # Errors
    /// [`IndexError::RowOutOfRange`], [`IndexError::Corrupt`] for inconsistent offsets, or
    /// [`IndexError::InvalidPayload`] for non-UTF-8 bytes.
    pub fn payload(&self, row: usize) -> Result<Option<&str>> {
        if !self.header.has(flags::HAS_PAYLOAD) {
            return Ok(None);
        }
        if row >= self.len() {
            return Err(IndexError::RowOutOfRange { row, rows: self.len() });
        }
        let offsets: &[u64] =
            bytemuck::try_cast_slice(self.section(self.header.payload_offsets))
                .map_err(|_| IndexError::corrupt(&self.path, "payload offsets misaligned"))?;
        let data = self.section(self.header.payload_data);
        let (a, b) = match (offsets.get(row), offsets.get(row + 1)) {
            (Some(&a), Some(&b)) => (a as usize, b as usize),
            _ => return Err(IndexError::corrupt(&self.path, "payload offset table truncated")),
        };
        let bytes = data
            .get(a..b)
            .ok_or_else(|| IndexError::corrupt(&self.path, "payload offsets out of range"))?;
        core::str::from_utf8(bytes).map(Some).map_err(|_| IndexError::InvalidPayload { row })
    }

    /// Zero-copy view of the HNSW graph, if present.
    #[must_use]
    pub fn graph(&self) -> Option<HnswGraphRef<'_>> {
        if !self.has_graph() {
            return None;
        }
        HnswGraphRef::parse(self.section(self.header.graph), self.len()).ok()
    }

    /// Searches this segment, allocating scratch space internally.
    ///
    /// # Errors
    /// Dimension mismatch, non-finite query or zero-norm query under cosine.
    pub fn search(&self, query: &[f32], params: &SearchParams) -> Result<Vec<Hit>> {
        let mut scratch = SearchScratch::new();
        // `k` is caller-controlled: never size allocations by more than the row count.
        let mut out = Vec::with_capacity(params.k.min(self.len()));
        self.search_into(&mut scratch, query, params, &mut out)?;
        Ok(out)
    }

    /// Searches with caller-provided scratch; allocation-free once `scratch` and `out` are warm.
    ///
    /// # Errors
    /// As [`search`](Self::search).
    pub fn search_into(
        &self,
        scratch: &mut SearchScratch,
        query: &[f32],
        params: &SearchParams,
        out: &mut Vec<Hit>,
    ) -> Result<()> {
        scratch.prepare(query, self.dim(), self.stride(), self.metric, self.kernels)?;
        scratch.topk.reset(params.k.min(self.len()));
        self.search_prepared(&scratch.query, params, &mut scratch.layer, &mut scratch.topk, 0);
        out.clear();
        let ids = self.ids();
        out.extend(scratch.topk.sort_best_first().iter().map(|n| {
            let row = (n.id & u64::from(u32::MAX)) as u32;
            Hit { row, id: ids[row as usize], score: n.score }
        }));
        Ok(())
    }

    /// Core search on an already prepared query. Pushes `(key_base | row, score)` into `topk`.
    ///
    /// # Panics
    /// Panics if `q.len() != self.stride()` (the safety of the unchecked kernels depends on it).
    pub(crate) fn search_prepared(
        &self,
        q: &[f32],
        params: &SearchParams,
        layer: &mut LayerScratch,
        topk: &mut TopK,
        key_base: u64,
    ) {
        assert_eq!(q.len(), self.stride(), "prepared query must match segment stride");
        if params.k == 0 || self.is_empty() {
            return;
        }
        let m = self.vectors();
        let graph = match params.mode {
            SearchMode::Exact => None,
            SearchMode::Auto if self.len() < AUTO_EXACT_BELOW => None,
            SearchMode::Auto | SearchMode::Approximate => self.graph(),
        };
        match graph {
            Some(g) => {
                let space = Space::new(self.kernels, m, self.metric);
                // SAFETY: `space` covers this segment's matrix (same row count as the graph,
                // validated at open) and `q.len() == stride` was asserted above.
                unsafe { g.search(&space, q, params.ef, layer, topk, key_base) };
            }
            None => flat::scan(self.kernels, m, q, self.metric == Metric::L2, topk, key_base),
        }
    }
}

/// Validates a header against the file length. O(1).
fn validate(path: &Path, h: &SegmentHeader, file_len: u64) -> Result<Metric> {
    if h.magic != MAGIC {
        return Err(IndexError::BadMagic { path: path.to_path_buf() });
    }
    if h.version != FORMAT_VERSION {
        return Err(IndexError::UnsupportedVersion {
            path: path.to_path_buf(),
            found: h.version,
            supported: FORMAT_VERSION,
        });
    }
    let computed = h.compute_crc();
    if computed != h.header_crc32 {
        return Err(IndexError::Checksum {
            path: path.to_path_buf(),
            what: "header",
            stored: h.header_crc32,
            computed,
        });
    }
    let corrupt = |r: &str| IndexError::corrupt(path, r);
    if h.flags & !flags::KNOWN != 0 {
        return Err(corrupt("unknown header flags"));
    }
    let metric = Metric::from_tag(h.metric)?;
    let dim = h.dim as usize;
    if dim == 0 || dim > MAX_DIM {
        return Err(corrupt("dimension out of range"));
    }
    if h.stride as usize != padded_stride(dim)? {
        return Err(corrupt("stride does not match padded dimension"));
    }
    if h.count > MAX_ROWS as u64 {
        return Err(corrupt("row count exceeds limit"));
    }

    let check = |s: Section, expected: Option<u64>, what: &str| -> Result<()> {
        let end = s.end().ok_or_else(|| corrupt(&format!("{what} section overflows")))?;
        if end > file_len {
            return Err(corrupt(&format!("{what} section extends past end of file")));
        }
        if !s.is_empty() && !s.offset.is_multiple_of(SECTION_ALIGN) {
            return Err(corrupt(&format!("{what} section is not page aligned")));
        }
        if let Some(e) = expected {
            if s.len != e {
                return Err(corrupt(&format!("{what} section has wrong size")));
            }
        }
        Ok(())
    };
    let vec_len = h
        .count
        .checked_mul(u64::from(h.stride))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| corrupt("vector section size overflows"))?;
    check(h.vectors, Some(vec_len), "vectors")?;
    check(h.ids, Some(h.count * 8), "ids")?;
    if h.has(flags::HAS_PAYLOAD) {
        check(h.payload_offsets, Some((h.count + 1) * 8), "payload offsets")?;
        check(h.payload_data, None, "payload data")?;
    } else if !h.payload_offsets.is_empty() || !h.payload_data.is_empty() {
        return Err(corrupt("payload sections present without flag"));
    }
    if h.has(flags::HAS_GRAPH) {
        check(h.graph, None, "graph")?;
    } else if !h.graph.is_empty() {
        return Err(corrupt("graph section present without flag"));
    }
    Ok(metric)
}
