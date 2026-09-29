//! On-disk segment format (`.srag`), version 1.
//!
//! A segment is an immutable file written once (`write → fsync → rename`) and then only ever
//! memory-mapped read-only. All integers are little-endian; all sections start on a 4 KiB page
//! boundary so each can be `madvise`d independently and every typed view is naturally aligned.
//!
//! ```text
//!  offset 0      ┌────────────────────────────────────────────┐
//!                │ SegmentHeader (192 B, 3 cache lines)       │ magic, version, metric, dim,
//!                │                                            │ stride, count, section table,
//!                │                                            │ body CRC32, header CRC32
//!  4096          ├────────────────────────────────────────────┤
//!                │ vectors: count × stride × f32              │ row-major, rows padded to 64 B,
//!                │                                            │ L2-normalized if metric=cosine
//!  align 4096    ├────────────────────────────────────────────┤
//!                │ ids: count × u64                           │ external ids
//!  align 4096    ├────────────────────────────────────────────┤
//!                │ payload offsets: (count + 1) × u64         │ optional
//!  align 4096    ├────────────────────────────────────────────┤
//!                │ payload data: UTF-8 bytes                  │ optional
//!  align 4096    ├────────────────────────────────────────────┤
//!                │ graph: GraphHeader + HNSW adjacency        │ optional (see `hnsw`)
//!                └────────────────────────────────────────────┘
//! ```
//!
//! ## Why not rkyv?
//!
//! rkyv shines for nested, pointer-rich object graphs. Everything here is a flat array of
//! fixed-width scalars, so a `#[repr(C)]` layout + `bytemuck` casts gives *true* zero-copy with an
//! O(1) validation cost (a 192-byte header check), no relative-pointer resolution and no
//! archive-format coupling. Opening a 10 GB segment touches exactly one page.
//!
//! ## Integrity
//!
//! The header carries its own CRC32 (checked on every open) and a CRC32 of the whole body
//! (checked on demand via `Segment::verify`, since reading gigabytes at open time would defeat
//! lazy paging). Every file-derived index is bounds-checked before use, so a corrupt body can
//! produce wrong answers but never undefined behaviour.

use bytemuck::{Pod, Zeroable};

#[cfg(not(target_endian = "little"))]
compile_error!("sovereign-index segments are little-endian; big-endian targets are unsupported");

/// File magic.
pub const MAGIC: [u8; 8] = *b"SVRGNIDX";
/// Graph section magic.
pub const GRAPH_MAGIC: [u8; 8] = *b"SVRGHNSW";
/// Current format version.
pub const FORMAT_VERSION: u32 = 1;
/// Section alignment (one page).
pub const SECTION_ALIGN: u64 = 4096;
/// Sub-section alignment inside the graph section.
pub const SUBSECTION_ALIGN: u64 = 64;
/// Maximum supported dimensionality.
pub const MAX_DIM: usize = 65_536;
/// Maximum rows per segment (HNSW node ids are `u32`; `u32::MAX` is reserved as a sentinel).
pub const MAX_ROWS: usize = (u32::MAX - 1) as usize;

/// Header flag bits.
pub mod flags {
    /// Stored vectors are L2-normalized.
    pub const NORMALIZED: u64 = 1 << 0;
    /// A graph section is present.
    pub const HAS_GRAPH: u64 = 1 << 1;
    /// Payload sections are present.
    pub const HAS_PAYLOAD: u64 = 1 << 2;
    /// All flags known to this version.
    pub const KNOWN: u64 = NORMALIZED | HAS_GRAPH | HAS_PAYLOAD;
}

/// `(offset, len)` of a byte range, relative to the file start (or graph section start).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Section {
    /// Byte offset.
    pub offset: u64,
    /// Byte length.
    pub len: u64,
}

impl Section {
    /// Exclusive end offset, if it does not overflow.
    #[must_use]
    pub fn end(&self) -> Option<u64> {
        self.offset.checked_add(self.len)
    }

    /// `true` if the section is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Fixed-size segment header at offset 0.
///
/// `#[repr(C, align(64))]` pins field order and places the struct on its own cache lines; the
/// explicit `_reserved` tail makes the size exactly 192 bytes with **no implicit padding**, which
/// `bytemuck::Pod` verifies at compile time (padding bytes would be uninitialized memory).
///
/// ```text
///  0        8      12     16    20      24      32      40           48
///  ├ magic ─┼ ver ─┼ met ─┼ dim ┼ stride┼ flags ┼ count ┼ fingerprint ┤
///  48 ─ vectors ─ 64 ─ ids ─ 80 ─ pay_off ─ 96 ─ pay_data ─ 112 ─ graph ─ 128
///  128 body_crc32 │ 132 header_crc32 │ 136 ────────── reserved ────────── 192
/// ```
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct SegmentHeader {
    /// [`MAGIC`].
    pub magic: [u8; 8],
    /// [`FORMAT_VERSION`].
    pub version: u32,
    /// [`sovereign_core::Metric`] tag.
    pub metric: u32,
    /// Logical dimensionality.
    pub dim: u32,
    /// Physical row length in `f32`s (multiple of 16).
    pub stride: u32,
    /// [`flags`] bitset.
    pub flags: u64,
    /// Number of rows.
    pub count: u64,
    /// Embedder fingerprint (model + config hash); guards against querying with the wrong model.
    pub fingerprint: u64,
    /// Vector matrix.
    pub vectors: Section,
    /// External ids.
    pub ids: Section,
    /// Payload offset table.
    pub payload_offsets: Section,
    /// Payload bytes.
    pub payload_data: Section,
    /// HNSW graph.
    pub graph: Section,
    /// CRC32 of every byte after the header.
    pub body_crc32: u32,
    /// CRC32 of the header with this field zeroed.
    pub header_crc32: u32,
    /// Reserved for future use; must be zero.
    pub _reserved: [u8; 56],
}

/// Size of [`SegmentHeader`] in bytes.
pub const HEADER_SIZE: usize = core::mem::size_of::<SegmentHeader>();
const _: () = assert!(HEADER_SIZE == 192);
const _: () = assert!(core::mem::align_of::<SegmentHeader>() == 64);

impl SegmentHeader {
    /// Computes the header CRC (over all bytes with `header_crc32` zeroed).
    #[must_use]
    pub fn compute_crc(&self) -> u32 {
        let mut h = *self;
        h.header_crc32 = 0;
        crc32fast::hash(bytemuck::bytes_of(&h))
    }

    /// Tests a flag bit.
    #[must_use]
    pub fn has(&self, flag: u64) -> bool {
        self.flags & flag != 0
    }
}

/// Header of the graph section. Offsets inside are relative to the graph section start.
///
/// Adjacency layout:
///
/// ```text
///  levels      : count × u8                         top layer of each node
///  layer0      : count × (1 + m0) × u32             [len, n0, n1, ... n(m0-1)] per node
///  upper_index : count × u32                        first upper row of node, or u32::MAX
///  upper       : upper_rows × (1 + m) × u32         rows for layers 1..=level, node-contiguous
/// ```
///
/// Fixed-width rows make neighbor lookup two multiplies and a bounds check — no pointer chasing
/// and no per-node allocation, and layer 0 (where search spends >90% of its time) is one dense
/// array that the hardware prefetcher handles well.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GraphHeader {
    /// [`GRAPH_MAGIC`].
    pub magic: [u8; 8],
    /// Max neighbors per node on layers ≥ 1.
    pub m: u32,
    /// Max neighbors per node on layer 0.
    pub m0: u32,
    /// Highest layer in the graph.
    pub max_level: u32,
    /// Entry-point node.
    pub entry_point: u32,
    /// Number of nodes (== segment row count).
    pub node_count: u64,
    /// `ef_construction` used at build time (informational).
    pub ef_construction: u32,
    /// Explicit padding; must be zero.
    pub _pad: u32,
    /// Per-node levels.
    pub levels: Section,
    /// Layer-0 adjacency.
    pub layer0: Section,
    /// Per-node index into `upper`.
    pub upper_index: Section,
    /// Upper-layer adjacency.
    pub upper: Section,
    /// RNG seed used for level assignment (informational).
    pub seed: u64,
    /// Reserved; must be zero.
    pub _reserved: [u8; 16],
}

/// Size of [`GraphHeader`] in bytes.
pub const GRAPH_HEADER_SIZE: usize = core::mem::size_of::<GraphHeader>();
const _: () = assert!(GRAPH_HEADER_SIZE == 128);

/// Rounds `v` up to a multiple of `align` (a power of two).
#[must_use]
pub const fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_crc_detects_changes() {
        let mut h = SegmentHeader::zeroed();
        h.magic = MAGIC;
        h.dim = 8;
        h.header_crc32 = h.compute_crc();
        let crc = h.header_crc32;
        assert_eq!(h.compute_crc(), crc, "crc must ignore its own field");
        h.dim = 9;
        assert_ne!(h.compute_crc(), crc);
    }

    #[test]
    fn alignment_helpers() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(193, 64), 256);
        assert_eq!(Section { offset: u64::MAX, len: 1 }.end(), None);
    }
}
