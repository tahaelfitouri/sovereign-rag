//! Read-only HNSW graph borrowed straight from a memory-mapped segment.

use sovereign_core::TopK;

use super::build::BuiltGraph;
use super::search::{greedy_step, search_layer, Adjacency, Cand, LayerScratch, Space};
use crate::format::{
    align_up, GraphHeader, Section, GRAPH_HEADER_SIZE, GRAPH_MAGIC, SUBSECTION_ALIGN,
};

/// Largest `m0` accepted from disk (keeps a corrupt header from implying absurd row sizes).
const MAX_M0: u32 = 1024;
const MAX_LEVELS: u32 = 64;

/// Zero-copy view of a serialized HNSW graph.
///
/// Construction validates the header in O(1): magic, parameter ranges and that every sub-section
/// lies inside the graph section with the exact expected size. Per-node data (levels, row indices,
/// neighbor ids) is bounds-checked on access, so a corrupt body can only degrade results.
#[derive(Clone, Copy, Debug)]
pub struct HnswGraphRef<'a> {
    header: GraphHeader,
    levels: &'a [u8],
    layer0: &'a [u32],
    upper_index: &'a [u32],
    upper: &'a [u32],
    m: usize,
    m0: usize,
}

fn sub(bytes: &[u8], s: Section) -> Result<&[u8], &'static str> {
    let start = usize::try_from(s.offset).map_err(|_| "sub-section offset overflow")?;
    let len = usize::try_from(s.len).map_err(|_| "sub-section length overflow")?;
    let end = start.checked_add(len).ok_or("sub-section end overflow")?;
    bytes.get(start..end).ok_or("sub-section out of bounds")
}

fn u32s(bytes: &[u8]) -> Result<&[u32], &'static str> {
    bytemuck::try_cast_slice(bytes).map_err(|_| "misaligned or ragged u32 sub-section")
}

impl<'a> HnswGraphRef<'a> {
    /// Parses and validates a graph section.
    ///
    /// # Errors
    /// A static description of the first structural problem found.
    pub fn parse(bytes: &'a [u8], node_count: usize) -> Result<Self, &'static str> {
        let head = bytes.get(..GRAPH_HEADER_SIZE).ok_or("graph section shorter than its header")?;
        let header: GraphHeader = bytemuck::pod_read_unaligned(head);
        if header.magic != GRAPH_MAGIC {
            return Err("bad graph magic");
        }
        if header.m < 2 || header.m0 < header.m || header.m0 > MAX_M0 {
            return Err("graph m/m0 out of range");
        }
        if header.max_level >= MAX_LEVELS {
            return Err("graph max_level out of range");
        }
        if header.node_count != node_count as u64 {
            return Err("graph node count does not match segment row count");
        }
        if node_count > 0 && header.entry_point as usize >= node_count {
            return Err("graph entry point out of range");
        }
        let (m, m0) = (header.m as usize, header.m0 as usize);

        let levels = sub(bytes, header.levels)?;
        let layer0 = u32s(sub(bytes, header.layer0)?)?;
        let upper_index = u32s(sub(bytes, header.upper_index)?)?;
        let upper = u32s(sub(bytes, header.upper)?)?;

        let expect0 = node_count.checked_mul(m0 + 1).ok_or("layer0 size overflow")?;
        if levels.len() != node_count || layer0.len() != expect0 || upper_index.len() != node_count
        {
            return Err("graph sub-section sizes do not match node count");
        }
        if upper.len() % (m + 1) != 0 {
            return Err("upper adjacency is not a whole number of rows");
        }
        if node_count > 0 && u32::from(levels[header.entry_point as usize]) != header.max_level {
            return Err("entry point is not on the top layer");
        }
        Ok(Self { header, levels, layer0, upper_index, upper, m, m0 })
    }

    /// Graph header (parameters).
    #[must_use]
    pub fn header(&self) -> &GraphHeader {
        &self.header
    }

    /// Number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// `true` if the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Neighbors of `node` on `layer` as a borrowed slice (empty if out of range).
    #[inline]
    #[must_use]
    pub fn neighbors(&self, node: u32, layer: usize) -> &'a [u32] {
        let node = node as usize;
        if layer == 0 {
            let w = self.m0 + 1;
            return match self.layer0.get(node * w..node * w + w) {
                Some(row) => &row[1..1 + (row[0] as usize).min(self.m0)],
                None => &[],
            };
        }
        match (self.levels.get(node), self.upper_index.get(node)) {
            (Some(&lvl), Some(&base)) if layer <= lvl as usize && base != u32::MAX => {
                let w = self.m + 1;
                let r = base as usize + layer - 1;
                match self.upper.get(r * w..r * w + w) {
                    Some(row) => &row[1..1 + (row[0] as usize).min(self.m)],
                    None => &[],
                }
            }
            _ => &[],
        }
    }

    /// k-NN search: greedy descent through the upper layers, then a beam of width `ef` on layer 0.
    /// Results are pushed into `out` as `(key_base | row, score)` with `score = −distance`.
    ///
    /// # Safety
    /// `space` must be built over the segment this graph belongs to (`space.len() == self.len()`),
    /// and `q.len()` must equal the segment stride.
    pub(crate) unsafe fn search(
        &self,
        space: &Space<'_>,
        q: &[f32],
        ef: usize,
        scratch: &mut LayerScratch,
        out: &mut TopK,
        key_base: u64,
    ) {
        if self.is_empty() || space.len() != self.len() {
            return;
        }
        let ep = self.header.entry_point;
        // SAFETY: `ep < node_count == space.len()` was validated in `parse`; `q` length is the
        // caller's contract.
        let mut cur = Cand { dist: unsafe { space.dist(q, ep) }, id: ep };
        for layer in (1..=self.header.max_level as usize).rev() {
            // SAFETY: `cur.id` is always a validated node id.
            cur = unsafe { greedy_step(self, space, q, cur, layer, &mut scratch.nbuf) };
        }
        // SAFETY: as above.
        unsafe { search_layer(self, space, q, &[cur], ef.max(out.k()), 0, scratch) };
        for c in scratch.results.drain() {
            out.push(key_base | u64::from(c.id), -c.dist);
        }
    }
}

impl Adjacency for HnswGraphRef<'_> {
    #[inline]
    fn neighbors_into(&self, node: u32, layer: usize, out: &mut Vec<u32>) {
        out.clear();
        out.extend_from_slice(self.neighbors(node, layer));
    }
}

/// Serializes a built graph into the section layout described in [`GraphHeader`].
pub(crate) fn encode(g: &BuiltGraph) -> Vec<u8> {
    let mut header = GraphHeader {
        magic: GRAPH_MAGIC,
        m: g.m as u32,
        m0: g.m0 as u32,
        max_level: g.max_level,
        entry_point: g.entry_point,
        node_count: g.levels.len() as u64,
        ef_construction: g.ef_construction as u32,
        _pad: 0,
        levels: Section::default(),
        layer0: Section::default(),
        upper_index: Section::default(),
        upper: Section::default(),
        seed: g.seed,
        _reserved: [0; 16],
    };
    let mut out = vec![0u8; GRAPH_HEADER_SIZE];
    let place = |out: &mut Vec<u8>, bytes: &[u8]| -> Section {
        let offset = align_up(out.len() as u64, SUBSECTION_ALIGN);
        out.resize(offset as usize, 0);
        out.extend_from_slice(bytes);
        Section { offset, len: bytes.len() as u64 }
    };
    header.levels = place(&mut out, &g.levels);
    header.layer0 = place(&mut out, bytemuck::cast_slice(&g.layer0));
    header.upper_index = place(&mut out, bytemuck::cast_slice(&g.upper_index));
    header.upper = place(&mut out, bytemuck::cast_slice(&g.upper));
    out[..GRAPH_HEADER_SIZE].copy_from_slice(bytemuck::bytes_of(&header));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::{build_graph, HnswParams};
    use sovereign_core::{
        padded_stride, push_padded_row, AlignedVec, MatrixRef, Metric, Xoshiro256pp,
    };

    /// Encode → parse → search entirely in memory (no mmap), so Miri can check the zero-copy parse
    /// and the unchecked-distance traversal of the frozen graph.
    #[test]
    fn encoded_graph_roundtrips_and_finds_stored_vectors() {
        let (n, dim) = if cfg!(miri) { (40, 16) } else { (1000, 32) };
        let stride = padded_stride(dim).unwrap();
        let k = sovereign_core::kernels();
        let mut rng = Xoshiro256pp::seed_from_u64(9);
        let mut buf = AlignedVec::new();
        let mut v = vec![0.0; dim];
        for _ in 0..n {
            rng.fill_gaussian(&mut v);
            k.normalize(&mut v).unwrap();
            push_padded_row(&mut buf, &v, stride).unwrap();
        }
        let m = MatrixRef::new(&buf, n, dim, stride).unwrap();
        let params = HnswParams { parallel: false, ..HnswParams::default() };
        let built = build_graph(m, Metric::Cosine, &params, k).unwrap();
        // The file places the graph section 4 KiB aligned; an AlignedVec gives the same guarantee.
        let bytes = AlignedVec::from_slice(&encode(&built));
        let g = HnswGraphRef::parse(&bytes, n).unwrap();
        assert_eq!(g.len(), n);
        assert!(HnswGraphRef::parse(&bytes, n + 1).is_err());

        let space = Space::new(k, m, Metric::Cosine);
        let mut scratch = LayerScratch::default();
        let mut found = 0;
        for i in 0..n {
            let mut out = TopK::new(1);
            // SAFETY: `space` is built over the matrix this graph indexes and rows are padded.
            unsafe { g.search(&space, m.row_padded(i), 32, &mut scratch, &mut out, 0) };
            found += usize::from(out.into_sorted_vec().first().map(|h| h.id) == Some(i as u64));
        }
        assert!(found * 100 >= n * 95, "self-recall {found}/{n}");
    }
}
