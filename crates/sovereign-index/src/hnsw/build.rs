//! Concurrent HNSW construction.
//!
//! # Concurrency model
//!
//! Insertion is parallelized across rayon workers the same way hnswlib does it:
//!
//! * Each `(node, layer)` adjacency list sits behind its own `parking_lot::Mutex` (1 byte + the
//!   `Vec`). Readers copy a list into a thread-local buffer and release the lock *before* computing
//!   any distance, so locks are held for a ~64-element `memcpy`, not for SIMD work.
//! * A thread holds **at most one** adjacency lock at a time, and the entry-point lock is always
//!   acquired before any adjacency lock — so there is no lock-order cycle and no deadlock.
//! * An insert whose level exceeds the current top layer holds the entry-point lock for its whole
//!   duration (rare: probability `≈ 1/M` per level), so the graph's top never changes under a
//!   concurrent top-level insert.
//!
//! Races between inserts only affect *which* edges are chosen (a concurrent insert may not see a
//! node that is still being linked), which perturbs graph quality negligibly — the same trade
//! every production HNSW makes. With `parallel = false` the build is fully deterministic for a
//! given seed.

use parking_lot::Mutex;
use rayon::prelude::*;
use sovereign_core::{Kernels, MatrixRef, Metric, Xoshiro256pp};

use super::search::{
    greedy_step, search_layer, select_neighbors, Adjacency, Cand, LayerScratch, Space,
};
use super::HnswParams;
use crate::error::{IndexError, Result};
use crate::format::MAX_ROWS;

/// Hard cap on the number of layers (level ~ log_M(n); 16 covers M=2 up to 65k nodes and M≥4
/// beyond 4 billion).
const MAX_LEVEL: usize = 16;

struct Node {
    /// `links[layer]` for `layer in 0..=level`.
    links: Box<[Mutex<Vec<u32>>]>,
}

/// An in-memory HNSW graph ready to be serialized into the fixed-width on-disk layout.
#[derive(Debug, Clone)]
pub(crate) struct BuiltGraph {
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub seed: u64,
    pub max_level: u32,
    pub entry_point: u32,
    pub levels: Vec<u8>,
    pub layer0: Vec<u32>,
    pub upper_index: Vec<u32>,
    pub upper: Vec<u32>,
}

struct Builder<'a> {
    nodes: Vec<Node>,
    levels: Vec<u8>,
    entry: Mutex<(u32, usize)>,
    space: Space<'a>,
    params: HnswParams,
}

impl Adjacency for Builder<'_> {
    #[inline]
    fn neighbors_into(&self, node: u32, layer: usize, out: &mut Vec<u32>) {
        out.clear();
        if let Some(list) = self.nodes.get(node as usize).and_then(|n| n.links.get(layer)) {
            out.extend_from_slice(&list.lock());
        }
    }
}

#[derive(Default)]
struct BuildScratch {
    layer: LayerScratch,
    entry: Vec<Cand>,
    sorted: Vec<Cand>,
    selected: Vec<Cand>,
    prune: Vec<Cand>,
    pruned: Vec<Cand>,
}

impl Builder<'_> {
    fn insert(&self, q: u32, s: &mut BuildScratch) {
        let level = self.levels[q as usize] as usize;
        // SAFETY (for every `space.*` call below): all node ids come from `0..rows` — either `q`,
        // the entry point (a previously inserted node) or ids read from adjacency lists, which only
        // ever contain inserted node ids. `qv` is a padded row, so its length equals the stride.
        let qv = self.space.matrix.row_padded(q as usize);

        let mut entry_guard = Some(self.entry.lock());
        let (ep, top) = **entry_guard.as_ref().expect("guard just taken");
        if level <= top {
            entry_guard = None; // only top-level inserts keep the global lock
        }

        // SAFETY: see above.
        let mut cur = Cand { dist: unsafe { self.space.dist(qv, ep) }, id: ep };
        for layer in (level + 1..=top).rev() {
            // SAFETY: see above.
            cur = unsafe { greedy_step(self, &self.space, qv, cur, layer, &mut s.layer.nbuf) };
        }

        s.entry.clear();
        s.entry.push(cur);
        for layer in (0..=level.min(top)).rev() {
            let m_layer = if layer == 0 { self.params.m0 } else { self.params.m };
            // SAFETY: see above.
            unsafe {
                search_layer(
                    self,
                    &self.space,
                    qv,
                    &s.entry,
                    self.params.ef_construction,
                    layer,
                    &mut s.layer,
                );
            }
            s.sorted.clear();
            // Under concurrent inserts another thread may already have linked `q` into a list
            // reachable here (its beam from the layer above can contain `q`), so the search can
            // find `q` itself. Never select a node as its own neighbor.
            s.sorted.extend(s.layer.results.drain().filter(|c| c.id != q));
            s.sorted.sort_unstable();
            // SAFETY: see above.
            unsafe { select_neighbors(&self.space, &s.sorted, m_layer, &mut s.selected) };

            {
                let mut own = self.nodes[q as usize].links[layer].lock();
                own.clear();
                own.extend(s.selected.iter().map(|c| c.id));
            }
            for i in 0..s.selected.len() {
                let nb = s.selected[i].id;
                self.connect(nb, q, layer, m_layer, s);
            }
            // Distances to `q` are layer-independent, so the whole beam seeds the next layer
            // without recomputation (the paper's `ep ← W`).
            core::mem::swap(&mut s.entry, &mut s.sorted);
        }

        if let Some(mut guard) = entry_guard {
            if level > guard.1 {
                *guard = (q, level);
            }
        }
    }

    /// Adds the back-link `nb -> q`, re-running the selection heuristic if `nb` is full.
    fn connect(&self, nb: u32, q: u32, layer: usize, m_layer: usize, s: &mut BuildScratch) {
        if nb == q {
            return; // no self-loops (see `insert`)
        }
        let Some(list) = self.nodes[nb as usize].links.get(layer) else { return };
        let mut links = list.lock();
        if links.contains(&q) {
            return;
        }
        if links.len() < m_layer {
            links.push(q);
            return;
        }
        s.prune.clear();
        // SAFETY: `nb`, `q` and every id already in `links` are inserted nodes, hence `< rows`.
        unsafe {
            s.prune.push(Cand { dist: self.space.dist_nodes(nb, q), id: q });
            for &x in links.iter() {
                s.prune.push(Cand { dist: self.space.dist_nodes(nb, x), id: x });
            }
        }
        s.prune.sort_unstable();
        // SAFETY: as above.
        unsafe { select_neighbors(&self.space, &s.prune, m_layer, &mut s.pruned) };
        links.clear();
        links.extend(s.pruned.iter().map(|c| c.id));
    }

    fn freeze(self) -> BuiltGraph {
        let n = self.nodes.len();
        let (m, m0) = (self.params.m, self.params.m0);
        let (entry_point, max_level) = *self.entry.lock();
        let mut layer0 = vec![0u32; n * (m0 + 1)];
        let mut upper_index = vec![u32::MAX; n];
        let mut upper = Vec::new();
        let mut upper_rows = 0u32;

        for (i, node) in self.nodes.into_iter().enumerate() {
            let mut links = node.links.into_vec().into_iter().map(Mutex::into_inner);
            let l0 = links.next().unwrap_or_default();
            let row = &mut layer0[i * (m0 + 1)..(i + 1) * (m0 + 1)];
            let len = l0.len().min(m0);
            row[0] = len as u32;
            row[1..=len].copy_from_slice(&l0[..len]);

            let mut first = true;
            for l in links {
                if first {
                    upper_index[i] = upper_rows;
                    first = false;
                }
                let len = l.len().min(m);
                upper.push(len as u32);
                upper.extend_from_slice(&l[..len]);
                upper.resize(upper.len() + (m - len), 0);
                upper_rows += 1;
            }
        }

        BuiltGraph {
            m,
            m0,
            ef_construction: self.params.ef_construction,
            seed: self.params.seed,
            max_level: max_level as u32,
            entry_point,
            levels: self.levels,
            layer0,
            upper_index,
            upper,
        }
    }
}

/// Draws the level of every node: `⌊−ln(U) · mL⌋`, `mL = 1/ln(M)`, i.e. a geometric distribution
/// with `P(level ≥ l) = M^{−l}`.
fn assign_levels(n: usize, params: &HnswParams) -> Vec<u8> {
    let ml = 1.0 / (params.m as f64).ln();
    let mut rng = Xoshiro256pp::seed_from_u64(params.seed);
    (0..n)
        .map(|_| {
            let u = 1.0 - rng.next_f64(); // (0, 1]
            ((-u.ln() * ml) as usize).min(MAX_LEVEL) as u8
        })
        .collect()
}

/// Builds an HNSW graph over the rows of `matrix` (rows must already be normalized for cosine).
pub(crate) fn build_graph(
    matrix: MatrixRef<'_>,
    metric: Metric,
    params: &HnswParams,
    kernels: &'static Kernels,
) -> Result<BuiltGraph> {
    params.validate()?;
    let n = matrix.rows();
    if n > MAX_ROWS {
        return Err(IndexError::TooManyRows { limit: MAX_ROWS });
    }
    let levels = assign_levels(n, params);
    let nodes = levels
        .iter()
        .map(|&l| Node {
            links: (0..=l as usize)
                .map(|layer| {
                    let cap = if layer == 0 { params.m0 } else { params.m } + 1;
                    Mutex::new(Vec::with_capacity(cap))
                })
                .collect(),
        })
        .collect();

    let builder = Builder {
        nodes,
        entry: Mutex::new((0, levels.first().copied().unwrap_or(0) as usize)),
        levels,
        space: Space::new(kernels, matrix, metric),
        params: params.clone(),
    };

    if n > 1 {
        if params.parallel {
            (1..n as u32).into_par_iter().for_each_init(BuildScratch::default, |s, q| {
                builder.insert(q, s);
            });
        } else {
            let mut s = BuildScratch::default();
            for q in 1..n as u32 {
                builder.insert(q, &mut s);
            }
        }
    }
    Ok(builder.freeze())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sovereign_core::{padded_stride, push_padded_row, AlignedVec};

    fn dataset(n: usize, dim: usize, seed: u64) -> (AlignedVec<f32>, usize) {
        let stride = padded_stride(dim).unwrap();
        let mut rng = Xoshiro256pp::seed_from_u64(seed);
        let mut buf = AlignedVec::new();
        let mut v = vec![0.0; dim];
        for _ in 0..n {
            rng.fill_gaussian(&mut v);
            sovereign_core::kernels().normalize(&mut v).unwrap();
            push_padded_row(&mut buf, &v, stride).unwrap();
        }
        (buf, stride)
    }

    #[test]
    fn levels_follow_geometric_distribution() {
        let p = HnswParams::default();
        let levels = assign_levels(100_000, &p);
        let above0 = levels.iter().filter(|&&l| l >= 1).count() as f64 / 1e5;
        assert!((above0 - 1.0 / p.m as f64).abs() < 0.01, "P(l>=1) = {above0}");
    }

    #[test]
    fn frozen_layout_is_consistent() {
        // Miri-sized under `cfg(miri)`; the full size natively.
        let n = if cfg!(miri) { 40 } else { 500 };
        let (buf, stride) = dataset(n, 24, 1);
        let m = MatrixRef::new(&buf, n, 24, stride).unwrap();
        let params =
            HnswParams { m: 8, m0: 16, ef_construction: 64, parallel: false, ..Default::default() };
        let g = build_graph(m, Metric::Cosine, &params, sovereign_core::kernels()).unwrap();
        assert_eq!(g.levels.len(), n);
        assert_eq!(g.layer0.len(), n * 17);
        assert_eq!(g.levels[g.entry_point as usize] as u32, g.max_level);
        for i in 0..n {
            let len = g.layer0[i * 17] as usize;
            assert!((1..=16).contains(&len), "node {i} has {len} layer-0 links");
            assert!(g.layer0[i * 17 + 1..i * 17 + 1 + len]
                .iter()
                .all(|&x| (x as usize) < n && x as usize != i));
            let lvl = g.levels[i] as usize;
            assert_eq!(g.upper_index[i] == u32::MAX, lvl == 0);
        }
        let upper_rows: usize = g.levels.iter().map(|&l| l as usize).sum();
        assert_eq!(g.upper.len(), upper_rows * 9);
    }

    #[test]
    fn parallel_build_links_are_valid() {
        let n = if cfg!(miri) { 48 } else { 2000 };
        let (buf, stride) = dataset(n, 16, 4);
        let m = MatrixRef::new(&buf, n, 16, stride).unwrap();
        let params = HnswParams { m: 6, m0: 12, ef_construction: 24, ..Default::default() };
        assert!(params.parallel);
        let g = build_graph(m, Metric::Cosine, &params, sovereign_core::kernels()).unwrap();
        assert_eq!(g.levels[g.entry_point as usize] as u32, g.max_level);
        for i in 0..n {
            let row = &g.layer0[i * 13..(i + 1) * 13];
            let len = row[0] as usize;
            assert!((1..=12).contains(&len), "node {i} has {len} layer-0 links");
            assert!(row[1..=len].iter().all(|&x| (x as usize) < n && x as usize != i));
        }
    }

    #[test]
    fn sequential_build_is_deterministic() {
        let n = if cfg!(miri) { 30 } else { 300 };
        let (buf, stride) = dataset(n, 16, 2);
        let m = MatrixRef::new(&buf, n, 16, stride).unwrap();
        let params = HnswParams { parallel: false, ..Default::default() };
        let a = build_graph(m, Metric::Cosine, &params, sovereign_core::kernels()).unwrap();
        let b = build_graph(m, Metric::Cosine, &params, sovereign_core::kernels()).unwrap();
        assert_eq!(a.layer0, b.layer0);
        assert_eq!(a.upper, b.upper);
    }

    #[test]
    fn degenerate_sizes() {
        let (buf, stride) = dataset(1, 8, 3);
        let m = MatrixRef::new(&buf, 1, 8, stride).unwrap();
        let g = build_graph(m, Metric::Cosine, &HnswParams::default(), sovereign_core::kernels())
            .unwrap();
        assert_eq!(g.entry_point, 0);
        assert_eq!(g.layer0[0], 0);
        let empty = MatrixRef::new(&[], 0, 8, stride).unwrap();
        let g =
            build_graph(empty, Metric::Cosine, &HnswParams::default(), sovereign_core::kernels())
                .unwrap();
        assert!(g.levels.is_empty());
    }
}
