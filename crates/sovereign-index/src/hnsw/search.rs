//! Search primitives shared by the concurrent builder and the frozen (mmap) graph.
//!
//! Both sides implement [`Adjacency`]; the layer search itself is written once, so the graph that
//! is searched at query time is traversed by exactly the code that built it.

use core::cmp::Ordering;
use std::collections::BinaryHeap;

use sovereign_core::hint::prefetch_lines;
use sovereign_core::{Kernels, MatrixRef, Metric};

/// A node with its distance to the current query (lower = closer).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cand {
    pub dist: f32,
    pub id: u32,
}

impl Eq for Cand {}

impl Ord for Cand {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist).then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Cand {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Min-heap adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Nearest(pub core::cmp::Reverse<Cand>);

/// Read access to a layered neighbor graph.
pub(crate) trait Adjacency {
    /// Replaces `out` with the neighbors of `node` on `layer` (empty if none / out of range).
    fn neighbors_into(&self, node: u32, layer: usize, out: &mut Vec<u32>);
}

/// "Visited" set with O(1) reset: each slot stores the epoch in which it was last marked.
/// Bumping the epoch invalidates every mark without touching memory; we only clear the array when
/// the `u32` epoch wraps (once every ~4 billion searches).
#[derive(Debug, Default)]
pub(crate) struct VisitedSet {
    marks: Vec<u32>,
    epoch: u32,
}

impl VisitedSet {
    /// Starts a new search over `n` nodes.
    #[inline]
    pub fn prepare(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    /// `true` if `id` was already visited in this epoch (out-of-range ids count as visited).
    #[inline(always)]
    pub fn contains(&self, id: u32) -> bool {
        self.marks.get(id as usize).is_none_or(|&m| m == self.epoch)
    }

    /// Marks `id`; returns `true` if it was not yet visited in this epoch.
    ///
    /// `id` must be `< n` from the last [`prepare`](Self::prepare) (callers validate node ids).
    #[inline(always)]
    pub fn insert(&mut self, id: u32) -> bool {
        let slot = &mut self.marks[id as usize];
        if *slot == self.epoch {
            false
        } else {
            *slot = self.epoch;
            true
        }
    }
}

/// Reusable per-thread buffers for layer searches. After warm-up, a search allocates nothing.
#[derive(Debug, Default)]
pub(crate) struct LayerScratch {
    pub visited: VisitedSet,
    /// Frontier, closest first.
    pub candidates: BinaryHeap<Nearest>,
    /// Best `ef` found so far, farthest on top.
    pub results: BinaryHeap<Cand>,
    /// Neighbor list copy of the node being expanded.
    pub nbuf: Vec<u32>,
}

/// Distance oracle over a matrix of (padded) rows.
///
/// For cosine/dot the distance is `-dot` (rows are pre-normalized for cosine); for L2 it is the
/// squared Euclidean distance. In both cases `score = -distance`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Space<'a> {
    pub kernels: &'static Kernels,
    pub matrix: MatrixRef<'a>,
    pub l2: bool,
}

impl<'a> Space<'a> {
    pub fn new(kernels: &'static Kernels, matrix: MatrixRef<'a>, metric: Metric) -> Self {
        Self { kernels, matrix, l2: metric == Metric::L2 }
    }

    /// Number of rows.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.matrix.rows()
    }

    /// Distance from `q` (length `stride`) to row `id`.
    ///
    /// # Safety
    /// `id < self.len()` and `q.len() == self.matrix.stride()`.
    #[inline(always)]
    pub unsafe fn dist(&self, q: &[f32], id: u32) -> f32 {
        // SAFETY: caller guarantees `id < rows`.
        let row = unsafe { self.matrix.row_padded_unchecked(id as usize) };
        // SAFETY: caller guarantees `q.len() == stride == row.len()`.
        unsafe {
            if self.l2 {
                self.kernels.l2_sq_unchecked(q, row)
            } else {
                -self.kernels.dot_unchecked(q, row)
            }
        }
    }

    /// Distance between rows `a` and `b`.
    ///
    /// # Safety
    /// `a < self.len()` and `b < self.len()`.
    #[inline(always)]
    pub unsafe fn dist_nodes(&self, a: u32, b: u32) -> f32 {
        // SAFETY: caller guarantees `a < rows`; the padded row has length `stride`.
        let qa = unsafe { self.matrix.row_padded_unchecked(a as usize) };
        // SAFETY: caller guarantees `b < rows`, and `qa.len() == stride`.
        unsafe { self.dist(qa, b) }
    }

    /// Prefetch the first cache lines of row `id` (a hint; `id` need not be valid).
    #[inline(always)]
    pub fn prefetch(&self, id: u32) {
        prefetch_lines(self.matrix.row_ptr(id as usize), 4);
    }
}

/// Greedy best-first descent on one layer with `ef = 1` (used on the upper layers).
///
/// # Safety
/// `cur.id < space.len()` and `q.len() == space.matrix.stride()`.
pub(crate) unsafe fn greedy_step<A: Adjacency>(
    adj: &A,
    space: &Space<'_>,
    q: &[f32],
    mut cur: Cand,
    layer: usize,
    nbuf: &mut Vec<u32>,
) -> Cand {
    let n = space.len();
    loop {
        let mut changed = false;
        adj.neighbors_into(cur.id, layer, nbuf);
        for &nb in nbuf.iter() {
            if nb as usize >= n {
                continue; // corrupt id: skip rather than trust it
            }
            // SAFETY: `nb < n` checked above; `q` length guaranteed by caller.
            let d = unsafe { space.dist(q, nb) };
            if d < cur.dist {
                cur = Cand { dist: d, id: nb };
                changed = true;
            }
        }
        if !changed {
            return cur;
        }
    }
}

/// Beam search on one layer (Malkov & Yashunin, Algorithm 2). Leaves the best `ef` candidates in
/// `s.results` (a max-heap on distance).
///
/// # Safety
/// Every `entry[i].id < space.len()` and `q.len() == space.matrix.stride()`.
pub(crate) unsafe fn search_layer<A: Adjacency>(
    adj: &A,
    space: &Space<'_>,
    q: &[f32],
    entry: &[Cand],
    ef: usize,
    layer: usize,
    s: &mut LayerScratch,
) {
    let n = space.len();
    let ef = ef.max(1);
    s.visited.prepare(n);
    s.candidates.clear();
    s.results.clear();

    for &e in entry {
        if s.visited.insert(e.id) {
            s.candidates.push(Nearest(core::cmp::Reverse(e)));
            s.results.push(e);
            if s.results.len() > ef {
                s.results.pop();
            }
        }
    }

    while let Some(Nearest(core::cmp::Reverse(c))) = s.candidates.pop() {
        let worst = s.results.peek().map_or(f32::INFINITY, |w| w.dist);
        if c.dist > worst && s.results.len() >= ef {
            break; // frontier is farther than everything we keep: converged
        }
        adj.neighbors_into(c.id, layer, &mut s.nbuf);

        // Prefetch every unvisited neighbor's vector before computing any distance, so the
        // DRAM/LLC misses overlap instead of serializing (hnswlib does the same).
        for &nb in &s.nbuf {
            if !s.visited.contains(nb) {
                space.prefetch(nb);
            }
        }

        for i in 0..s.nbuf.len() {
            let nb = s.nbuf[i];
            if nb as usize >= n || !s.visited.insert(nb) {
                continue;
            }
            // SAFETY: `nb < n` checked above; `q` length guaranteed by caller.
            let d = unsafe { space.dist(q, nb) };
            let worst = s.results.peek().map_or(f32::INFINITY, |w| w.dist);
            if s.results.len() < ef || d < worst {
                let cand = Cand { dist: d, id: nb };
                s.candidates.push(Nearest(core::cmp::Reverse(cand)));
                s.results.push(cand);
                if s.results.len() > ef {
                    s.results.pop();
                }
            }
        }
    }
}

/// Neighbor-selection heuristic (Algorithm 4 without candidate extension, as in hnswlib).
///
/// `sorted` must be ordered by ascending distance to the base node. A candidate is kept only if
/// it is closer to the base than to every already-selected neighbor, which spreads links across
/// directions (keeps "highway" edges between clusters) instead of wiring a node only to its own
/// dense neighborhood.
///
/// # Safety
/// Every id in `sorted` must be `< space.len()`.
pub(crate) unsafe fn select_neighbors(
    space: &Space<'_>,
    sorted: &[Cand],
    m: usize,
    out: &mut Vec<Cand>,
) {
    out.clear();
    for &c in sorted {
        if out.len() >= m {
            break;
        }
        // SAFETY: all ids come from `sorted`, which the caller guarantees are in range.
        let good = out.iter().all(|s| unsafe { space.dist_nodes(s.id, c.id) } >= c.dist);
        if good {
            out.push(c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visited_epochs() {
        let mut v = VisitedSet::default();
        v.prepare(4);
        assert!(v.insert(1));
        assert!(!v.insert(1));
        v.prepare(4);
        assert!(v.insert(1));
        v.epoch = u32::MAX;
        v.prepare(8);
        assert_eq!(v.epoch, 1);
        assert!(v.insert(7));
    }

    #[test]
    fn cand_ordering() {
        let a = Cand { dist: 0.1, id: 5 };
        let b = Cand { dist: 0.1, id: 6 };
        let c = Cand { dist: -1.0, id: 9 };
        let mut v = vec![b, a, c];
        v.sort();
        assert_eq!(v, vec![c, a, b]);
    }
}
