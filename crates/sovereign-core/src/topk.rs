//! Allocation-free bounded top-k selection.
//!
//! [`TopK`] keeps the `k` best `(id, score)` pairs in a binary **min**-heap keyed on score, so the
//! root is the *worst* retained candidate and the admission test is a single comparison against
//! [`TopK::threshold`]. For a scan over `n` candidates this is `O(n + m log k)` where `m` is the
//! number of admissions — and for random data `m ≈ k·ln(n/k)`, so almost every candidate is
//! rejected by one predictable branch.
//!
//! # Determinism
//!
//! Ties are broken by id (lower id wins). The result set is therefore a pure function of the
//! candidate multiset, independent of scan order — parallel partitioned scans merged with
//! [`TopK::merge`] return bit-identical results to a sequential scan.

use core::cmp::Ordering;

use crate::hint::unlikely;

/// A scored candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// Candidate identifier (row index, packed segment/row key or external id).
    pub id: u64,
    /// Similarity score; higher is better.
    pub score: f32,
}

impl Neighbor {
    /// Total order where `Greater` means *better*: higher score, then lower id.
    #[inline(always)]
    #[must_use]
    pub fn cmp_quality(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score).then_with(|| other.id.cmp(&self.id))
    }

    #[inline(always)]
    fn worse_than(&self, other: &Self) -> bool {
        self.cmp_quality(other) == Ordering::Less
    }
}

/// Bounded collector of the `k` best candidates. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct TopK {
    k: usize,
    heap: Vec<Neighbor>,
}

impl TopK {
    /// Creates a collector for the `k` best candidates. Allocates once, up front.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k, heap: Vec::with_capacity(k) }
    }

    /// Clears the collector and changes `k`, reusing the allocation when possible.
    pub fn reset(&mut self, k: usize) {
        self.heap.clear();
        if k > self.heap.capacity() {
            self.heap.reserve_exact(k);
        }
        self.k = k;
    }

    /// Capacity `k`.
    #[inline]
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Number of retained candidates (`≤ k`).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// `true` if nothing has been retained.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// `true` once `k` candidates are retained.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// Score a new candidate must beat to be admitted (`-∞` while not full).
    #[inline(always)]
    #[must_use]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::NEG_INFINITY
        } else {
            self.heap[0].score
        }
    }

    /// Offers a candidate. Returns `true` if it was retained. NaN scores are rejected.
    #[inline(always)]
    pub fn push(&mut self, id: u64, score: f32) -> bool {
        // Fast reject: the overwhelmingly common case in a large scan.
        if unlikely(self.heap.len() < self.k) {
            if score.is_nan() {
                return false;
            }
            self.heap.push(Neighbor { id, score });
            self.sift_up(self.heap.len() - 1);
            return true;
        }
        if self.k == 0 || score < self.heap[0].score {
            return false;
        }
        self.replace_root(Neighbor { id, score })
    }

    #[inline(never)]
    fn replace_root(&mut self, cand: Neighbor) -> bool {
        if cand.score.is_nan() || !self.heap[0].worse_than(&cand) {
            return false;
        }
        self.heap[0] = cand;
        self.sift_down(0);
        true
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.heap[i].worse_than(&self.heap[parent]) {
                self.heap.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let l = 2 * i + 1;
            if l >= n {
                break;
            }
            let r = l + 1;
            let worst_child = if r < n && self.heap[r].worse_than(&self.heap[l]) { r } else { l };
            if self.heap[worst_child].worse_than(&self.heap[i]) {
                self.heap.swap(i, worst_child);
                i = worst_child;
            } else {
                break;
            }
        }
    }

    /// Folds another collector's candidates into this one.
    pub fn merge(&mut self, other: &TopK) {
        for n in &other.heap {
            self.push(n.id, n.score);
        }
    }

    /// Retained candidates in heap (unspecified) order.
    #[must_use]
    pub fn as_unsorted_slice(&self) -> &[Neighbor] {
        &self.heap
    }

    /// Sorts the retained candidates best-first and returns them. The collector must be
    /// [`reset`](Self::reset) before reuse (the heap property is destroyed).
    pub fn sort_best_first(&mut self) -> &[Neighbor] {
        self.heap.sort_unstable_by(|a, b| b.cmp_quality(a));
        &self.heap
    }

    /// Consumes the collector, returning candidates best-first.
    #[must_use]
    pub fn into_sorted_vec(mut self) -> Vec<Neighbor> {
        self.heap.sort_unstable_by(|a, b| b.cmp_quality(a));
        self.heap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xoshiro256pp;

    fn brute_force(scores: &[f32], k: usize) -> Vec<Neighbor> {
        let mut all: Vec<Neighbor> = scores
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_nan())
            .map(|(i, &s)| Neighbor { id: i as u64, score: s })
            .collect();
        all.sort_unstable_by(|a, b| b.cmp_quality(a));
        all.truncate(k);
        all
    }

    #[test]
    fn matches_brute_force() {
        let mut rng = Xoshiro256pp::seed_from_u64(3);
        for &k in &[0usize, 1, 5, 64] {
            let scores: Vec<f32> = (0..5000).map(|_| rng.next_f32()).collect();
            let mut t = TopK::new(k);
            for (i, &s) in scores.iter().enumerate() {
                t.push(i as u64, s);
            }
            assert_eq!(t.into_sorted_vec(), brute_force(&scores, k));
        }
    }

    #[test]
    fn ties_break_by_id_and_merge_is_order_independent() {
        let scores = [1.0f32, 3.0, 3.0, 2.0, 3.0, 0.5, 3.0];
        let mut seq = TopK::new(3);
        for (i, &s) in scores.iter().enumerate() {
            seq.push(i as u64, s);
        }
        let mut a = TopK::new(3);
        let mut b = TopK::new(3);
        for (i, &s) in scores.iter().enumerate().rev() {
            if i % 2 == 0 {
                a.push(i as u64, s)
            } else {
                b.push(i as u64, s)
            };
        }
        b.merge(&a);
        let expected: Vec<u64> = vec![1, 2, 4];
        assert_eq!(seq.into_sorted_vec().iter().map(|n| n.id).collect::<Vec<_>>(), expected);
        assert_eq!(b.into_sorted_vec().iter().map(|n| n.id).collect::<Vec<_>>(), expected);
    }

    #[test]
    fn nan_rejected_and_threshold() {
        let mut t = TopK::new(2);
        assert_eq!(t.threshold(), f32::NEG_INFINITY);
        assert!(!t.push(0, f32::NAN));
        assert!(t.push(1, 0.5));
        assert!(t.push(2, 0.7));
        assert_eq!(t.threshold(), 0.5);
        assert!(!t.push(3, f32::NAN));
        assert!(!t.push(4, 0.1));
        assert!(t.push(5, 0.9));
        assert_eq!(t.threshold(), 0.7);
        t.reset(1);
        assert!(t.is_empty());
        assert_eq!(t.k(), 1);
    }
}
