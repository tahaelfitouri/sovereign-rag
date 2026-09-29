//! Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2016).
//!
//! ```text
//!   layer 2   ●───────────────────────●                  few nodes, long edges
//!             │                       │
//!   layer 1   ●──────●────────●───────●───────●          P(level ≥ l) = M^-l
//!             │      │        │       │       │
//!   layer 0   ●──●──●──●──●──●──●──●──●──●──●──●──●      every node, ≤ m0 short edges
//! ```
//!
//! Search enters at the top, descends greedily (`ef = 1`) to layer 1, then runs a beam search of
//! width `ef` on layer 0. Expected cost is `O(log n)` distance evaluations, versus `O(n)` for a
//! flat scan — this is what makes sub-millisecond queries over millions of 1536-d vectors
//! possible, since a flat scan is bounded by memory bandwidth (≈6 GB per million vectors).

mod build;
mod graph;
pub(crate) mod search;

pub(crate) use build::build_graph;
pub(crate) use graph::encode;
pub use graph::HnswGraphRef;

use crate::error::{IndexError, Result};

/// HNSW construction parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HnswParams {
    /// Max links per node on layers ≥ 1. Typical: 12–48. Memory grows linearly in `m`.
    pub m: usize,
    /// Max links per node on layer 0. `2·m` is the value recommended by the paper.
    pub m0: usize,
    /// Beam width during construction. Higher = better graph, slower build. Typical: 100–400.
    pub ef_construction: usize,
    /// Seed for level assignment.
    pub seed: u64,
    /// Insert in parallel with rayon (non-deterministic edge choice under races).
    pub parallel: bool,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self { m: 16, m0: 32, ef_construction: 200, seed: 0x5EED_CAFE, parallel: true }
    }
}

impl HnswParams {
    /// Parameters with `m0 = 2·m`.
    #[must_use]
    pub fn with_m(m: usize) -> Self {
        Self { m, m0: 2 * m, ..Self::default() }
    }

    /// Validates parameter ranges.
    ///
    /// # Errors
    /// [`IndexError::InvalidConfig`] describing the offending parameter.
    pub fn validate(&self) -> Result<()> {
        if !(2..=256).contains(&self.m) {
            return Err(IndexError::InvalidConfig(format!("hnsw m={} not in 2..=256", self.m)));
        }
        if self.m0 < self.m || self.m0 > 1024 {
            return Err(IndexError::InvalidConfig(format!(
                "hnsw m0={} must be in m..=1024 (m={})",
                self.m0, self.m
            )));
        }
        if self.ef_construction < self.m {
            return Err(IndexError::InvalidConfig(format!(
                "hnsw ef_construction={} must be >= m={}",
                self.ef_construction, self.m
            )));
        }
        Ok(())
    }
}
