//! Query-side types: parameters, results and reusable scratch space.

use sovereign_core::{AlignedVec, Kernels, Metric, TopK};

use crate::error::{IndexError, Result};
use crate::hnsw::search::LayerScratch;

/// How a segment should be searched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SearchMode {
    /// HNSW if the segment has a graph, otherwise an exact scan.
    #[default]
    Auto,
    /// Always scan every vector (100% recall, `O(n)`).
    Exact,
    /// Require HNSW; segments without a graph fall back to an exact scan.
    Approximate,
}

/// Search parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchParams {
    /// Number of neighbors to return.
    pub k: usize,
    /// HNSW beam width on layer 0 (clamped to `≥ k`). Higher = better recall, slower.
    pub ef: usize,
    /// Strategy.
    pub mode: SearchMode,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self { k: 10, ef: 64, mode: SearchMode::Auto }
    }
}

impl SearchParams {
    /// Parameters for the `k` nearest neighbors with default `ef`.
    #[must_use]
    pub fn top(k: usize) -> Self {
        Self { k, ..Self::default() }
    }

    /// Builder: set `ef`.
    #[must_use]
    pub fn ef(mut self, ef: usize) -> Self {
        self.ef = ef;
        self
    }

    /// Builder: set the mode.
    #[must_use]
    pub fn mode(mut self, mode: SearchMode) -> Self {
        self.mode = mode;
        self
    }
}

/// One search result from a single segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Row inside the segment.
    pub row: u32,
    /// External id stored for that row.
    pub id: u64,
    /// Similarity score (higher is better; `−‖x−q‖²` for L2).
    pub score: f32,
}

/// Reusable per-thread search buffers. Keep one per worker thread and pass it to
/// `search_with` to make steady-state queries allocation-free.
#[derive(Debug)]
pub struct SearchScratch {
    pub(crate) query: AlignedVec<f32>,
    pub(crate) topk: TopK,
    pub(crate) layer: LayerScratch,
}

impl Default for SearchScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchScratch {
    /// Creates empty scratch space; buffers grow on first use and are then reused.
    #[must_use]
    pub fn new() -> Self {
        Self { query: AlignedVec::new(), topk: TopK::new(0), layer: LayerScratch::default() }
    }

    /// Copies `query` into the internal 64-byte aligned buffer, zero-pads it to `stride` and
    /// normalizes it for cosine. Returns nothing; the prepared query lives in `self.query`.
    pub(crate) fn prepare(
        &mut self,
        query: &[f32],
        dim: usize,
        stride: usize,
        metric: Metric,
        kernels: &Kernels,
    ) -> Result<()> {
        if query.len() != dim {
            return Err(sovereign_core::CoreError::DimensionMismatch {
                expected: dim,
                actual: query.len(),
            }
            .into());
        }
        if let Some(index) = query.iter().position(|x| !x.is_finite()) {
            return Err(IndexError::NonFinite { id: u64::MAX, index });
        }
        self.query.clear();
        self.query.extend_from_slice(query);
        self.query.resize(stride, 0.0);
        if metric.normalizes() {
            kernels.normalize(&mut self.query)?;
        }
        Ok(())
    }
}
