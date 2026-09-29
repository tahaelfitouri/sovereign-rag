//! # sovereign-index
//!
//! Memory-mapped, zero-copy vector storage and search.
//!
//! * [`SegmentWriter`] stages vectors (64-byte aligned, padded, normalized) and writes an
//!   immutable `.srag` file atomically, optionally with an HNSW graph built in parallel.
//! * [`Segment`] maps a file read-only and serves vectors, ids, payloads and the graph as borrowed
//!   slices — opening is O(1) regardless of file size.
//! * [`IndexStore`] manages a directory of segments behind an `ArcSwap<Snapshot>`: lock-free reads,
//!   atomic publish/compaction swaps, crash-safe manifest.
//!
//! ```no_run
//! use sovereign_index::{IndexStore, SearchParams, SegmentConfig, WriteOptions};
//! use sovereign_core::Metric;
//!
//! # fn main() -> sovereign_index::Result<()> {
//! let config = SegmentConfig { dim: 4, metric: Metric::Cosine, fingerprint: 42 };
//! let store = IndexStore::open_or_create("/tmp/demo-store", config)?;
//!
//! let mut w = store.writer()?;
//! w.push(1, &[1.0, 0.0, 0.0, 0.0], Some("east"))?;
//! w.push(2, &[0.0, 1.0, 0.0, 0.0], Some("north"))?;
//! store.add_segment(&w, &WriteOptions::default())?;
//!
//! let results = store.search(&[0.9, 0.1, 0.0, 0.0], &SearchParams::top(1))?;
//! assert_eq!(results.hits[0].id, 1);
//! assert_eq!(results.payload(&results.hits[0])?, Some("east"));
//! # Ok(()) }
//! ```

pub mod error;
mod flat;
pub mod format;
pub mod hnsw;
mod query;
mod segment;
pub mod store;
mod writer;

pub use error::{IndexError, Result};
pub use hnsw::{HnswGraphRef, HnswParams};
pub use query::{Hit, SearchMode, SearchParams, SearchScratch};
pub use segment::{AccessPattern, OpenOptions, Segment};
pub use store::{IndexStore, SearchResults, SegmentEntry, Snapshot, StoreConfig, StoreHit};
pub use writer::{SegmentConfig, SegmentWriter, WriteOptions, WriteReport};

use sovereign_core::Metric;

/// Minimal interface every searchable index implements.
///
/// This is the integration seam for alternative backends — a FAISS (`faiss-rs`) index, a remote
/// vector DB, or a GPU searcher can implement it and be used anywhere a [`Segment`] is.
pub trait VectorIndex: Send + Sync {
    /// Vector dimensionality.
    fn dim(&self) -> usize;
    /// Similarity metric.
    fn metric(&self) -> Metric;
    /// Number of indexed vectors.
    fn len(&self) -> usize;
    /// `true` if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// k-NN search.
    ///
    /// # Errors
    /// Implementation-defined; typically query validation errors.
    fn search(&self, query: &[f32], params: &SearchParams) -> Result<Vec<Hit>>;
}

impl VectorIndex for Segment {
    fn dim(&self) -> usize {
        Segment::dim(self)
    }
    fn metric(&self) -> Metric {
        Segment::metric(self)
    }
    fn len(&self) -> usize {
        Segment::len(self)
    }
    fn search(&self, query: &[f32], params: &SearchParams) -> Result<Vec<Hit>> {
        Segment::search(self, query, params)
    }
}
