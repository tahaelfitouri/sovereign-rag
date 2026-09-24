//! # sovereign-pipeline
//!
//! Asynchronous, backpressure-aware document ingestion for `sovereign-rag`:
//!
//! * [`ring`] — lock-free async SPSC ring buffers (the stage transport);
//! * [`chunker`] — zero-allocation, structure-aware Markdown/code chunking over `&str`;
//! * [`embed`] — the [`Embedder`] trait and an offline feature-hashing embedder;
//! * [`IngestPipeline`] — reader → chunker → N embed workers → segment writer;
//! * [`metrics`] — false-sharing-free live counters for progress/throughput reporting.

pub mod chunker;
pub mod embed;
pub mod error;
pub mod ingest;
pub mod metrics;
pub mod payload;
pub mod ring;

pub use chunker::{estimate_tokens, Chunk, ChunkKind, Chunker, ChunkerConfig, SourceKind};
pub use embed::{EmbedError, Embedder, HashEmbedder};
pub use error::{PipelineError, Result};
pub use ingest::{discover, ChunkRecord, Document, IngestConfig, IngestPipeline, IngestReport};
pub use metrics::{MetricsSnapshot, PipelineMetrics};
