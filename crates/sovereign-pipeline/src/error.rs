//! Error type for `sovereign-pipeline`.

use std::path::PathBuf;

use thiserror::Error;

/// Convenience alias.
pub type Result<T, E = PipelineError> = core::result::Result<T, E>;

/// Errors raised by the ingestion pipeline.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PipelineError {
    /// Filesystem failure.
    #[error("I/O error on {path}: {source}")]
    Io {
        /// Path involved.
        path: PathBuf,
        /// OS error.
        #[source]
        source: std::io::Error,
    },

    /// The embedder failed.
    #[error("embedding failed: {0}")]
    Embed(String),

    /// The index rejected a row.
    #[error(transparent)]
    Index(#[from] sovereign_index::IndexError),

    /// A pipeline stage exited early, closing its channel.
    #[error("pipeline stage `{0}` terminated unexpectedly")]
    StageClosed(&'static str),

    /// A stage panicked or was cancelled.
    #[error("pipeline task failed: {0}")]
    Join(#[from] tokio::task::JoinError),

    /// Invalid configuration.
    #[error("invalid pipeline configuration: {0}")]
    Config(String),
}
