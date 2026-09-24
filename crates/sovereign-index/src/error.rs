//! Error type for `sovereign-index`.

use std::path::PathBuf;

use sovereign_core::CoreError;
use thiserror::Error;

/// Convenience alias.
pub type Result<T, E = IndexError> = core::result::Result<T, E>;

/// Errors raised while building, opening or querying an index.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IndexError {
    /// Underlying I/O failure.
    #[error("I/O error on {path}: {source}")]
    Io {
        /// File or directory involved.
        path: PathBuf,
        /// OS error.
        #[source]
        source: std::io::Error,
    },

    /// A core primitive rejected its input (dimension mismatch, zero norm, ...).
    #[error(transparent)]
    Core(#[from] CoreError),

    /// The file does not start with the segment magic.
    #[error("{path}: not a sovereign segment (bad magic)")]
    BadMagic {
        /// Offending file.
        path: PathBuf,
    },

    /// The file was written by an incompatible format version.
    #[error("{path}: unsupported format version {found} (this build reads {supported})")]
    UnsupportedVersion {
        /// Offending file.
        path: PathBuf,
        /// Version found in the header.
        found: u32,
        /// Version this build understands.
        supported: u32,
    },

    /// A CRC32 check failed.
    #[error("{path}: {what} checksum mismatch (stored {stored:#010x}, computed {computed:#010x})")]
    Checksum {
        /// Offending file.
        path: PathBuf,
        /// Which region failed ("header" / "body").
        what: &'static str,
        /// Stored checksum.
        stored: u32,
        /// Recomputed checksum.
        computed: u32,
    },

    /// Structural corruption detected during validation.
    #[error("{path}: corrupt segment: {reason}")]
    Corrupt {
        /// Offending file.
        path: PathBuf,
        /// Human-readable reason.
        reason: String,
    },

    /// Invalid configuration or parameters.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// A vector contains NaN/∞.
    #[error("vector for id {id} contains a non-finite value at index {index}")]
    NonFinite {
        /// External id of the vector.
        id: u64,
        /// First offending component.
        index: usize,
    },

    /// Too many rows for a single segment (node ids are `u32`).
    #[error("segment row limit exceeded ({limit} rows)")]
    TooManyRows {
        /// Maximum rows per segment.
        limit: usize,
    },

    /// Segments with incompatible dimension / metric / embedder fingerprint were mixed.
    #[error("incompatible segment: {0}")]
    Incompatible(String),

    /// The store manifest could not be parsed.
    #[error("{path}: invalid manifest: {reason}")]
    Manifest {
        /// Manifest file.
        path: PathBuf,
        /// Reason.
        reason: String,
    },

    /// A row index was out of range.
    #[error("row {row} out of range (segment has {rows} rows)")]
    RowOutOfRange {
        /// Requested row.
        row: usize,
        /// Row count.
        rows: usize,
    },

    /// A payload was not valid UTF-8.
    #[error("payload of row {row} is not valid UTF-8")]
    InvalidPayload {
        /// Row index.
        row: usize,
    },
}

impl IndexError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io { path: path.into(), source }
    }

    pub(crate) fn corrupt(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        Self::Corrupt { path: path.into(), reason: reason.into() }
    }
}
