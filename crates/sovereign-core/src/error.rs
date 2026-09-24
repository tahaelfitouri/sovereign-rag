//! Error type for `sovereign-core`.

use thiserror::Error;

/// Convenience alias used throughout the crate.
pub type Result<T, E = CoreError> = core::result::Result<T, E>;

/// Every fallible operation in `sovereign-core` reports one of these.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoreError {
    /// Two vectors (or a vector and a matrix) disagree on dimensionality.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// The dimensionality the operation required.
        expected: usize,
        /// The dimensionality it was given.
        actual: usize,
    },

    /// A zero-copy view was requested over memory that violates the element alignment.
    #[error("buffer at {addr:#x} is not aligned to {required} bytes")]
    Misaligned {
        /// Address of the offending buffer.
        addr: usize,
        /// Required alignment in bytes.
        required: usize,
    },

    /// A buffer's length is inconsistent with the shape it is supposed to hold.
    #[error("invalid buffer layout: {0}")]
    InvalidLayout(&'static str),

    /// A size computation overflowed `usize`.
    #[error("capacity overflow")]
    CapacityOverflow,

    /// The requested SIMD backend is not available on this CPU / architecture.
    #[error("SIMD backend `{0}` is not supported on this CPU")]
    UnsupportedBackend(&'static str),

    /// A backend name could not be parsed.
    #[error("unknown SIMD backend `{0}` (expected one of: auto, scalar, avx2, avx512, neon)")]
    UnknownBackend(String),

    /// The global kernel table was already initialized with a different backend.
    #[error("SIMD backend already initialized as `{active}`; cannot switch to `{requested}`")]
    BackendAlreadyInitialized {
        /// Backend that is currently active.
        active: &'static str,
        /// Backend that was requested.
        requested: &'static str,
    },

    /// A metric tag or name could not be decoded.
    #[error("invalid metric `{0}` (expected one of: cosine, dot, l2)")]
    InvalidMetric(String),

    /// A vector with zero (or subnormal) L2 norm cannot be normalized for cosine similarity.
    #[error("vector has zero norm and cannot be normalized")]
    ZeroNorm,

    /// A vector contains NaN or infinity.
    #[error("vector contains a non-finite value at index {index}")]
    NonFinite {
        /// Index of the first offending component.
        index: usize,
    },
}
