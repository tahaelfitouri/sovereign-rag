//! # sovereign-core
//!
//! Low-level primitives for the `sovereign-rag` engine:
//!
//! | module        | what it provides                                                        |
//! |---------------|-------------------------------------------------------------------------|
//! | [`simd`]      | AVX-512 / AVX2+FMA / NEON / portable distance kernels, runtime dispatch |
//! | [`aligned`]   | 64-byte aligned growable buffers and cache-line wrappers                |
//! | [`matrix`]    | zero-copy row-major views with cache-line padded rows                   |
//! | [`arena`]     | bump allocator for per-batch scratch memory                             |
//! | [`topk`]      | allocation-free, deterministic bounded top-k selection                  |
//! | [`metric`]    | similarity metrics and their on-disk tags                               |
//! | [`hint`]      | stable-Rust branch prediction and prefetch hints                        |
//! | [`rng`]       | deterministic xoshiro256++ PRNG                                         |
//!
//! ```
//! use sovereign_core::{kernels, Metric};
//!
//! let k = kernels(); // best backend for this CPU, detected once
//! let a = [1.0f32, 2.0, 3.0, 4.0];
//! let b = [4.0f32, 3.0, 2.0, 1.0];
//! assert_eq!(k.dot(&a, &b), 20.0);
//! assert!((k.cosine(&a, &a) - 1.0).abs() < 1e-6);
//! assert_eq!(k.score(Metric::L2, &a, &a), 0.0);
//! ```
//!
//! ## Unsafe code policy
//!
//! `unsafe` is confined to: SIMD kernels (pointer loads under verified CPU features), the aligned
//! buffer and arena allocators, and unchecked row access after one-time shape validation. Every
//! `unsafe` block carries a `SAFETY:` comment stating the invariant it relies on, and
//! `unsafe_op_in_unsafe_fn` is denied so unsafe functions do not silently widen the unsafe scope.

pub mod aligned;
pub mod arena;
pub mod error;
pub mod hint;
pub mod matrix;
pub mod metric;
pub mod rng;
pub mod simd;
pub mod topk;

pub use aligned::{AlignedVec, CacheAligned, CACHE_LINE};
pub use arena::Arena;
pub use error::{CoreError, Result};
pub use matrix::{padded_stride, push_padded_row, MatrixRef, LANES_PER_LINE};
pub use metric::Metric;
pub use rng::Xoshiro256pp;
pub use simd::{init_backend, kernels, Backend, Kernels};
pub use topk::{Neighbor, TopK};
