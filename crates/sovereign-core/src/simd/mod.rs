//! SIMD distance kernels with one-time runtime CPU dispatch.
//!
//! # Dispatch model
//!
//! Binaries are compiled for the baseline ISA (x86-64-v1 / armv8-a), so they run anywhere. The
//! first call to [`kernels()`] probes the CPU (`cpuid` via `is_x86_feature_detected!`) and
//! installs a `&'static Kernels` table of function pointers for the best backend:
//!
//! ```text
//!   kernels() ──► OnceLock<&'static Kernels> ──► AVX512 | AVX2 | NEON | SCALAR tables
//!                  (one atomic acquire load        │
//!                   after initialization)          └─► k.dot(a, b) = one indirect call
//! ```
//!
//! Hot loops fetch the table **once** and call through it per vector. The indirect call is
//! perfectly predicted after the first iteration and costs ~1 ns against the ~100+ ns of a
//! 1536-d dot product — cheaper than the alternative of monomorphizing every search loop per
//! backend, and it keeps the index crate free of `target_feature` plumbing.
//!
//! Set `SOVEREIGN_SIMD=scalar|avx2|avx512|neon` to force a backend (ignored if unsupported), or
//! call [`init_backend`] before first use.

#[cfg(target_arch = "aarch64")]
mod neon;
mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86;

use core::fmt;
use core::str::FromStr;
use std::sync::OnceLock;

use crate::error::{CoreError, Result};
use crate::hint::unlikely;
use crate::metric::Metric;

/// Portable reference kernels, exposed for benchmarking and verification.
pub mod reference {
    pub use super::scalar::{cosine_naive, dot_naive, l2_sq_naive};
}

/// Environment variable consulted on first use of [`kernels()`].
pub const BACKEND_ENV: &str = "SOVEREIGN_SIMD";

/// Instruction-set backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// Portable Rust with 8 independent accumulators (auto-vectorized to SSE2/NEON by LLVM).
    Scalar,
    /// x86_64 AVX2 + FMA, 256-bit registers.
    Avx2,
    /// x86_64 AVX-512F, 512-bit registers with masked tails.
    Avx512,
    /// AArch64 Advanced SIMD, 128-bit registers.
    Neon,
}

impl Backend {
    /// Every backend, best first.
    pub const ALL: [Backend; 4] = [Backend::Avx512, Backend::Avx2, Backend::Neon, Backend::Scalar];

    /// Short lowercase name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Neon => "neon",
        }
    }

    /// Number of `f32` lanes processed per vector instruction.
    #[must_use]
    pub const fn lanes(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Avx2 => 8,
            Self::Avx512 => 16,
            Self::Neon => 4,
        }
    }

    /// Register width in bits.
    #[must_use]
    pub const fn register_bits(self) -> usize {
        self.lanes() * 32
    }

    /// Whether this CPU can execute the backend.
    #[must_use]
    pub fn is_supported(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx512 => std::arch::is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// The best supported backend on this machine.
    #[must_use]
    pub fn detect() -> Self {
        Self::ALL.into_iter().find(|b| b.is_supported()).unwrap_or(Self::Scalar)
    }

    /// All backends supported on this machine, best first.
    #[must_use]
    pub fn supported() -> Vec<Self> {
        Self::ALL.into_iter().filter(|b| b.is_supported()).collect()
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Backend {
    type Err = CoreError;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "scalar" | "portable" => Ok(Self::Scalar),
            "avx2" => Ok(Self::Avx2),
            "avx512" | "avx-512" | "avx512f" => Ok(Self::Avx512),
            "neon" => Ok(Self::Neon),
            _ => Err(CoreError::UnknownBackend(s.to_owned())),
        }
    }
}

type PairFn = unsafe fn(&[f32], &[f32]) -> f32;
type UnaryFn = unsafe fn(&[f32]) -> f32;
type X4Fn = unsafe fn(&[f32], [&[f32]; 4]) -> [f32; 4];

/// A table of distance kernels for one backend.
///
/// # Invariant
///
/// A `&'static Kernels` can only be obtained through [`Kernels::for_backend`], [`kernels()`] or
/// [`init_backend`], all of which verify that the CPU supports the backend. The function pointers
/// are private, so safe code can never call a kernel on an unsupported CPU. That invariant is
/// what makes the `unsafe` calls in the safe wrappers below sound.
pub struct Kernels {
    backend: Backend,
    dot: PairFn,
    l2_sq: PairFn,
    cosine: PairFn,
    norm_sq: UnaryFn,
    dot_x4: X4Fn,
}

impl fmt::Debug for Kernels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernels").field("backend", &self.backend).finish_non_exhaustive()
    }
}

static SCALAR: Kernels = Kernels {
    backend: Backend::Scalar,
    dot: scalar::dot,
    l2_sq: scalar::l2_sq,
    cosine: scalar::cosine,
    norm_sq: scalar::norm_sq,
    dot_x4: scalar::dot_x4,
};

#[cfg(target_arch = "x86_64")]
static AVX2: Kernels = Kernels {
    backend: Backend::Avx2,
    dot: x86::dot_avx2,
    l2_sq: x86::l2_sq_avx2,
    cosine: x86::cosine_avx2,
    norm_sq: x86::norm_sq_avx2,
    dot_x4: x86::dot_x4_avx2,
};

#[cfg(target_arch = "x86_64")]
static AVX512: Kernels = Kernels {
    backend: Backend::Avx512,
    dot: x86::dot_avx512,
    l2_sq: x86::l2_sq_avx512,
    cosine: x86::cosine_avx512,
    norm_sq: x86::norm_sq_avx512,
    dot_x4: x86::dot_x4_avx512,
};

#[cfg(target_arch = "aarch64")]
static NEON: Kernels = Kernels {
    backend: Backend::Neon,
    dot: neon::dot_neon,
    l2_sq: neon::l2_sq_neon,
    cosine: neon::cosine_neon,
    norm_sq: neon::norm_sq_neon,
    dot_x4: neon::dot_x4_neon,
};

#[cold]
#[inline(never)]
#[track_caller]
fn len_mismatch(a: usize, b: usize) -> ! {
    panic!("vector length mismatch: {a} != {b}")
}

#[inline(always)]
#[track_caller]
fn check_len(a: usize, b: usize) {
    if unlikely(a != b) {
        len_mismatch(a, b);
    }
}

impl Kernels {
    /// Returns the kernel table for `backend` if this CPU supports it.
    ///
    /// # Errors
    /// [`CoreError::UnsupportedBackend`] if the CPU or target architecture lacks the ISA.
    pub fn for_backend(backend: Backend) -> Result<&'static Kernels> {
        if !backend.is_supported() {
            return Err(CoreError::UnsupportedBackend(backend.name()));
        }
        match backend {
            Backend::Scalar => Ok(&SCALAR),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => Ok(&AVX2),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => Ok(&AVX512),
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => Ok(&NEON),
            #[allow(unreachable_patterns)]
            other => Err(CoreError::UnsupportedBackend(other.name())),
        }
    }

    /// The backend these kernels implement.
    #[inline]
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Inner product `Σ aᵢbᵢ`.
    ///
    /// # Panics
    /// Panics if `a.len() != b.len()`.
    #[inline]
    #[track_caller]
    #[must_use]
    pub fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        check_len(a.len(), b.len());
        // SAFETY: the table's backend is supported by this CPU (type invariant).
        unsafe { (self.dot)(a, b) }
    }

    /// Squared Euclidean distance `Σ (aᵢ − bᵢ)²`.
    ///
    /// # Panics
    /// Panics if `a.len() != b.len()`.
    #[inline]
    #[track_caller]
    #[must_use]
    pub fn l2_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        check_len(a.len(), b.len());
        // SAFETY: the table's backend is supported by this CPU (type invariant).
        unsafe { (self.l2_sq)(a, b) }
    }

    /// Cosine similarity `a·b / (‖a‖‖b‖)`, fused into one pass. Returns `0.0` if either norm is 0.
    ///
    /// # Panics
    /// Panics if `a.len() != b.len()`.
    #[inline]
    #[track_caller]
    #[must_use]
    pub fn cosine(&self, a: &[f32], b: &[f32]) -> f32 {
        check_len(a.len(), b.len());
        // SAFETY: the table's backend is supported by this CPU (type invariant).
        unsafe { (self.cosine)(a, b) }
    }

    /// Squared L2 norm `Σ aᵢ²`.
    #[inline]
    #[must_use]
    pub fn norm_sq(&self, a: &[f32]) -> f32 {
        // SAFETY: the table's backend is supported by this CPU (type invariant).
        unsafe { (self.norm_sq)(a) }
    }

    /// Scores one query against four rows at once (register-blocked; ~1.6x fewer loads).
    ///
    /// # Panics
    /// Panics if any row length differs from `q.len()`.
    #[inline]
    #[track_caller]
    #[must_use]
    pub fn dot_x4(&self, q: &[f32], rows: [&[f32]; 4]) -> [f32; 4] {
        for r in rows {
            check_len(q.len(), r.len());
        }
        // SAFETY: the table's backend is supported by this CPU (type invariant).
        unsafe { (self.dot_x4)(q, rows) }
    }

    /// Fallible [`dot`](Self::dot).
    ///
    /// # Errors
    /// [`CoreError::DimensionMismatch`] on length mismatch.
    #[inline]
    pub fn try_dot(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(CoreError::DimensionMismatch { expected: a.len(), actual: b.len() });
        }
        // SAFETY: the table's backend is supported by this CPU (type invariant).
        Ok(unsafe { (self.dot)(a, b) })
    }

    /// Fallible [`cosine`](Self::cosine).
    ///
    /// # Errors
    /// [`CoreError::DimensionMismatch`] on length mismatch.
    #[inline]
    pub fn try_cosine(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(CoreError::DimensionMismatch { expected: a.len(), actual: b.len() });
        }
        // SAFETY: the table's backend is supported by this CPU (type invariant).
        Ok(unsafe { (self.cosine)(a, b) })
    }

    /// Dot product **without** the length check, for inner loops whose shapes were validated once
    /// up front (e.g. a query padded to the index stride against rows of that stride).
    ///
    /// # Safety
    /// This is memory-safe for any lengths (kernels clamp to the shorter slice), but it is marked
    /// `unsafe` because a mismatch silently computes a truncated dot product. Callers must
    /// guarantee `a.len() == b.len()`.
    #[inline(always)]
    #[must_use]
    pub unsafe fn dot_unchecked(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        // SAFETY: backend supported (type invariant); memory safety does not depend on lengths.
        unsafe { (self.dot)(a, b) }
    }

    /// Squared L2 distance without the length check.
    ///
    /// # Safety
    /// Same contract as [`dot_unchecked`](Self::dot_unchecked).
    #[inline(always)]
    #[must_use]
    pub unsafe fn l2_sq_unchecked(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        // SAFETY: backend supported (type invariant); memory safety does not depend on lengths.
        unsafe { (self.l2_sq)(a, b) }
    }

    /// Four-row dot product without length checks.
    ///
    /// # Safety
    /// Same contract as [`dot_unchecked`](Self::dot_unchecked), for every row.
    #[inline(always)]
    #[must_use]
    pub unsafe fn dot_x4_unchecked(&self, q: &[f32], rows: [&[f32]; 4]) -> [f32; 4] {
        // SAFETY: backend supported (type invariant); memory safety does not depend on lengths.
        unsafe { (self.dot_x4)(q, rows) }
    }

    /// Similarity score for `metric` (higher is better). For [`Metric::Cosine`] this computes the
    /// full cosine; indexes store normalized vectors and use [`Metric::Dot`]-style scoring.
    ///
    /// # Panics
    /// Panics if `a.len() != b.len()`.
    #[inline]
    #[track_caller]
    #[must_use]
    pub fn score(&self, metric: Metric, a: &[f32], b: &[f32]) -> f32 {
        match metric {
            Metric::Cosine => self.cosine(a, b),
            Metric::Dot => self.dot(a, b),
            Metric::L2 => -self.l2_sq(a, b),
        }
    }

    /// Normalizes `v` to unit L2 norm in place.
    ///
    /// # Errors
    /// [`CoreError::ZeroNorm`] if the norm is zero, subnormal or non-finite.
    pub fn normalize(&self, v: &mut [f32]) -> Result<()> {
        let n2 = self.norm_sq(v);
        if !(n2.is_finite() && n2 > f32::MIN_POSITIVE) {
            return Err(CoreError::ZeroNorm);
        }
        let inv = 1.0 / n2.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
        Ok(())
    }
}

static ACTIVE: OnceLock<&'static Kernels> = OnceLock::new();

fn select_default() -> &'static Kernels {
    let forced = std::env::var(BACKEND_ENV)
        .ok()
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("auto"))
        .and_then(|s| s.parse::<Backend>().ok())
        .and_then(|b| Kernels::for_backend(b).ok());
    forced.unwrap_or_else(|| {
        Kernels::for_backend(Backend::detect()).expect("detected backend is supported")
    })
}

/// The process-wide kernel table (best supported backend, or `SOVEREIGN_SIMD` if valid).
#[inline]
#[must_use]
pub fn kernels() -> &'static Kernels {
    ACTIVE.get_or_init(select_default)
}

/// Pins the process-wide backend. Must be called before the first [`kernels()`] call to take
/// effect; calling it again with the same backend is a no-op.
///
/// # Errors
/// * [`CoreError::UnsupportedBackend`] if the CPU lacks the ISA.
/// * [`CoreError::BackendAlreadyInitialized`] if a different backend is already active.
pub fn init_backend(backend: Backend) -> Result<&'static Kernels> {
    let wanted = Kernels::for_backend(backend)?;
    let active = *ACTIVE.get_or_init(|| wanted);
    if active.backend != backend {
        return Err(CoreError::BackendAlreadyInitialized {
            active: active.backend.name(),
            requested: backend.name(),
        });
    }
    Ok(active)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xoshiro256pp;

    fn dot_f64(a: &[f32], b: &[f32]) -> (f64, f64) {
        let exact: f64 = a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
        let mag: f64 = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) * f64::from(y)).abs()).sum();
        (exact, mag)
    }

    fn l2_f64(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum()
    }

    fn close(got: f32, want: f64, magnitude: f64) -> bool {
        // Error bound for a blocked f32 summation: a small multiple of eps·Σ|terms|.
        (f64::from(got) - want).abs() <= 1e-5 * magnitude + 1e-6
    }

    fn random_pair(rng: &mut Xoshiro256pp, n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        rng.fill_uniform(&mut a, -1.0, 1.0);
        rng.fill_uniform(&mut b, -1.0, 1.0);
        (a, b)
    }

    #[test]
    fn all_backends_match_f64_reference_for_every_tail_length() {
        let mut rng = Xoshiro256pp::seed_from_u64(0xC0FFEE);
        for backend in Backend::supported() {
            let k = Kernels::for_backend(backend).unwrap();
            for n in (0..=200).chain([255, 256, 257, 383, 384, 768, 1024, 1536, 3072]) {
                let (a, b) = random_pair(&mut rng, n);
                let (want, mag) = dot_f64(&a, &b);
                assert!(close(k.dot(&a, &b), want, mag), "{backend} dot n={n}");

                let l2 = l2_f64(&a, &b);
                assert!(close(k.l2_sq(&a, &b), l2, l2), "{backend} l2 n={n}");

                let (na, ma) = dot_f64(&a, &a);
                assert!(close(k.norm_sq(&a), na, ma), "{backend} norm n={n}");

                let (nb, _) = dot_f64(&b, &b);
                let cos_want = if na * nb > 0.0 { want / (na * nb).sqrt() } else { 0.0 };
                let cos = k.cosine(&a, &b);
                assert!(
                    (f64::from(cos) - cos_want).abs() < 1e-4,
                    "{backend} cos n={n}: {cos} vs {cos_want}"
                );
            }
        }
    }

    #[test]
    fn dot_x4_matches_single_row_kernel() {
        let mut rng = Xoshiro256pp::seed_from_u64(99);
        for backend in Backend::supported() {
            let k = Kernels::for_backend(backend).unwrap();
            for n in [0usize, 1, 7, 8, 15, 16, 17, 31, 33, 100, 384, 1536, 1537] {
                let mut data = vec![0.0f32; 5 * n];
                rng.fill_uniform(&mut data, -1.0, 1.0);
                let (q, rows) = data.split_at(n);
                let r: Vec<&[f32]> = rows.chunks_exact(n.max(1)).take(4).collect();
                let rows4 = if n == 0 { [&[][..]; 4] } else { [r[0], r[1], r[2], r[3]] };
                let got = k.dot_x4(q, rows4);
                for (g, row) in got.iter().zip(rows4) {
                    let (want, mag) = dot_f64(q, row);
                    assert!(close(*g, want, mag), "{backend} x4 n={n}");
                }
            }
        }
    }

    #[test]
    fn cosine_edge_cases() {
        for backend in Backend::supported() {
            let k = Kernels::for_backend(backend).unwrap();
            let z = [0.0f32; 37];
            let o = [1.0f32; 37];
            assert_eq!(k.cosine(&z, &o), 0.0);
            assert!((k.cosine(&o, &o) - 1.0).abs() < 1e-6);
            let neg: Vec<f32> = o.iter().map(|x| -x).collect();
            assert!((k.cosine(&o, &neg) + 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn normalize_and_errors() {
        let k = kernels();
        let mut v = vec![3.0f32, 4.0];
        k.normalize(&mut v).unwrap();
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        assert_eq!(k.normalize(&mut [0.0; 4]), Err(CoreError::ZeroNorm));
        assert_eq!(k.normalize(&mut [f32::NAN; 4]), Err(CoreError::ZeroNorm));
        assert!(k.try_dot(&[1.0], &[1.0, 2.0]).is_err());
        assert_eq!(k.score(Metric::L2, &[1.0, 1.0], &[0.0, 0.0]), -2.0);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn mismatched_lengths_panic() {
        let _ = kernels().dot(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn backend_parsing_and_support() {
        assert_eq!("AVX512".parse::<Backend>().unwrap(), Backend::Avx512);
        assert!("sse9".parse::<Backend>().is_err());
        assert!(Backend::Scalar.is_supported());
        assert_eq!(Backend::detect(), Backend::supported()[0]);
        assert!(Kernels::for_backend(Backend::Scalar).is_ok());
        #[cfg(target_arch = "x86_64")]
        assert!(Kernels::for_backend(Backend::Neon).is_err());
        let active = kernels().backend();
        assert!(init_backend(active).is_ok());
    }
}
