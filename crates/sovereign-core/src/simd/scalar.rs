//! Portable kernels.
//!
//! Two flavours live here:
//!
//! * `*_naive` — the textbook one-accumulator loop. A strict left-to-right floating-point sum is
//!   a serial dependency chain (`s = s + x·y` must wait for the previous add), so LLVM is *not
//!   allowed* to vectorize it without `-ffast-math`-style reassociation. It runs at one
//!   element per FP-add latency (~4 cycles). This is the benchmark baseline.
//! * The unrolled kernels (the [`Backend::Scalar`](super::Backend::Scalar) backend) keep eight
//!   independent accumulators. Breaking the chain lets the out-of-order core overlap the adds,
//!   and LLVM will usually pack the eight lanes into SSE2/NEON registers on its own.
//!
//! Note: none of these use `f32::mul_add`. Without the `fma` target feature enabled at compile
//! time, `mul_add` lowers to a call into libm's `fmaf` — correct, but ~10–20x slower than a
//! separate multiply and add.

/// Textbook dot product. Baseline for benchmarks and a reference for tests.
#[inline]
#[must_use]
pub fn dot_naive(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Textbook squared Euclidean distance.
#[inline]
#[must_use]
pub fn l2_sq_naive(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Textbook three-pass cosine similarity (`0.0` if either vector has zero norm).
#[inline]
#[must_use]
pub fn cosine_naive(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_naive(a, b);
    let na = dot_naive(a, a);
    let nb = dot_naive(b, b);
    finish_cosine(dot, na, nb)
}

/// Shared cosine epilogue: `dot / sqrt(na · nb)` with a zero-norm guard.
///
/// A single `sqrt` of the product (instead of two) saves a long-latency op; the product of two
/// `f32` squared norms cannot overflow for realistic embeddings (would need components ~1e19).
#[inline(always)]
#[must_use]
pub fn finish_cosine(dot: f32, na: f32, nb: f32) -> f32 {
    let denom = na * nb;
    if denom <= f32::MIN_POSITIVE {
        0.0
    } else {
        dot / denom.sqrt()
    }
}

const U: usize = 8;

/// 8-accumulator dot product.
#[inline]
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f32; U];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<U>(), b.as_chunks::<U>());
    for (x, y) in ca.iter().zip(cb) {
        for j in 0..U {
            acc[j] += x[j] * y[j];
        }
    }
    let mut s = reduce8(acc);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// 8-accumulator squared Euclidean distance.
#[inline]
#[must_use]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f32; U];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<U>(), b.as_chunks::<U>());
    for (x, y) in ca.iter().zip(cb) {
        for j in 0..U {
            let d = x[j] - y[j];
            acc[j] += d * d;
        }
    }
    let mut s = reduce8(acc);
    for (x, y) in ra.iter().zip(rb) {
        let d = x - y;
        s += d * d;
    }
    s
}

/// Fused single-pass cosine similarity: `dot`, `‖a‖²` and `‖b‖²` accumulated together so each
/// element is loaded once.
#[inline]
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (mut d, mut na, mut nb) = ([0.0f32; U], [0.0f32; U], [0.0f32; U]);
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<U>(), b.as_chunks::<U>());
    for (x, y) in ca.iter().zip(cb) {
        for j in 0..U {
            d[j] += x[j] * y[j];
            na[j] += x[j] * x[j];
            nb[j] += y[j] * y[j];
        }
    }
    let (mut sd, mut sa, mut sb) = (reduce8(d), reduce8(na), reduce8(nb));
    for (x, y) in ra.iter().zip(rb) {
        sd += x * y;
        sa += x * x;
        sb += y * y;
    }
    finish_cosine(sd, sa, sb)
}

/// Squared L2 norm.
#[inline]
#[must_use]
pub fn norm_sq(a: &[f32]) -> f32 {
    dot(a, a)
}

/// One query against four rows.
#[inline]
#[must_use]
pub fn dot_x4(q: &[f32], rows: [&[f32]; 4]) -> [f32; 4] {
    [dot(q, rows[0]), dot(q, rows[1]), dot(q, rows[2]), dot(q, rows[3])]
}

/// Pairwise tree reduction — better rounding than a left fold and shorter dependency chain.
#[inline(always)]
fn reduce8(a: [f32; U]) -> f32 {
    ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))
}
