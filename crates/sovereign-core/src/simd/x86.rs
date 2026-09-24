//! x86_64 kernels: AVX2 + FMA (Haswell+/Zen1+) and AVX-512F (Skylake-SP+/Zen4+).
//!
//! # Design notes
//!
//! * **Multiple accumulators.** `vfmadd231ps` has 4-cycle latency and two issue ports on modern
//!   cores, so a single accumulator chain would idle the FMA units ~87% of the time. A dot product
//!   issues 2 loads per FMA and cores sustain 2 loads/cycle, so the loop is *load-bound* at
//!   1 FMA/cycle; four independent accumulators are enough to cover the latency. The x4 kernels
//!   share one query load across four rows (5 loads per 4 FMAs) to lift that ceiling when the
//!   data is cache-resident.
//! * **No tails on the hot path.** Rows produced by this crate are padded to a multiple of 16
//!   lanes, so the tail code below only runs for ad-hoc slices. AVX-512 handles tails with a
//!   masked load (`vmovups zmm{k}{z}`): masked-off lanes are neither read nor faulted on, so we can
//!   safely "over-read" past the end of a slice without touching the memory.
//! * **Safety contract.** Every `unsafe fn` here requires only that the CPU supports the enabled
//!   target features. Lengths are clamped to `min(a.len(), b.len())` inside each kernel, so a
//!   length mismatch is a logic error (caught by the safe wrappers), never a memory-safety bug.
//! * **Frequency licensing.** On Skylake-SP/Cascade Lake, sustained 512-bit FMA traffic can drop
//!   the core clock by one "license" level. Ice Lake, Sapphire Rapids and Zen 4 do not suffer from
//!   this in practice. Force AVX2 with `SOVEREIGN_SIMD=avx2` if profiling shows a regression.

use core::arch::x86_64::*;

use super::scalar::finish_cosine;

// ---------------------------------------------------------------------------------------------
// AVX2 + FMA
// ---------------------------------------------------------------------------------------------

/// Horizontal sum of the 8 lanes of `v`.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn hsum256(v: __m256) -> f32 {
    // [a b c d | e f g h] -> [a+e b+f c+g d+h]
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    // movehdup: [1 1 3 3]; add -> [0+1 . 2+3 .]
    let shuf = _mm_movehdup_ps(s);
    let sums = _mm_add_ps(s, shuf);
    // movehl: bring lane 2 down; add_ss -> lane 0 = total
    let shuf = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf))
}

/// Dot product, AVX2 + FMA.
///
/// # Safety
/// The CPU must support `avx2` and `fma`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let (mut acc0, mut acc1, mut acc2, mut acc3) =
        (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n`, so all loads read inside `[0, n)` of both slices.
        let (a0, a1, a2, a3, b0, b1, b2, b3) = unsafe {
            (
                _mm256_loadu_ps(pa.add(i)),
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i)),
                _mm256_loadu_ps(pb.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 24)),
            )
        };
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        acc2 = _mm256_fmadd_ps(a2, b2, acc2);
        acc3 = _mm256_fmadd_ps(a3, b3, acc3);
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        let (x, y) = unsafe { (_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))) };
        acc0 = _mm256_fmadd_ps(x, y, acc0);
        i += 8;
    }
    let mut sum = hsum256(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    while i < n {
        // SAFETY: `i < n`.
        sum += unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) };
        i += 1;
    }
    sum
}

/// Squared Euclidean distance, AVX2 + FMA.
///
/// # Safety
/// The CPU must support `avx2` and `fma`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let (mut acc0, mut acc1, mut acc2, mut acc3) =
        (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n`.
        let (d0, d1, d2, d3) = unsafe {
            (
                _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))),
                _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8))),
                _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16))),
                _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24))),
            )
        };
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
        i += 32;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        let d = unsafe { _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))) };
        acc0 = _mm256_fmadd_ps(d, d, acc0);
        i += 8;
    }
    let mut sum = hsum256(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    while i < n {
        // SAFETY: `i < n`.
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        sum += d * d;
        i += 1;
    }
    sum
}

/// Fused single-pass cosine similarity, AVX2 + FMA.
///
/// Three FMAs per pair of loads: this kernel is FMA-port-bound rather than load-bound, so a 2x
/// unroll (six accumulators) is enough to saturate both FMA ports.
///
/// # Safety
/// The CPU must support `avx2` and `fma`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let z = _mm256_setzero_ps();
    let (mut d0, mut d1, mut na0, mut na1, mut nb0, mut nb1) = (z, z, z, z, z, z);
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n`.
        let (a0, a1, b0, b1) = unsafe {
            (
                _mm256_loadu_ps(pa.add(i)),
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i)),
                _mm256_loadu_ps(pb.add(i + 8)),
            )
        };
        d0 = _mm256_fmadd_ps(a0, b0, d0);
        d1 = _mm256_fmadd_ps(a1, b1, d1);
        na0 = _mm256_fmadd_ps(a0, a0, na0);
        na1 = _mm256_fmadd_ps(a1, a1, na1);
        nb0 = _mm256_fmadd_ps(b0, b0, nb0);
        nb1 = _mm256_fmadd_ps(b1, b1, nb1);
        i += 16;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        let (x, y) = unsafe { (_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))) };
        d0 = _mm256_fmadd_ps(x, y, d0);
        na0 = _mm256_fmadd_ps(x, x, na0);
        nb0 = _mm256_fmadd_ps(y, y, nb0);
        i += 8;
    }
    let mut dot = hsum256(_mm256_add_ps(d0, d1));
    let mut sa = hsum256(_mm256_add_ps(na0, na1));
    let mut sb = hsum256(_mm256_add_ps(nb0, nb1));
    while i < n {
        // SAFETY: `i < n`.
        let (x, y) = unsafe { (*a.get_unchecked(i), *b.get_unchecked(i)) };
        dot += x * y;
        sa += x * x;
        sb += y * y;
        i += 1;
    }
    finish_cosine(dot, sa, sb)
}

/// Squared L2 norm, AVX2 + FMA.
///
/// # Safety
/// The CPU must support `avx2` and `fma`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn norm_sq_avx2(a: &[f32]) -> f32 {
    // SAFETY: same target features as this function.
    unsafe { dot_avx2(a, a) }
}

/// One query against four rows, sharing each query load across four FMAs (register blocking).
///
/// # Safety
/// The CPU must support `avx2` and `fma`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_x4_avx2(q: &[f32], rows: [&[f32]; 4]) -> [f32; 4] {
    let n = q.len().min(rows[0].len()).min(rows[1].len()).min(rows[2].len()).min(rows[3].len());
    let pq = q.as_ptr();
    let [p0, p1, p2, p3] = rows.map(<[f32]>::as_ptr);
    let z = _mm256_setzero_ps();
    let (mut s0, mut s1, mut s2, mut s3) = (z, z, z, z);
    let (mut t0, mut t1, mut t2, mut t3) = (z, z, z, z);
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n <= len` of the query and of every row.
        unsafe {
            let qa = _mm256_loadu_ps(pq.add(i));
            let qb = _mm256_loadu_ps(pq.add(i + 8));
            s0 = _mm256_fmadd_ps(qa, _mm256_loadu_ps(p0.add(i)), s0);
            s1 = _mm256_fmadd_ps(qa, _mm256_loadu_ps(p1.add(i)), s1);
            s2 = _mm256_fmadd_ps(qa, _mm256_loadu_ps(p2.add(i)), s2);
            s3 = _mm256_fmadd_ps(qa, _mm256_loadu_ps(p3.add(i)), s3);
            t0 = _mm256_fmadd_ps(qb, _mm256_loadu_ps(p0.add(i + 8)), t0);
            t1 = _mm256_fmadd_ps(qb, _mm256_loadu_ps(p1.add(i + 8)), t1);
            t2 = _mm256_fmadd_ps(qb, _mm256_loadu_ps(p2.add(i + 8)), t2);
            t3 = _mm256_fmadd_ps(qb, _mm256_loadu_ps(p3.add(i + 8)), t3);
        }
        i += 16;
    }
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let qa = _mm256_loadu_ps(pq.add(i));
            s0 = _mm256_fmadd_ps(qa, _mm256_loadu_ps(p0.add(i)), s0);
            s1 = _mm256_fmadd_ps(qa, _mm256_loadu_ps(p1.add(i)), s1);
            s2 = _mm256_fmadd_ps(qa, _mm256_loadu_ps(p2.add(i)), s2);
            s3 = _mm256_fmadd_ps(qa, _mm256_loadu_ps(p3.add(i)), s3);
        }
        i += 8;
    }
    let mut out = [
        hsum256(_mm256_add_ps(s0, t0)),
        hsum256(_mm256_add_ps(s1, t1)),
        hsum256(_mm256_add_ps(s2, t2)),
        hsum256(_mm256_add_ps(s3, t3)),
    ];
    while i < n {
        // SAFETY: `i < n`.
        unsafe {
            let x = *q.get_unchecked(i);
            for (o, r) in out.iter_mut().zip(rows) {
                *o += x * *r.get_unchecked(i);
            }
        }
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------------------------
// AVX-512F
// ---------------------------------------------------------------------------------------------

/// Mask selecting the low `rem` lanes (`rem < 16`).
#[inline(always)]
fn tail_mask(rem: usize) -> __mmask16 {
    debug_assert!(rem < 16);
    ((1u32 << rem) - 1) as __mmask16
}

/// Dot product, AVX-512F.
///
/// # Safety
/// The CPU must support `avx512f`.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let (mut acc0, mut acc1, mut acc2, mut acc3) =
        (_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps());
    let mut i = 0;
    while i + 64 <= n {
        // SAFETY: `i + 64 <= n`, so all loads read inside `[0, n)` of both slices.
        let (a0, a1, a2, a3, b0, b1, b2, b3) = unsafe {
            (
                _mm512_loadu_ps(pa.add(i)),
                _mm512_loadu_ps(pa.add(i + 16)),
                _mm512_loadu_ps(pa.add(i + 32)),
                _mm512_loadu_ps(pa.add(i + 48)),
                _mm512_loadu_ps(pb.add(i)),
                _mm512_loadu_ps(pb.add(i + 16)),
                _mm512_loadu_ps(pb.add(i + 32)),
                _mm512_loadu_ps(pb.add(i + 48)),
            )
        };
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        acc2 = _mm512_fmadd_ps(a2, b2, acc2);
        acc3 = _mm512_fmadd_ps(a3, b3, acc3);
        i += 64;
    }
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n`.
        let (x, y) = unsafe { (_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i))) };
        acc0 = _mm512_fmadd_ps(x, y, acc0);
        i += 16;
    }
    if i < n {
        let m = tail_mask(n - i);
        // SAFETY: `pa.add(i)`/`pb.add(i)` are in bounds (`i < n`); lanes at or beyond `n` are
        // masked off, and AVX-512 masked loads neither access nor fault on masked lanes.
        let (x, y) =
            unsafe { (_mm512_maskz_loadu_ps(m, pa.add(i)), _mm512_maskz_loadu_ps(m, pb.add(i))) };
        acc1 = _mm512_fmadd_ps(x, y, acc1);
    }
    _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)))
}

/// Squared Euclidean distance, AVX-512F.
///
/// # Safety
/// The CPU must support `avx512f`.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn l2_sq_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let (mut acc0, mut acc1, mut acc2, mut acc3) =
        (_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps());
    let mut i = 0;
    while i + 64 <= n {
        // SAFETY: `i + 64 <= n`.
        let (d0, d1, d2, d3) = unsafe {
            (
                _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i))),
                _mm512_sub_ps(_mm512_loadu_ps(pa.add(i + 16)), _mm512_loadu_ps(pb.add(i + 16))),
                _mm512_sub_ps(_mm512_loadu_ps(pa.add(i + 32)), _mm512_loadu_ps(pb.add(i + 32))),
                _mm512_sub_ps(_mm512_loadu_ps(pa.add(i + 48)), _mm512_loadu_ps(pb.add(i + 48))),
            )
        };
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2);
        acc3 = _mm512_fmadd_ps(d3, d3, acc3);
        i += 64;
    }
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n`.
        let d = unsafe { _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i))) };
        acc0 = _mm512_fmadd_ps(d, d, acc0);
        i += 16;
    }
    if i < n {
        let m = tail_mask(n - i);
        // SAFETY: in-bounds base pointers; masked lanes are not accessed (see `dot_avx512`).
        let d = unsafe {
            _mm512_sub_ps(_mm512_maskz_loadu_ps(m, pa.add(i)), _mm512_maskz_loadu_ps(m, pb.add(i)))
        };
        acc1 = _mm512_fmadd_ps(d, d, acc1);
    }
    _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)))
}

/// Fused single-pass cosine similarity, AVX-512F.
///
/// # Safety
/// The CPU must support `avx512f`.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn cosine_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let z = _mm512_setzero_ps();
    let (mut d0, mut d1, mut na0, mut na1, mut nb0, mut nb1) = (z, z, z, z, z, z);
    let mut i = 0;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n`.
        let (a0, a1, b0, b1) = unsafe {
            (
                _mm512_loadu_ps(pa.add(i)),
                _mm512_loadu_ps(pa.add(i + 16)),
                _mm512_loadu_ps(pb.add(i)),
                _mm512_loadu_ps(pb.add(i + 16)),
            )
        };
        d0 = _mm512_fmadd_ps(a0, b0, d0);
        d1 = _mm512_fmadd_ps(a1, b1, d1);
        na0 = _mm512_fmadd_ps(a0, a0, na0);
        na1 = _mm512_fmadd_ps(a1, a1, na1);
        nb0 = _mm512_fmadd_ps(b0, b0, nb0);
        nb1 = _mm512_fmadd_ps(b1, b1, nb1);
        i += 32;
    }
    while i < n {
        let rem = n - i;
        let (x, y) = if rem >= 16 {
            // SAFETY: `i + 16 <= n`.
            unsafe { (_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i))) }
        } else {
            let m = tail_mask(rem);
            // SAFETY: in-bounds base pointers; masked lanes are not accessed.
            unsafe { (_mm512_maskz_loadu_ps(m, pa.add(i)), _mm512_maskz_loadu_ps(m, pb.add(i))) }
        };
        d0 = _mm512_fmadd_ps(x, y, d0);
        na0 = _mm512_fmadd_ps(x, x, na0);
        nb0 = _mm512_fmadd_ps(y, y, nb0);
        i += 16;
    }
    finish_cosine(
        _mm512_reduce_add_ps(_mm512_add_ps(d0, d1)),
        _mm512_reduce_add_ps(_mm512_add_ps(na0, na1)),
        _mm512_reduce_add_ps(_mm512_add_ps(nb0, nb1)),
    )
}

/// Squared L2 norm, AVX-512F.
///
/// # Safety
/// The CPU must support `avx512f`.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn norm_sq_avx512(a: &[f32]) -> f32 {
    // SAFETY: same target features as this function.
    unsafe { dot_avx512(a, a) }
}

/// One query against four rows, AVX-512F, 2x unrolled (eight accumulators).
///
/// # Safety
/// The CPU must support `avx512f`.
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn dot_x4_avx512(q: &[f32], rows: [&[f32]; 4]) -> [f32; 4] {
    let n = q.len().min(rows[0].len()).min(rows[1].len()).min(rows[2].len()).min(rows[3].len());
    let pq = q.as_ptr();
    let [p0, p1, p2, p3] = rows.map(<[f32]>::as_ptr);
    let z = _mm512_setzero_ps();
    let (mut s0, mut s1, mut s2, mut s3) = (z, z, z, z);
    let (mut t0, mut t1, mut t2, mut t3) = (z, z, z, z);
    let mut i = 0;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n <= len` of the query and every row.
        unsafe {
            let qa = _mm512_loadu_ps(pq.add(i));
            let qb = _mm512_loadu_ps(pq.add(i + 16));
            s0 = _mm512_fmadd_ps(qa, _mm512_loadu_ps(p0.add(i)), s0);
            s1 = _mm512_fmadd_ps(qa, _mm512_loadu_ps(p1.add(i)), s1);
            s2 = _mm512_fmadd_ps(qa, _mm512_loadu_ps(p2.add(i)), s2);
            s3 = _mm512_fmadd_ps(qa, _mm512_loadu_ps(p3.add(i)), s3);
            t0 = _mm512_fmadd_ps(qb, _mm512_loadu_ps(p0.add(i + 16)), t0);
            t1 = _mm512_fmadd_ps(qb, _mm512_loadu_ps(p1.add(i + 16)), t1);
            t2 = _mm512_fmadd_ps(qb, _mm512_loadu_ps(p2.add(i + 16)), t2);
            t3 = _mm512_fmadd_ps(qb, _mm512_loadu_ps(p3.add(i + 16)), t3);
        }
        i += 32;
    }
    while i < n {
        let rem = n - i;
        let m = if rem >= 16 { 0xFFFF } else { tail_mask(rem) };
        // SAFETY: base pointers are in bounds (`i < n`); lanes beyond `n` are masked off and
        // therefore never accessed.
        unsafe {
            let qa = _mm512_maskz_loadu_ps(m, pq.add(i));
            s0 = _mm512_fmadd_ps(qa, _mm512_maskz_loadu_ps(m, p0.add(i)), s0);
            s1 = _mm512_fmadd_ps(qa, _mm512_maskz_loadu_ps(m, p1.add(i)), s1);
            s2 = _mm512_fmadd_ps(qa, _mm512_maskz_loadu_ps(m, p2.add(i)), s2);
            s3 = _mm512_fmadd_ps(qa, _mm512_maskz_loadu_ps(m, p3.add(i)), s3);
        }
        i += 16;
    }
    [
        _mm512_reduce_add_ps(_mm512_add_ps(s0, t0)),
        _mm512_reduce_add_ps(_mm512_add_ps(s1, t1)),
        _mm512_reduce_add_ps(_mm512_add_ps(s2, t2)),
        _mm512_reduce_add_ps(_mm512_add_ps(s3, t3)),
    ]
}
