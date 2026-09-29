//! AArch64 NEON kernels (Graviton, Ampere, Apple M-series, Neoverse).
//!
//! NEON (Advanced SIMD) is mandatory in the AArch64 base ISA, so no runtime detection is needed.
//! Registers are 128-bit (4 × f32). Neoverse V1/V2 and Apple Firestorm have four 128-bit FMA
//! pipes with 4-cycle latency, so we keep four accumulators live (16 floats per iteration) to
//! cover latency × throughput.
//!
//! Safety contract mirrors the x86 module: kernels clamp to the shorter length, and the only
//! precondition is that NEON is available (always true on AArch64).

use core::arch::aarch64::*;

use super::scalar::finish_cosine;

/// Dot product, NEON.
///
/// # Safety
/// The CPU must support `neon` (guaranteed on AArch64).
#[target_feature(enable = "neon")]
pub(super) unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let (mut acc0, mut acc1, mut acc2, mut acc3) =
        (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n`, so every load reads inside `[0, n)` of both slices.
        unsafe {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
        }
        i += 16;
    }
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n`.
        unsafe { acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i))) };
        i += 4;
    }
    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    while i < n {
        // SAFETY: `i < n`.
        sum += unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) };
        i += 1;
    }
    sum
}

/// Squared Euclidean distance, NEON.
///
/// # Safety
/// The CPU must support `neon`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn l2_sq_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let (mut acc0, mut acc1, mut acc2, mut acc3) =
        (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n`.
        unsafe {
            let d0 = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            let d1 = vsubq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            let d2 = vsubq_f32(vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            let d3 = vsubq_f32(vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            acc0 = vfmaq_f32(acc0, d0, d0);
            acc1 = vfmaq_f32(acc1, d1, d1);
            acc2 = vfmaq_f32(acc2, d2, d2);
            acc3 = vfmaq_f32(acc3, d3, d3);
        }
        i += 16;
    }
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n`.
        unsafe {
            let d = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc0 = vfmaq_f32(acc0, d, d);
        }
        i += 4;
    }
    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    while i < n {
        // SAFETY: `i < n`.
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        sum += d * d;
        i += 1;
    }
    sum
}

/// Fused single-pass cosine similarity, NEON.
///
/// # Safety
/// The CPU must support `neon`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let z = vdupq_n_f32(0.0);
    let (mut d0, mut d1, mut na0, mut na1, mut nb0, mut nb1) = (z, z, z, z, z, z);
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let a0 = vld1q_f32(pa.add(i));
            let a1 = vld1q_f32(pa.add(i + 4));
            let b0 = vld1q_f32(pb.add(i));
            let b1 = vld1q_f32(pb.add(i + 4));
            d0 = vfmaq_f32(d0, a0, b0);
            d1 = vfmaq_f32(d1, a1, b1);
            na0 = vfmaq_f32(na0, a0, a0);
            na1 = vfmaq_f32(na1, a1, a1);
            nb0 = vfmaq_f32(nb0, b0, b0);
            nb1 = vfmaq_f32(nb1, b1, b1);
        }
        i += 8;
    }
    let mut dot = vaddvq_f32(vaddq_f32(d0, d1));
    let mut sa = vaddvq_f32(vaddq_f32(na0, na1));
    let mut sb = vaddvq_f32(vaddq_f32(nb0, nb1));
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

/// Squared L2 norm, NEON.
///
/// # Safety
/// The CPU must support `neon`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn norm_sq_neon(a: &[f32]) -> f32 {
    // SAFETY: same target features as this function.
    unsafe { dot_neon(a, a) }
}

/// One query against four rows, NEON, 2x unrolled.
///
/// # Safety
/// The CPU must support `neon`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn dot_x4_neon(q: &[f32], rows: [&[f32]; 4]) -> [f32; 4] {
    let n = q.len().min(rows[0].len()).min(rows[1].len()).min(rows[2].len()).min(rows[3].len());
    let pq = q.as_ptr();
    let [p0, p1, p2, p3] = rows.map(<[f32]>::as_ptr);
    let z = vdupq_n_f32(0.0);
    let (mut s0, mut s1, mut s2, mut s3) = (z, z, z, z);
    let (mut t0, mut t1, mut t2, mut t3) = (z, z, z, z);
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n <= len` of the query and every row.
        unsafe {
            let qa = vld1q_f32(pq.add(i));
            let qb = vld1q_f32(pq.add(i + 4));
            s0 = vfmaq_f32(s0, qa, vld1q_f32(p0.add(i)));
            s1 = vfmaq_f32(s1, qa, vld1q_f32(p1.add(i)));
            s2 = vfmaq_f32(s2, qa, vld1q_f32(p2.add(i)));
            s3 = vfmaq_f32(s3, qa, vld1q_f32(p3.add(i)));
            t0 = vfmaq_f32(t0, qb, vld1q_f32(p0.add(i + 4)));
            t1 = vfmaq_f32(t1, qb, vld1q_f32(p1.add(i + 4)));
            t2 = vfmaq_f32(t2, qb, vld1q_f32(p2.add(i + 4)));
            t3 = vfmaq_f32(t3, qb, vld1q_f32(p3.add(i + 4)));
        }
        i += 8;
    }
    let mut out = [
        vaddvq_f32(vaddq_f32(s0, t0)),
        vaddvq_f32(vaddq_f32(s1, t1)),
        vaddvq_f32(vaddq_f32(s2, t2)),
        vaddvq_f32(vaddq_f32(s3, t3)),
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
