//! Exact (brute-force) search over a matrix of padded rows.
//!
//! The scan is a straight line through memory, which is the access pattern DRAM and the hardware
//! stream prefetchers are best at; for dot/cosine it scores four rows per call with the
//! register-blocked `dot_x4` kernel (one query load feeds four FMAs). Large segments are split
//! into fixed-size blocks scanned by rayon workers, each with a private [`TopK`], merged at the
//! end. Because [`TopK`] breaks ties by id, the parallel result is identical to a sequential scan.

use rayon::prelude::*;
use sovereign_core::{Kernels, MatrixRef, TopK};

/// Rows per parallel work item: 2048 rows × 6 KiB (1536-d) = 12 MiB per task — big enough to
/// amortize scheduling, small enough to balance across cores.
const BLOCK_ROWS: usize = 2048;
/// Below this many `f32`s (~16 MiB) the scan stays on the calling thread.
const PARALLEL_THRESHOLD: usize = 4 << 20;

/// Everything a scan needs besides the row range and the output collector.
#[derive(Clone, Copy)]
struct Scan<'a> {
    k: &'a Kernels,
    m: MatrixRef<'a>,
    q: &'a [f32],
    l2: bool,
    key_base: u64,
}

/// Scans rows `[start, end)` into `out`, keyed by `key_base | row`.
///
/// # Safety
/// `ctx.q.len() == ctx.m.stride()` and `start <= end <= ctx.m.rows()`.
unsafe fn scan_range(ctx: Scan<'_>, start: usize, end: usize, out: &mut TopK) {
    let Scan { k, m, q, l2, key_base } = ctx;
    let mut r = start;
    if l2 {
        while r < end {
            // SAFETY: `r < end <= rows`; lengths equal per the caller contract.
            let d = unsafe { k.l2_sq_unchecked(q, m.row_padded_unchecked(r)) };
            out.push(key_base | r as u64, -d);
            r += 1;
        }
        return;
    }
    while r + 4 <= end {
        // SAFETY: `r + 3 < end <= rows`; all rows have length `stride == q.len()`.
        let s = unsafe {
            k.dot_x4_unchecked(
                q,
                [
                    m.row_padded_unchecked(r),
                    m.row_padded_unchecked(r + 1),
                    m.row_padded_unchecked(r + 2),
                    m.row_padded_unchecked(r + 3),
                ],
            )
        };
        for (j, &score) in s.iter().enumerate() {
            out.push(key_base | (r + j) as u64, score);
        }
        r += 4;
    }
    while r < end {
        // SAFETY: `r < end <= rows`.
        let s = unsafe { k.dot_unchecked(q, m.row_padded_unchecked(r)) };
        out.push(key_base | r as u64, s);
        r += 1;
    }
}

/// Exact top-k over all rows of `m`, pushing into `out`. Parallelizes large matrices.
///
/// `q` must be prepared (padded to `m.stride()`, normalized for cosine).
pub(crate) fn scan(
    k: &Kernels,
    m: MatrixRef<'_>,
    q: &[f32],
    l2: bool,
    out: &mut TopK,
    key_base: u64,
) {
    assert_eq!(q.len(), m.stride(), "query must be padded to the matrix stride");
    let rows = m.rows();
    let ctx = Scan { k, m, q, l2, key_base };
    if rows.saturating_mul(m.stride()) < PARALLEL_THRESHOLD {
        // SAFETY: `q.len() == stride` asserted above; range is the whole matrix.
        unsafe { scan_range(ctx, 0, rows, out) };
        return;
    }
    let kk = out.k();
    let blocks = rows.div_ceil(BLOCK_ROWS);
    let merged = (0..blocks)
        .into_par_iter()
        .fold(
            || TopK::new(kk),
            |mut t, b| {
                let start = b * BLOCK_ROWS;
                let end = (start + BLOCK_ROWS).min(rows);
                // SAFETY: `q.len() == stride`; `start < end <= rows`.
                unsafe { scan_range(ctx, start, end, &mut t) };
                t
            },
        )
        .reduce(
            || TopK::new(kk),
            |mut a, b| {
                a.merge(&b);
                a
            },
        );
    out.merge(&merged);
}

#[cfg(test)]
mod tests {
    use super::*;
    use sovereign_core::{padded_stride, push_padded_row, AlignedVec, Xoshiro256pp};

    /// Small enough for Miri: covers every unsafe block in `scan_range` (1x4 blocked rows, the
    /// remainder rows and the L2 path) and checks results against a plain brute force.
    #[test]
    fn small_scan_matches_brute_force() {
        let (rows, dim) = (37, 20); // 37 = 9 blocks of 4 + 1 remainder row
        let stride = padded_stride(dim).unwrap();
        let mut rng = Xoshiro256pp::seed_from_u64(5);
        let mut buf = AlignedVec::new();
        let mut v = vec![0.0; dim];
        for _ in 0..rows {
            rng.fill_gaussian(&mut v);
            push_padded_row(&mut buf, &v, stride).unwrap();
        }
        let m = MatrixRef::new(&buf, rows, dim, stride).unwrap();
        let mut q = vec![0.0; stride];
        rng.fill_gaussian(&mut q[..dim]);
        let k = sovereign_core::kernels();
        for l2 in [false, true] {
            let mut got = TopK::new(rows);
            scan(k, m, &q, l2, &mut got, 7 << 32);
            let got = got.into_sorted_vec();
            let mut want: Vec<(f32, u64)> = (0..rows)
                .map(|r| {
                    let s =
                        if l2 { -k.l2_sq(&q, m.row_padded(r)) } else { k.dot(&q, m.row_padded(r)) };
                    (s, (7 << 32) | r as u64)
                })
                .collect();
            want.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            assert_eq!(got.len(), rows);
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.id, w.1, "l2={l2}");
                assert!((g.score - w.0).abs() <= 1e-5 * w.0.abs().max(1.0), "l2={l2}");
            }
        }
    }

    #[test]
    fn parallel_scan_matches_sequential_exactly() {
        let dim = 64;
        let stride = padded_stride(dim).unwrap();
        let rows = PARALLEL_THRESHOLD / stride + 1234;
        let mut rng = Xoshiro256pp::seed_from_u64(11);
        let mut buf = AlignedVec::with_capacity(rows * stride);
        let mut v = vec![0.0; dim];
        for _ in 0..rows {
            rng.fill_gaussian(&mut v);
            push_padded_row(&mut buf, &v, stride).unwrap();
        }
        let m = MatrixRef::new(&buf, rows, dim, stride).unwrap();
        let mut q = vec![0.0; stride];
        rng.fill_gaussian(&mut q[..dim]);
        let k = sovereign_core::kernels();
        for l2 in [false, true] {
            let mut par = TopK::new(20);
            scan(k, m, &q, l2, &mut par, 0);
            let mut seq = TopK::new(20);
            let ctx = Scan { k, m, q: &q, l2, key_base: 0 };
            // SAFETY: q padded to stride; full range.
            unsafe { scan_range(ctx, 0, rows, &mut seq) };
            assert_eq!(par.into_sorted_vec(), seq.into_sorted_vec());
        }
    }
}
