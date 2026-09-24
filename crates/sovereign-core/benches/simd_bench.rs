//! SIMD distance-kernel benchmarks.
//!
//! ```text
//! cargo bench -p sovereign-core --bench simd_bench                 # everything
//! cargo bench -p sovereign-core --bench simd_bench -- cosine_1536  # one group
//! SOVEREIGN_SIMD=avx2 cargo bench ...                              # pin dispatch
//! ```
//!
//! Groups:
//!
//! * `cosine_1536`   — naive 3-pass scalar vs. fused SIMD cosine vs. pre-normalized dot
//!   (what the index actually runs) on OpenAI-sized 1536-d vectors.
//! * `dot_1536`, `l2_1536` — per-backend kernels vs. the naive scalar loop.
//! * `dim_sweep`     — cosine across common embedding sizes (MiniLM 384 … text-embedding-3-large 3072).
//! * `alignment_1536`— 64-byte aligned vs. 4-byte misaligned operands (cache-line split cost).
//! * `scan_1536`     — one query against a block of rows: one row per call vs. 1x4 register blocking.
//! * `topk`          — bounded top-k selection over a million scores.
//!
//! Throughput is reported in bytes of vector data read per iteration, so Criterion's GiB/s
//! column can be compared directly against memory bandwidth.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sovereign_core::simd::reference;
use sovereign_core::{
    padded_stride, push_padded_row, AlignedVec, Backend, Kernels, MatrixRef, TopK, Xoshiro256pp,
};

const DIM: usize = 1536;
const F32: usize = core::mem::size_of::<f32>();

fn random_vec(rng: &mut Xoshiro256pp, n: usize) -> AlignedVec<f32> {
    let mut v = AlignedVec::zeroed(n);
    rng.fill_gaussian(&mut v);
    v
}

fn normalized(v: &AlignedVec<f32>) -> AlignedVec<f32> {
    let mut out = v.clone();
    sovereign_core::kernels().normalize(&mut out).expect("gaussian vector has non-zero norm");
    out
}

fn backends() -> Vec<(Backend, &'static Kernels)> {
    Backend::supported()
        .into_iter()
        .map(|b| (b, Kernels::for_backend(b).expect("supported backend")))
        .collect()
}

fn bench_cosine_1536(c: &mut Criterion) {
    let mut rng = Xoshiro256pp::seed_from_u64(1);
    let (a, b) = (random_vec(&mut rng, DIM), random_vec(&mut rng, DIM));
    let (an, bn) = (normalized(&a), normalized(&b));

    let mut g = c.benchmark_group("cosine_1536");
    g.throughput(Throughput::Bytes((2 * DIM * F32) as u64));
    g.bench_function("scalar_naive_3pass", |bch| {
        bch.iter(|| reference::cosine_naive(black_box(&a), black_box(&b)))
    });
    for (backend, k) in backends() {
        g.bench_function(BenchmarkId::new("fused", backend), |bch| {
            bch.iter(|| k.cosine(black_box(&a), black_box(&b)))
        });
        g.bench_function(BenchmarkId::new("prenormalized_dot", backend), |bch| {
            bch.iter(|| k.dot(black_box(&an), black_box(&bn)))
        });
    }
    g.finish();
}

fn bench_dot_1536(c: &mut Criterion) {
    let mut rng = Xoshiro256pp::seed_from_u64(2);
    let (a, b) = (random_vec(&mut rng, DIM), random_vec(&mut rng, DIM));

    let mut g = c.benchmark_group("dot_1536");
    g.throughput(Throughput::Bytes((2 * DIM * F32) as u64));
    g.bench_function("scalar_naive", |bch| {
        bch.iter(|| reference::dot_naive(black_box(&a), black_box(&b)))
    });
    for (backend, k) in backends() {
        g.bench_function(BenchmarkId::from_parameter(backend), |bch| {
            bch.iter(|| k.dot(black_box(&a), black_box(&b)))
        });
    }
    g.finish();
}

fn bench_l2_1536(c: &mut Criterion) {
    let mut rng = Xoshiro256pp::seed_from_u64(3);
    let (a, b) = (random_vec(&mut rng, DIM), random_vec(&mut rng, DIM));

    let mut g = c.benchmark_group("l2_1536");
    g.throughput(Throughput::Bytes((2 * DIM * F32) as u64));
    g.bench_function("scalar_naive", |bch| {
        bch.iter(|| reference::l2_sq_naive(black_box(&a), black_box(&b)))
    });
    for (backend, k) in backends() {
        g.bench_function(BenchmarkId::from_parameter(backend), |bch| {
            bch.iter(|| k.l2_sq(black_box(&a), black_box(&b)))
        });
    }
    g.finish();
}

fn bench_dim_sweep(c: &mut Criterion) {
    let mut rng = Xoshiro256pp::seed_from_u64(4);
    let best = sovereign_core::kernels();
    let mut g = c.benchmark_group("dim_sweep");
    for dim in [128usize, 384, 768, 1024, 1536, 3072] {
        let (a, b) = (random_vec(&mut rng, dim), random_vec(&mut rng, dim));
        g.throughput(Throughput::Bytes((2 * dim * F32) as u64));
        g.bench_with_input(BenchmarkId::new("scalar_naive", dim), &dim, |bch, _| {
            bch.iter(|| reference::cosine_naive(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new(best.backend().name(), dim), &dim, |bch, _| {
            bch.iter(|| best.cosine(black_box(&a), black_box(&b)))
        });
    }
    g.finish();
}

fn bench_alignment(c: &mut Criterion) {
    let mut rng = Xoshiro256pp::seed_from_u64(5);
    // Allocate one extra float so we can view the buffers at +4 bytes (misaligned by 4).
    let a = random_vec(&mut rng, DIM + 1);
    let b = random_vec(&mut rng, DIM + 1);
    let best = sovereign_core::kernels();

    let mut g = c.benchmark_group("alignment_1536");
    g.throughput(Throughput::Bytes((2 * DIM * F32) as u64));
    g.bench_function(BenchmarkId::new("aligned_64B", best.backend()), |bch| {
        bch.iter(|| best.dot(black_box(&a[..DIM]), black_box(&b[..DIM])))
    });
    g.bench_function(BenchmarkId::new("misaligned_4B", best.backend()), |bch| {
        bch.iter(|| best.dot(black_box(&a[1..]), black_box(&b[1..])))
    });
    g.finish();
}

fn bench_scan(c: &mut Criterion) {
    let mut rng = Xoshiro256pp::seed_from_u64(6);
    let stride = padded_stride(DIM).expect("small dim");
    let best = sovereign_core::kernels();
    let q = random_vec(&mut rng, stride);

    let mut g = c.benchmark_group("scan_1536");
    g.sample_size(30);
    // 256 rows = 1.5 MiB (L2-resident on most server cores), 4096 rows = 24 MiB (LLC / DRAM).
    for rows in [256usize, 4096] {
        let mut buf = AlignedVec::with_capacity(rows * stride);
        let mut tmp = vec![0.0f32; DIM];
        for _ in 0..rows {
            rng.fill_gaussian(&mut tmp);
            push_padded_row(&mut buf, &tmp, stride).expect("row fits stride");
        }
        let m = MatrixRef::new(&buf, rows, DIM, stride).expect("valid shape");
        let mut out = vec![0.0f32; rows];
        g.throughput(Throughput::Bytes((rows * stride * F32) as u64));

        g.bench_with_input(BenchmarkId::new("row_at_a_time", rows), &rows, |bch, _| {
            bch.iter(|| {
                for (i, o) in out.iter_mut().enumerate() {
                    *o = best.dot(&q, m.row_padded(i));
                }
                black_box(&out);
            })
        });
        g.bench_with_input(BenchmarkId::new("blocked_1x4", rows), &rows, |bch, _| {
            bch.iter(|| {
                for (i, o) in out.chunks_exact_mut(4).enumerate() {
                    let r = i * 4;
                    let s = best.dot_x4(
                        &q,
                        [
                            m.row_padded(r),
                            m.row_padded(r + 1),
                            m.row_padded(r + 2),
                            m.row_padded(r + 3),
                        ],
                    );
                    o.copy_from_slice(&s);
                }
                black_box(&out);
            })
        });
    }
    g.finish();
}

fn bench_topk(c: &mut Criterion) {
    let mut rng = Xoshiro256pp::seed_from_u64(7);
    let scores: Vec<f32> = (0..1_000_000).map(|_| rng.next_f32()).collect();
    let mut g = c.benchmark_group("topk");
    g.throughput(Throughput::Elements(scores.len() as u64));
    for k in [10usize, 100] {
        g.bench_with_input(BenchmarkId::new("push_1M", k), &k, |bch, &k| {
            let mut t = TopK::new(k);
            bch.iter(|| {
                t.reset(k);
                for (i, &s) in scores.iter().enumerate() {
                    t.push(i as u64, s);
                }
                black_box(t.threshold())
            })
        });
    }
    g.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = config();
    targets = bench_cosine_1536, bench_dot_1536, bench_l2_1536, bench_dim_sweep,
              bench_alignment, bench_scan, bench_topk
}
criterion_main!(benches);
