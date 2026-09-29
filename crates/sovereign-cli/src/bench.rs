//! `sovereign bench`: an end-to-end synthetic benchmark.
//!
//! 1. Kernel throughput per SIMD backend vs. the naive scalar loop (ns/op, GFLOP/s, GB/s).
//! 2. Build a segment of clustered random vectors (optionally with HNSW) on disk and map it.
//! 3. Exact-scan and HNSW query latency (p50/p99, QPS) and HNSW recall@k against the exact scan.
//!
//! For statistically rigorous kernel numbers use the Criterion suite
//! (`cargo bench -p sovereign-core`); this command is the quick, whole-system view.

use std::collections::HashSet;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use console::style;
use sovereign_core::simd::reference;
use sovereign_core::{AlignedVec, Backend, Kernels, Metric, Xoshiro256pp};
use sovereign_index::{
    HnswParams, SearchMode, SearchParams, SearchScratch, Segment, SegmentConfig, SegmentWriter,
    WriteOptions,
};

use crate::{ui, BenchArgs};

/// Removes the benchmark directory on exit (including on error).
struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Times `f` over enough iterations to run for roughly `budget`, returning ns per call.
fn time_per_call(budget: Duration, mut f: impl FnMut() -> f32) -> f64 {
    // Warm up and calibrate.
    let mut iters = 64u64;
    loop {
        let t = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        let e = t.elapsed();
        if e >= budget / 8 || iters >= 1 << 26 {
            let scale = (budget.as_secs_f64() / e.as_secs_f64().max(1e-9)).max(1.0);
            iters = ((iters as f64) * scale) as u64;
            break;
        }
        iters *= 4;
    }
    let t = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    t.elapsed().as_nanos() as f64 / iters as f64
}

fn kernels_section(dim: usize, seed: u64) {
    ui::header(&format!("Kernels · {dim}-d · single core"));
    let mut rng = Xoshiro256pp::seed_from_u64(seed);
    let mut a = AlignedVec::zeroed(dim);
    let mut b = AlignedVec::zeroed(dim);
    rng.fill_gaussian(&mut a);
    rng.fill_gaussian(&mut b);
    let budget = Duration::from_millis(250);
    let bytes = (2 * dim * 4) as f64;

    println!(
        "  {:<22} {:>10} {:>10} {:>10} {:>9}",
        style("kernel").dim(),
        style("ns/op").dim(),
        style("GFLOP/s").dim(),
        style("GB/s").dim(),
        style("speedup").dim()
    );
    let row = |name: &str, ns: f64, flops_per_elem: f64, base: f64| {
        let gflops = flops_per_elem * dim as f64 / ns;
        println!(
            "  {:<22} {:>10.1} {:>10.2} {:>10.2} {:>8.1}x",
            name,
            ns,
            gflops,
            bytes / ns,
            base / ns
        );
    };

    let base_cos = time_per_call(budget, || reference::cosine_naive(black_box(&a), black_box(&b)));
    row("cosine · naive scalar", base_cos, 6.0, base_cos);
    for backend in Backend::supported() {
        let k = Kernels::for_backend(backend).expect("supported");
        let ns = time_per_call(budget, || k.cosine(black_box(&a), black_box(&b)));
        row(&format!("cosine · {backend}"), ns, 6.0, base_cos);
    }
    let base_dot = time_per_call(budget, || reference::dot_naive(black_box(&a), black_box(&b)));
    row("dot · naive scalar", base_dot, 2.0, base_dot);
    for backend in Backend::supported() {
        let k = Kernels::for_backend(backend).expect("supported");
        let ns = time_per_call(budget, || k.dot(black_box(&a), black_box(&b)));
        row(&format!("dot · {backend}"), ns, 2.0, base_dot);
    }
}

fn clustered(rng: &mut Xoshiro256pp, centers: &[Vec<f32>], i: usize, out: &mut [f32]) {
    let c = &centers[i % centers.len()];
    for (o, &x) in out.iter_mut().zip(c) {
        *o = x + 0.4 * rng.next_gaussian();
    }
}

fn latency_line(name: &str, lat: &mut [Duration]) {
    lat.sort_unstable();
    let total: Duration = lat.iter().sum();
    let qps = lat.len() as f64 / total.as_secs_f64().max(1e-9);
    ui::kv(
        name,
        format!(
            "p50 {:>9}  p99 {:>9}  {:>9.0} QPS (queries issued serially)",
            ui::duration(ui::percentile(lat, 50.0)),
            ui::duration(ui::percentile(lat, 99.0)),
            qps
        ),
    );
}

pub fn run(a: &BenchArgs) -> Result<()> {
    anyhow::ensure!(
        a.dim > 0 && a.rows > 0 && a.queries > 0 && a.k > 0,
        "--dim, --rows, --queries and -k must all be at least 1"
    );
    let active = sovereign_core::kernels().backend();
    ui::header("System");
    if let Some(cpu) = ui::cpu_model() {
        ui::kv("cpu", cpu);
    }
    ui::kv("threads", std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get));
    ui::kv(
        "simd backend",
        format!("{active} ({}-bit, {} lanes)", active.register_bits(), active.lanes()),
    );
    ui::kv(
        "lane efficiency",
        format!("{:.1}% at {}d (rows padded to 64 B)", 100.0 * ui::lane_efficiency(a.dim), a.dim),
    );

    kernels_section(a.dim, a.seed);

    // ---- dataset ------------------------------------------------------------------------
    ui::header(&format!(
        "Dataset · {} × {}-d (clustered gaussian)",
        ui::count(a.rows as u64),
        a.dim
    ));
    let rss0 = ui::rss_bytes();
    let mut rng = Xoshiro256pp::seed_from_u64(a.seed);
    let centers: Vec<Vec<f32>> = (0..64)
        .map(|_| {
            let mut c = vec![0.0; a.dim];
            rng.fill_gaussian(&mut c);
            c
        })
        .collect();
    let config = SegmentConfig { dim: a.dim, metric: Metric::Cosine, fingerprint: 0xBE7C };
    let mut writer = SegmentWriter::with_capacity(config, a.rows)?;
    let mut v = vec![0.0f32; a.dim];
    let t = Instant::now();
    for i in 0..a.rows {
        clustered(&mut rng, &centers, i, &mut v);
        writer.push(i as u64, &v, None)?;
    }
    ui::kv("generate + normalize", ui::duration(t.elapsed()));
    ui::kv("staging memory", ui::bytes(writer.memory_bytes() as u64));

    let dir = TempDir(std::env::temp_dir().join(format!("sovereign-bench-{}", std::process::id())));
    std::fs::create_dir_all(&dir.0)?;
    let path = dir.0.join("bench.srag");
    let opts = WriteOptions {
        hnsw: (!a.no_hnsw)
            .then(|| HnswParams { ef_construction: a.ef_construction, ..HnswParams::with_m(a.m) }),
        sync: false,
    };
    let sp = ui::spinner(if a.no_hnsw {
        "writing segment".into()
    } else {
        format!("building HNSW (m={}, ef_c={})", a.m, a.ef_construction)
    });
    let report = writer.write(&path, &opts)?;
    sp.finish_and_clear();
    drop(writer);
    if report.has_graph {
        ui::kv(
            "hnsw build",
            format!(
                "{}  ({:.0} inserts/s, {} threads)",
                ui::duration(report.build_time),
                a.rows as f64 / report.build_time.as_secs_f64().max(1e-9),
                std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
            ),
        );
    }
    ui::kv(
        "segment",
        format!("{} written in {}", ui::bytes(report.bytes), ui::duration(report.write_time)),
    );

    let t = Instant::now();
    let seg = Segment::open(&path)?;
    ui::kv("open (mmap + validate)", ui::duration(t.elapsed()));

    // ---- queries ------------------------------------------------------------------------
    ui::header(&format!("Search · k={} · {} queries", a.k, a.queries));
    let queries: Vec<Vec<f32>> = (0..a.queries)
        .map(|i| {
            let mut q = vec![0.0; a.dim];
            clustered(&mut rng, &centers, i * 7 + 3, &mut q);
            q
        })
        .collect();
    let mut scratch = SearchScratch::new();
    let mut out = Vec::with_capacity(a.k);

    // Warm the page cache (vectors via one exact scan, graph via one graph search) so timings
    // measure compute, not first-touch page faults.
    let warm = SearchParams::top(a.k).mode(SearchMode::Exact);
    seg.search_into(&mut scratch, &queries[0], &warm, &mut out)?;
    if seg.has_graph() {
        let warm = SearchParams::top(a.k).ef(a.ef * 2).mode(SearchMode::Approximate);
        seg.search_into(&mut scratch, &queries[0], &warm, &mut out)?;
    }

    let exact = SearchParams::top(a.k).mode(SearchMode::Exact);
    let mut truth: Vec<HashSet<u64>> = Vec::with_capacity(queries.len());
    let mut lat = Vec::with_capacity(queries.len());
    for q in &queries {
        let t = Instant::now();
        seg.search_into(&mut scratch, q, &exact, &mut out)?;
        lat.push(t.elapsed());
        truth.push(out.iter().map(|h| h.id).collect());
    }
    latency_line("exact scan (rayon)", &mut lat);
    let scanned = (a.rows * sovereign_core::padded_stride(a.dim)? * 4) as f64;
    ui::kv(
        "scan bandwidth",
        format!(
            "{:.1} GB/s effective (p50)",
            scanned / ui::percentile(&lat, 50.0).as_nanos() as f64
        ),
    );

    if seg.has_graph() {
        for ef in [a.ef / 2, a.ef, a.ef * 2].into_iter().filter(|&e| e >= a.k) {
            // HNSW queries are single-threaded: one core per query, scale out by running many.
            let p = SearchParams::top(a.k).ef(ef).mode(SearchMode::Approximate);
            let mut lat = Vec::with_capacity(queries.len());
            let mut hits = 0usize;
            for (q, t_set) in queries.iter().zip(&truth) {
                let t = Instant::now();
                seg.search_into(&mut scratch, q, &p, &mut out)?;
                lat.push(t.elapsed());
                hits += out.iter().filter(|h| t_set.contains(&h.id)).count();
            }
            let recall = hits as f64 / (queries.len() * a.k).max(1) as f64;
            latency_line(&format!("hnsw ef={ef}"), &mut lat);
            ui::kv("", format!("recall@{} = {}", a.k, style(format!("{:.3}", recall)).bold()));
        }
    }

    ui::header("Memory");
    ui::kv("mapped segment", ui::bytes(seg.mapped_bytes() as u64));
    if let (Some(r0), Some(r1)) = (rss0, ui::rss_bytes()) {
        ui::kv("rss (start → now)", format!("{} → {}", ui::bytes(r0), ui::bytes(r1)));
    }
    println!(
        "\n  {}",
        style(
            "note: RSS includes page-cache pages of the mapped segment that this process touched"
        )
        .dim()
    );
    Ok(())
}
