//! `ingest`, `query`, `info`, `compact` and `simd` commands.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use sovereign_core::{Backend, Metric};
use sovereign_index::{
    HnswParams, IndexStore, OpenOptions, SearchMode, SearchParams, SearchScratch, SegmentConfig,
    WriteOptions,
};
use sovereign_pipeline::{
    payload, ChunkerConfig, Embedder, HashEmbedder, IngestConfig, IngestPipeline,
};

use crate::{ui, CompactArgs, GraphArgs, IngestArgs, QueryArgs, StoreArg};

fn write_options(g: &GraphArgs) -> WriteOptions {
    WriteOptions {
        hnsw: (!g.no_hnsw)
            .then(|| HnswParams { ef_construction: g.ef_construction, ..HnswParams::with_m(g.m) }),
        sync: !g.no_sync,
    }
}

fn backend_line() -> String {
    let b = sovereign_core::kernels().backend();
    format!("{b} ({}-bit, {} × f32 lanes)", b.register_bits(), b.lanes())
}

pub fn ingest(a: &IngestArgs) -> Result<()> {
    let embedder = Arc::new(HashEmbedder::new(a.dim).map_err(|e| anyhow::anyhow!(e))?);
    let config =
        SegmentConfig { dim: a.dim, metric: Metric::Cosine, fingerprint: embedder.fingerprint() };
    // Validate graph parameters before doing any work (they are otherwise only checked after the
    // whole pipeline has run, at graph-build time).
    let opts = write_options(&a.graph);
    if let Some(p) = &opts.hnsw {
        p.validate()?;
    }
    let store = IndexStore::open_or_create(&a.store.index, config)
        .with_context(|| format!("opening store {}", a.store.index.display()))?;

    let mut cfg = IngestConfig {
        chunker: ChunkerConfig { target_bytes: a.chunk_bytes, max_bytes: a.max_chunk_bytes },
        batch_size: a.batch,
        dedup: !a.no_dedup,
        ..IngestConfig::default()
    };
    if let Some(w) = a.workers {
        cfg.embed_workers = w;
    }
    let workers = cfg.embed_workers;
    let pipeline = IngestPipeline::new(embedder.clone(), cfg)?;
    let metrics = pipeline.metrics();

    println!(
        "{} {} files into {}  ·  embedder {} ({}d)  ·  simd {}  ·  {} workers",
        style("▶").cyan().bold(),
        style("ingesting").bold(),
        style(a.store.index.display()).underlined(),
        embedder.name(),
        a.dim,
        backend_line(),
        workers
    );

    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} [{elapsed_precise}] {bar:28.cyan/blue} {pos}/{len} files  {msg}",
        )?
        .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let (writer, report) = rt.block_on(async {
        let (m, bar) = (Arc::clone(&metrics), pb.clone());
        let ticker = tokio::spawn(async move {
            let mut iv = tokio::time::interval(Duration::from_millis(120));
            loop {
                iv.tick().await;
                let s = m.snapshot();
                bar.set_length(s.files_total);
                bar.set_position(s.files_done + s.files_skipped);
                bar.set_message(format!(
                    "{} chunks · {:>7.0} tok/s · {:>6.1} MiB/s · embed {:>5.1} µs/vec · rss {}",
                    ui::count(s.chunks),
                    s.tokens_per_sec(),
                    s.mib_per_sec(),
                    s.embed_us_per_vector(),
                    ui::rss_bytes().map_or_else(|| "n/a".into(), ui::bytes),
                ));
            }
        });
        let res = pipeline.run(&a.paths, store.writer()?).await;
        ticker.abort();
        res.map_err(anyhow::Error::from)
    })?;
    pb.finish_and_clear();

    ui::header("Pipeline");
    ui::kv("files", format!("{} indexed, {} skipped", report.files_indexed, report.files_skipped));
    ui::kv("input", ui::bytes(report.bytes));
    ui::kv(
        "chunks",
        format!("{} ({} duplicates dropped)", ui::count(report.chunks), report.duplicates),
    );
    ui::kv("tokens (est.)", ui::count(report.tokens));
    let secs = report.elapsed.as_secs_f64().max(1e-9);
    ui::kv("wall time", ui::duration(report.elapsed));
    ui::kv(
        "throughput",
        format!(
            "{:.0} tokens/s · {:.0} chunks/s · {:.1} MiB/s",
            report.tokens as f64 / secs,
            report.chunks as f64 / secs,
            report.bytes as f64 / secs / (1024.0 * 1024.0)
        ),
    );

    if writer.is_empty() {
        println!("\n{} nothing to index", style("!").yellow().bold());
        return Ok(());
    }

    let sp = ui::spinner(if opts.hnsw.is_some() {
        format!(
            "building HNSW graph over {} vectors + writing segment",
            ui::count(writer.len() as u64)
        )
    } else {
        "writing segment".to_owned()
    });
    let w = store.add_segment(&writer, &opts)?;
    sp.finish_and_clear();

    ui::header("Segment");
    ui::kv("file", w.path.display());
    ui::kv("vectors", ui::count(w.rows as u64));
    ui::kv("size", ui::bytes(w.bytes));
    if w.has_graph {
        ui::kv("hnsw build", ui::duration(w.build_time));
    }
    ui::kv(
        if opts.sync { "write + fsync" } else { "write (no fsync)" },
        ui::duration(w.write_time),
    );
    let snap = store.load();
    ui::kv(
        "store",
        format!(
            "generation {} · {} segment(s) · {} vectors",
            snap.generation(),
            snap.segments().len(),
            ui::count(snap.len() as u64)
        ),
    );
    ui::kv("peak rss", ui::peak_rss_bytes().map_or_else(|| "n/a".into(), ui::bytes));
    Ok(())
}

fn open_store(s: &StoreArg) -> Result<IndexStore> {
    IndexStore::open(&s.index, OpenOptions::default()).with_context(|| {
        format!("opening store {} (run `sovereign ingest` first?)", s.index.display())
    })
}

fn snippet(text: &str, max_chars: usize) -> String {
    let mut out = String::with_capacity(max_chars + 1);
    let mut last_space = false;
    for c in text.chars() {
        let c = if c.is_whitespace() { ' ' } else { c };
        if c == ' ' && last_space {
            continue;
        }
        last_space = c == ' ';
        if out.chars().count() >= max_chars {
            out.push('…');
            break;
        }
        out.push(c);
    }
    out.trim().to_owned()
}

pub fn query(a: &QueryArgs) -> Result<()> {
    let store = open_store(&a.store)?;
    let cfg = *store.config();
    let embedder = HashEmbedder::new(cfg.dim).map_err(|e| anyhow::anyhow!(e))?;
    if embedder.fingerprint() != cfg.fingerprint {
        bail!(
            "store was built with a different embedder (fingerprint {:016x}, this build has {:016x})",
            cfg.fingerprint,
            embedder.fingerprint()
        );
    }
    let text = a.text.join(" ");

    let t0 = Instant::now();
    let mut qv = vec![0.0f32; cfg.dim];
    embedder.embed_batch(&[&text], &mut qv).map_err(|e| anyhow::anyhow!(e))?;
    let embed_time = t0.elapsed();

    let params = SearchParams::top(a.k).ef(a.ef).mode(if a.exact {
        SearchMode::Exact
    } else {
        SearchMode::Auto
    });
    let snap = store.snapshot();
    let mut scratch = SearchScratch::new();
    let mut lat = Vec::with_capacity(a.repeat.max(1));
    let mut hits = Vec::new();
    for _ in 0..a.repeat.max(1) {
        let t = Instant::now();
        hits = snap.search_with(&mut scratch, &qv, &params)?;
        lat.push(t.elapsed());
    }
    let first = lat[0];
    lat.sort_unstable();

    println!(
        "{} {}  {}",
        style("?").cyan().bold(),
        style(&text).bold(),
        style(format!(
            "({} vectors · {} segment(s) · {})",
            ui::count(snap.len() as u64),
            snap.segments().len(),
            if a.exact { "exact" } else { "auto: hnsw, exact below 1k rows" }
        ))
        .dim()
    );
    for (rank, h) in hits.iter().enumerate() {
        let seg = &snap.segments()[h.ordinal as usize].segment;
        let raw = seg.payload(h.row as usize)?.unwrap_or("");
        match payload::decode(raw) {
            Some(p) => {
                let heading = p.heading.map(|h| format!(" › {h}")).unwrap_or_default();
                println!(
                    "\n {} {}  {}{}",
                    style(format!("{:>2}.", rank + 1)).bold(),
                    style(format!("{:.4}", h.score)).green(),
                    style(format!("{}#{}..{}", p.path, p.start, p.end)).cyan(),
                    style(heading).yellow()
                );
                let body = if a.full { p.text.to_owned() } else { snippet(p.text, 220) };
                for line in body.lines() {
                    println!("     {line}");
                }
            }
            None => {
                println!("\n {:>2}. {:.4}  id={} {}", rank + 1, h.score, h.id, snippet(raw, 220))
            }
        }
    }
    if hits.is_empty() {
        println!("  (no results)");
    }
    println!();
    ui::kv("embed", ui::duration(embed_time));
    if a.repeat > 1 {
        ui::kv("search (cold)", ui::duration(first));
        ui::kv(
            "search p50 / p99",
            format!(
                "{} / {}",
                ui::duration(ui::percentile(&lat, 50.0)),
                ui::duration(ui::percentile(&lat, 99.0))
            ),
        );
    } else {
        ui::kv("search", ui::duration(first));
    }
    ui::kv("simd", backend_line());
    Ok(())
}

pub fn info(a: &StoreArg) -> Result<()> {
    let store = open_store(a)?;
    let snap = store.snapshot();
    let cfg = store.config();
    ui::header("Store");
    ui::kv("path", store.dir().display());
    ui::kv(
        "dimension",
        format!("{} (stride {})", cfg.dim, sovereign_core::padded_stride(cfg.dim)?),
    );
    ui::kv("metric", cfg.metric);
    ui::kv("fingerprint", format!("{:016x}", cfg.fingerprint));
    ui::kv("generation", snap.generation());
    ui::kv("vectors", ui::count(snap.len() as u64));
    ui::kv("mapped", ui::bytes(snap.mapped_bytes() as u64));

    ui::header("Segments");
    println!(
        "  {:<18} {:>10} {:>11} {:>7} {:>5} {:>7}  {}",
        style("file").dim(),
        style("vectors").dim(),
        style("size").dim(),
        style("graph").dim(),
        style("m").dim(),
        style("layers").dim(),
        style("payload").dim()
    );
    for e in snap.segments() {
        let s = &e.segment;
        let (g, m, layers) = match s.graph() {
            Some(g) => ("hnsw", g.header().m.to_string(), (g.header().max_level + 1).to_string()),
            None => ("flat", "-".into(), "-".into()),
        };
        let name =
            s.path().file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        println!(
            "  {:<18} {:>10} {:>11} {:>7} {:>5} {:>7}  {}",
            name,
            ui::count(s.len() as u64),
            ui::bytes(s.mapped_bytes() as u64),
            g,
            m,
            layers,
            if s.payload(0).ok().flatten().is_some() { "yes" } else { "no" }
        );
    }
    Ok(())
}

pub fn compact(a: &CompactArgs) -> Result<()> {
    let store = open_store(&a.store)?;
    let before = store.load().segments().len();
    let sp = ui::spinner(format!("compacting {before} segments"));
    let res = store.compact(&write_options(&a.graph))?;
    sp.finish_and_clear();
    match res {
        Some(r) => {
            ui::header("Compaction");
            ui::kv("segments", format!("{before} → {}", store.load().segments().len()));
            ui::kv("vectors", ui::count(r.rows as u64));
            ui::kv("new segment", r.path.display());
            ui::kv("size", ui::bytes(r.bytes));
            ui::kv("build + write", ui::duration(r.build_time + r.write_time));
        }
        None => println!("nothing to compact ({before} segment)"),
    }
    Ok(())
}

pub fn simd() {
    ui::header("SIMD");
    if let Some(cpu) = ui::cpu_model() {
        ui::kv("cpu", cpu);
    }
    ui::kv("architecture", std::env::consts::ARCH);
    for b in Backend::ALL {
        let mark = if b.is_supported() {
            style("✔ supported").green()
        } else {
            style("✘ unavailable").red()
        };
        ui::kv(b.name(), format!("{mark}  ({}-bit)", b.register_bits()));
    }
    ui::kv("active", style(backend_line()).bold());
    for dim in [100usize, 384, 768, 1536] {
        ui::kv(
            &format!("lane efficiency {dim}d"),
            format!("{:.1}%", 100.0 * ui::lane_efficiency(dim)),
        );
    }
}
