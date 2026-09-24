//! `sovereign` — command-line front end for the sovereign-rag engine.

mod bench;
mod commands;
mod ui;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use sovereign_core::Backend;

/// Zero-copy, SIMD-accelerated retrieval engine.
#[derive(Debug, Parser)]
#[command(name = "sovereign", version, about, propagate_version = true)]
struct Cli {
    /// SIMD backend: auto, scalar, avx2, avx512, neon.
    #[arg(long, global = true, env = "SOVEREIGN_SIMD", default_value = "auto")]
    simd: String,

    /// Log filter (e.g. `warn`, `info`, `sovereign_index=debug`).
    #[arg(long, global = true, env = "SOVEREIGN_LOG", default_value = "warn")]
    log: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Chunk, embed and index files or directories into a store.
    Ingest(IngestArgs),
    /// Search a store with a natural-language query.
    Query(QueryArgs),
    /// Show a store's configuration, segments and memory mapping.
    Info(StoreArg),
    /// Merge all segments of a store into one (rebuilds the graph).
    Compact(CompactArgs),
    /// Synthetic benchmark: SIMD kernels, exact scan, HNSW build / recall / latency.
    Bench(BenchArgs),
    /// Report CPU SIMD capabilities and the selected backend.
    Simd,
}

/// Store location shared by several commands.
#[derive(Debug, Args)]
pub struct StoreArg {
    /// Index store directory.
    #[arg(short, long, default_value = ".sovereign")]
    pub index: PathBuf,
}

/// Graph construction flags shared by `ingest` and `compact`.
#[derive(Debug, Args, Clone)]
pub struct GraphArgs {
    /// Skip the HNSW graph (exact search only).
    #[arg(long)]
    pub no_hnsw: bool,
    /// HNSW max links per node (layer 0 uses 2·m).
    #[arg(long, default_value_t = 16)]
    pub m: usize,
    /// HNSW construction beam width.
    #[arg(long, default_value_t = 200)]
    pub ef_construction: usize,
    /// Skip fsync (faster; not crash-safe).
    #[arg(long)]
    pub no_sync: bool,
}

/// `sovereign ingest`.
#[derive(Debug, Args)]
pub struct IngestArgs {
    /// Files or directories to ingest.
    #[arg(required = true)]
    pub paths: Vec<PathBuf>,
    #[command(flatten)]
    pub store: StoreArg,
    /// Embedding dimension (fixed at store creation).
    #[arg(long, default_value_t = sovereign_pipeline::HashEmbedder::DEFAULT_DIM)]
    pub dim: usize,
    /// Target chunk size in bytes.
    #[arg(long, default_value_t = 1200)]
    pub chunk_bytes: usize,
    /// Hard maximum chunk size in bytes.
    #[arg(long, default_value_t = 2400)]
    pub max_chunk_bytes: usize,
    /// Chunks per embedding batch.
    #[arg(long, default_value_t = 64)]
    pub batch: usize,
    /// Embedding workers (default: number of CPUs).
    #[arg(long)]
    pub workers: Option<usize>,
    /// Keep duplicate chunks.
    #[arg(long)]
    pub no_dedup: bool,
    #[command(flatten)]
    pub graph: GraphArgs,
}

/// `sovereign query`.
#[derive(Debug, Args)]
pub struct QueryArgs {
    /// Query text.
    #[arg(required = true)]
    pub text: Vec<String>,
    #[command(flatten)]
    pub store: StoreArg,
    /// Number of results.
    #[arg(short, default_value_t = 5)]
    pub k: usize,
    /// HNSW beam width.
    #[arg(long, default_value_t = 64)]
    pub ef: usize,
    /// Exact scan instead of HNSW.
    #[arg(long)]
    pub exact: bool,
    /// Print full chunk text instead of a snippet.
    #[arg(long)]
    pub full: bool,
    /// Repeat the search N times and report latency percentiles.
    #[arg(long, default_value_t = 1)]
    pub repeat: usize,
}

/// `sovereign compact`.
#[derive(Debug, Args)]
pub struct CompactArgs {
    #[command(flatten)]
    pub store: StoreArg,
    #[command(flatten)]
    pub graph: GraphArgs,
}

/// `sovereign bench`.
#[derive(Debug, Args)]
pub struct BenchArgs {
    /// Vector dimensionality.
    #[arg(long, default_value_t = 1536)]
    pub dim: usize,
    /// Number of vectors in the synthetic dataset.
    #[arg(long, default_value_t = 20_000)]
    pub rows: usize,
    /// Number of queries.
    #[arg(long, default_value_t = 200)]
    pub queries: usize,
    /// Neighbors per query.
    #[arg(short, default_value_t = 10)]
    pub k: usize,
    /// HNSW search beam width.
    #[arg(long, default_value_t = 64)]
    pub ef: usize,
    /// HNSW max links per node.
    #[arg(long, default_value_t = 16)]
    pub m: usize,
    /// HNSW construction beam width.
    #[arg(long, default_value_t = 100)]
    pub ef_construction: usize,
    /// Skip the HNSW part.
    #[arg(long)]
    pub no_hnsw: bool,
    /// RNG seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

fn init_simd(requested: &str) -> Result<()> {
    if requested.eq_ignore_ascii_case("auto") {
        let _ = sovereign_core::kernels();
        return Ok(());
    }
    let backend: Backend = requested.parse()?;
    sovereign_core::init_backend(backend)
        .with_context(|| format!("cannot use SIMD backend `{backend}`"))?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&cli.log).unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    init_simd(&cli.simd)?;

    match cli.command {
        Command::Ingest(a) => commands::ingest(&a),
        Command::Query(a) => commands::query(&a),
        Command::Info(a) => commands::info(&a),
        Command::Compact(a) => commands::compact(&a),
        Command::Bench(a) => bench::run(&a),
        Command::Simd => {
            commands::simd();
            Ok(())
        }
    }
}
