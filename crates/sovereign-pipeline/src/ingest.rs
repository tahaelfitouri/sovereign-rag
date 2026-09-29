//! The ingestion pipeline.
//!
//! ```text
//!                   ring (docs)             rings (batches)          rings (vectors)
//!  ┌──────────┐    ┌──────────┐    ┌─────────┐ ─► ┌──────────┐ ─► ┐
//!  │ discover │ ─► │  reader  │ ─► │ chunker │ ─► │ embed #1 │ ─► ├─► select_all ─► ┌────────┐
//!  │ (blocking│    │ (tokio   │    │ (zero-  │ ─► │ embed #2 │ ─► │                  │ writer │
//!  │  walk)   │    │  fs)     │    │  copy)  │ ─► │ embed #N │ ─► ┘                  └────────┘
//!  └──────────┘    └──────────┘    └─────────┘    └──────────┘   spawn_blocking + Arena
//! ```
//!
//! * Every edge is a bounded SPSC ring, so memory is bounded end to end: when embedding is the
//!   bottleneck, workers' input rings fill, the chunker parks, the reader parks, and file reads
//!   stop — backpressure propagates to the disk instead of piling up in RAM.
//! * Fan-out is round-robin with a non-blocking first pass (`try_send` to the next free worker),
//!   so one slow batch does not stall the others. Fan-in merges the output rings with
//!   `select_all`, so no ring is shared by two producers.
//! * Chunk records carry `Arc<Document>` + byte ranges — chunk text is never copied between
//!   stages. The only copy of the text is the one written into the segment payload.
//! * Embedding runs on the blocking pool (it is CPU-bound and would stall the reactor). Each
//!   worker owns an [`Arena`] that holds the batch's contextualized input strings and is reset per
//!   batch, so building `"file › heading\ntext"` inputs costs zero `malloc`s per chunk.
//! * Exact-duplicate chunks are dropped by the workers *before* embedding via a concurrent
//!   `DashSet` of content hashes (the one place several threads race on shared state).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashSet;
use futures::stream::{self, StreamExt};
use sovereign_core::Arena;
use sovereign_index::SegmentWriter;
use tokio::task::JoinHandle;

use crate::chunker::{estimate_tokens, ChunkKind, Chunker, ChunkerConfig, SourceKind};
use crate::embed::{hash_bytes, Embedder};
use crate::error::{PipelineError, Result};
use crate::metrics::PipelineMetrics;
use crate::payload;
use crate::ring::{self, Consumer, Producer, TrySendError};

/// Directory names never descended into.
const SKIP_DIRS: &[&str] = &["target", "node_modules", "__pycache__", "venv", "dist"];

/// Pipeline tuning knobs.
#[derive(Clone, Debug)]
pub struct IngestConfig {
    /// Chunk size limits.
    pub chunker: ChunkerConfig,
    /// Chunks per embedding call.
    pub batch_size: usize,
    /// Capacity (in batches) of each worker's input and output ring.
    pub queue_depth: usize,
    /// Capacity (in documents) of the reader → chunker ring.
    pub doc_queue: usize,
    /// Number of parallel embedding workers.
    pub embed_workers: usize,
    /// Files larger than this are skipped.
    pub max_file_bytes: u64,
    /// Drop exact-duplicate chunks before embedding.
    pub dedup: bool,
    /// Prefix `file › heading` to the text that is embedded (not to the stored payload).
    pub contextualize: bool,
}

impl Default for IngestConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map_or(4, std::num::NonZeroUsize::get);
        Self {
            chunker: ChunkerConfig::default(),
            batch_size: 64,
            queue_depth: 4,
            doc_queue: 16,
            embed_workers: cpus.clamp(1, 16),
            max_file_bytes: 8 << 20,
            dedup: true,
            contextualize: true,
        }
    }
}

impl IngestConfig {
    fn validate(&self) -> Result<()> {
        self.chunker.validate().map_err(PipelineError::Config)?;
        if self.batch_size == 0
            || self.embed_workers == 0
            || self.queue_depth == 0
            || self.doc_queue == 0
        {
            return Err(PipelineError::Config(
                "batch_size, embed_workers, queue_depth and doc_queue must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// A source document, shared by all of its chunks.
#[derive(Debug)]
pub struct Document {
    /// Path as given / discovered.
    pub path: String,
    /// Full UTF-8 contents.
    pub text: String,
    /// Chunking grammar.
    pub kind: SourceKind,
}

/// A chunk in flight: a byte range into a shared document (no text copy).
#[derive(Clone, Debug)]
pub struct ChunkRecord {
    /// Global, deterministic sequence number (becomes the vector id).
    pub seq: u64,
    /// Owning document.
    pub doc: Arc<Document>,
    /// Byte range of the chunk in `doc.text`.
    pub start: usize,
    /// Exclusive end.
    pub end: usize,
    /// Byte range of the enclosing heading in `doc.text`.
    pub heading: Option<(usize, usize)>,
    /// Content kind.
    pub kind: ChunkKind,
}

impl ChunkRecord {
    /// Chunk text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.doc.text[self.start..self.end]
    }

    /// Enclosing heading text.
    #[must_use]
    pub fn heading(&self) -> Option<&str> {
        self.heading.map(|(s, e)| &self.doc.text[s..e])
    }
}

struct EmbeddedBatch {
    records: Vec<ChunkRecord>,
    vectors: Vec<f32>,
}

/// Summary of a pipeline run.
#[derive(Clone, Debug, Default)]
pub struct IngestReport {
    /// Files discovered.
    pub files_total: u64,
    /// Files chunked.
    pub files_indexed: u64,
    /// Files skipped.
    pub files_skipped: u64,
    /// Chunks produced.
    pub chunks: u64,
    /// Duplicate chunks dropped.
    pub duplicates: u64,
    /// Estimated tokens.
    pub tokens: u64,
    /// Bytes read.
    pub bytes: u64,
    /// Vectors staged.
    pub vectors: u64,
    /// Wall time.
    pub elapsed: Duration,
}

/// Asynchronous ingestion pipeline. See the [module docs](self).
pub struct IngestPipeline {
    embedder: Arc<dyn Embedder>,
    config: IngestConfig,
    metrics: Arc<PipelineMetrics>,
}

impl std::fmt::Debug for IngestPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IngestPipeline")
            .field("embedder", &self.embedder.name())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl IngestPipeline {
    /// Creates a pipeline.
    ///
    /// # Errors
    /// [`PipelineError::Config`] for invalid settings.
    pub fn new(embedder: Arc<dyn Embedder>, config: IngestConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { embedder, config, metrics: Arc::new(PipelineMetrics::new()) })
    }

    /// Live counters (poll them from another task to render progress).
    ///
    /// Counters accumulate across runs and the throughput clock starts at construction, so build
    /// a fresh pipeline per run when reporting rates.
    #[must_use]
    pub fn metrics(&self) -> Arc<PipelineMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Ingests every supported file under `roots` into `writer` and returns it for publishing.
    ///
    /// # Errors
    /// Config/embedder mismatches, missing roots, embedding or index errors. The first *root
    /// cause* is reported (secondary "stage closed" errors from the shutdown cascade are dropped).
    pub async fn run(
        &self,
        roots: &[PathBuf],
        mut writer: SegmentWriter,
    ) -> Result<(SegmentWriter, IngestReport)> {
        let dim = self.embedder.dim();
        if writer.config().dim != dim || writer.config().fingerprint != self.embedder.fingerprint()
        {
            return Err(PipelineError::Config(format!(
                "writer (dim={}, fingerprint={:016x}) does not match embedder `{}` (dim={dim}, fingerprint={:016x})",
                writer.config().dim,
                writer.config().fingerprint,
                self.embedder.name(),
                self.embedder.fingerprint()
            )));
        }
        let started = Instant::now();
        let m = Arc::clone(&self.metrics);

        let roots = roots.to_vec();
        let max = self.config.max_file_bytes;
        let (files, skipped) = tokio::task::spawn_blocking(move || discover(&roots, max)).await??;
        PipelineMetrics::add(&m.files_total, files.len() as u64);
        PipelineMetrics::add(&m.files_skipped, skipped);

        let mut handles: Vec<JoinHandle<Result<()>>> = Vec::new();
        let (doc_tx, doc_rx) = ring::channel(self.config.doc_queue);
        handles.push(tokio::spawn(read_stage(files, doc_tx, Arc::clone(&m))));

        let dedup = self.config.dedup.then(|| Arc::new(DashSet::new()));
        let mut batch_txs = Vec::with_capacity(self.config.embed_workers);
        let mut out_rxs = Vec::with_capacity(self.config.embed_workers);
        for _ in 0..self.config.embed_workers {
            let (btx, brx) = ring::channel(self.config.queue_depth);
            let (otx, orx) = ring::channel(self.config.queue_depth);
            batch_txs.push(btx);
            out_rxs.push(orx);
            handles.push(tokio::spawn(embed_stage(
                brx,
                otx,
                Arc::clone(&self.embedder),
                dedup.clone(),
                Arc::clone(&m),
                self.config.contextualize,
            )));
        }
        handles.push(tokio::spawn(chunk_stage(
            doc_rx,
            batch_txs,
            self.config.chunker,
            self.config.batch_size,
            Arc::clone(&m),
        )));

        // Writer stage runs inline: it owns the SegmentWriter.
        let mut first_err: Option<PipelineError> = None;
        let mut merged = stream::select_all(out_rxs);
        let mut payload_buf = String::new();
        while let Some(batch) = merged.next().await {
            if let Err(e) = write_batch(&mut writer, &batch, dim, &mut payload_buf) {
                first_err = Some(e);
                break;
            }
            PipelineMetrics::add(&m.written, batch.records.len() as u64);
        }
        drop(merged); // closes the output rings; upstream stages observe it and unwind

        for h in handles {
            let res = h.await.map_err(PipelineError::from).and_then(|r| r);
            if let Err(e) = res {
                record_error(&mut first_err, e);
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }

        let s = m.snapshot();
        Ok((
            writer,
            IngestReport {
                files_total: s.files_total,
                files_indexed: s.files_done,
                files_skipped: s.files_skipped,
                chunks: s.chunks,
                duplicates: s.duplicates,
                tokens: s.tokens,
                bytes: s.bytes,
                vectors: s.written,
                elapsed: started.elapsed(),
            },
        ))
    }
}

/// Keeps the most informative error: a root cause beats a `StageClosed` cascade.
fn record_error(slot: &mut Option<PipelineError>, e: PipelineError) {
    match slot {
        None => *slot = Some(e),
        Some(PipelineError::StageClosed(_)) if !matches!(e, PipelineError::StageClosed(_)) => {
            *slot = Some(e)
        }
        Some(_) => {}
    }
}

/// Recursively lists supported files under `roots` (sorted, deterministic). Symlinks and hidden or
/// build directories are skipped. Returns the files and the number skipped for size.
///
/// # Errors
/// [`PipelineError::Io`] if a root does not exist or a directory cannot be read.
pub fn discover(
    roots: &[PathBuf],
    max_file_bytes: u64,
) -> Result<(Vec<(PathBuf, SourceKind)>, u64)> {
    let io = |p: &Path, e| PipelineError::Io { path: p.to_path_buf(), source: e };
    let mut out = Vec::new();
    let mut skipped = 0u64;
    let mut consider = |path: PathBuf, len: u64, out: &mut Vec<(PathBuf, SourceKind)>| {
        if let Some(kind) = SourceKind::from_path(&path) {
            if len <= max_file_bytes {
                out.push((path, kind));
            } else {
                skipped += 1;
            }
        }
    };
    for root in roots {
        let meta = std::fs::metadata(root).map_err(|e| io(root, e))?;
        if meta.is_file() {
            consider(root.clone(), meta.len(), &mut out);
            continue;
        }
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).map_err(|e| io(&dir, e))? {
                let entry = entry.map_err(|e| io(&dir, e))?;
                let ft = entry.file_type().map_err(|e| io(&entry.path(), e))?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if ft.is_dir() {
                    if !name.starts_with('.') && !SKIP_DIRS.contains(&name.as_ref()) {
                        stack.push(entry.path());
                    }
                } else if ft.is_file() {
                    let len = entry.metadata().map_err(|e| io(&entry.path(), e))?.len();
                    consider(entry.path(), len, &mut out);
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out.dedup_by(|a, b| a.0 == b.0);
    Ok((out, skipped))
}

async fn read_stage(
    files: Vec<(PathBuf, SourceKind)>,
    mut tx: Producer<Arc<Document>>,
    m: Arc<PipelineMetrics>,
) -> Result<()> {
    for (path, kind) in files {
        let text = match tokio::fs::read(&path).await.map(String::from_utf8) {
            Ok(Ok(text)) => text,
            Ok(Err(_)) => {
                tracing::debug!(path = %path.display(), "skipping non-UTF-8 file");
                PipelineMetrics::add(&m.files_skipped, 1);
                continue;
            }
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "skipping unreadable file");
                PipelineMetrics::add(&m.files_skipped, 1);
                continue;
            }
        };
        PipelineMetrics::add(&m.bytes, text.len() as u64);
        let doc = Arc::new(Document { path: path.to_string_lossy().into_owned(), text, kind });
        tx.send(doc).await.map_err(|_| PipelineError::StageClosed("chunker"))?;
    }
    Ok(())
}

async fn chunk_stage(
    mut rx: Consumer<Arc<Document>>,
    mut txs: Vec<Producer<Vec<ChunkRecord>>>,
    cfg: ChunkerConfig,
    batch_size: usize,
    m: Arc<PipelineMetrics>,
) -> Result<()> {
    let (mut seq, mut rr) = (0u64, 0usize);
    let mut batch = Vec::with_capacity(batch_size);
    while let Some(doc) = rx.recv().await {
        let base = doc.text.as_ptr() as usize;
        let (mut n, mut tokens) = (0u64, 0u64);
        for c in Chunker::new(&doc.text, doc.kind, cfg) {
            tokens += estimate_tokens(c.text) as u64;
            // `heading` borrows from `doc.text`; store it as a byte range.
            let heading = c.heading.map(|h| {
                let s = h.as_ptr() as usize - base;
                (s, s + h.len())
            });
            batch.push(ChunkRecord {
                seq,
                doc: Arc::clone(&doc),
                start: c.start,
                end: c.end,
                heading,
                kind: c.kind,
            });
            seq += 1;
            n += 1;
            if batch.len() == batch_size {
                let full = std::mem::replace(&mut batch, Vec::with_capacity(batch_size));
                dispatch(&mut txs, &mut rr, full).await?;
            }
        }
        PipelineMetrics::add(&m.chunks, n);
        PipelineMetrics::add(&m.tokens, tokens);
        PipelineMetrics::add(&m.files_done, 1);
    }
    if !batch.is_empty() {
        dispatch(&mut txs, &mut rr, batch).await?;
    }
    Ok(())
}

/// Round-robin fan-out: first pass never blocks (first worker with a free slot wins), otherwise
/// wait on the next worker in rotation.
async fn dispatch(
    txs: &mut [Producer<Vec<ChunkRecord>>],
    rr: &mut usize,
    mut batch: Vec<ChunkRecord>,
) -> Result<()> {
    let n = txs.len();
    for i in 0..n {
        let idx = (*rr + i) % n;
        match txs[idx].try_send(batch) {
            Ok(()) => {
                *rr = (idx + 1) % n;
                return Ok(());
            }
            Err(TrySendError::Full(b)) => batch = b,
            Err(TrySendError::Closed(_)) => return Err(PipelineError::StageClosed("embedder")),
        }
    }
    let idx = *rr;
    *rr = (idx + 1) % n;
    txs[idx].send(batch).await.map_err(|_| PipelineError::StageClosed("embedder"))
}

fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Builds `"file › heading\ntext"` in the arena (heading omitted if the chunk starts with it).
fn contextual_text<'a>(arena: &'a Arena, r: &'a ChunkRecord) -> &'a str {
    let name = file_name(&r.doc.path);
    match r.heading() {
        Some(h) if !r.text().starts_with('#') && !h.is_empty() => {
            arena.alloc_str_concat(&[name, " › ", h, "\n", r.text()])
        }
        _ => arena.alloc_str_concat(&[name, "\n", r.text()]),
    }
}

async fn embed_stage(
    mut rx: Consumer<Vec<ChunkRecord>>,
    mut tx: Producer<EmbeddedBatch>,
    embedder: Arc<dyn Embedder>,
    dedup: Option<Arc<DashSet<u64>>>,
    m: Arc<PipelineMetrics>,
    contextualize: bool,
) -> Result<()> {
    let dim = embedder.dim();
    let mut arena = Arena::with_chunk_size(256 * 1024);
    while let Some(mut records) = rx.recv().await {
        if let Some(seen) = &dedup {
            let before = records.len();
            records.retain(|r| seen.insert(hash_bytes(r.text().as_bytes(), 0)));
            PipelineMetrics::add(&m.duplicates, (before - records.len()) as u64);
            if records.is_empty() {
                continue;
            }
        }
        let emb = Arc::clone(&embedder);
        let t0 = Instant::now();
        let (returned_arena, records, vectors, res) = tokio::task::spawn_blocking(move || {
            arena.reset();
            let mut vectors = vec![0.0f32; records.len() * dim];
            let texts: Vec<&str> = records
                .iter()
                .map(|r| if contextualize { contextual_text(&arena, r) } else { r.text() })
                .collect();
            let res = emb.embed_batch(&texts, &mut vectors);
            drop(texts);
            (arena, records, vectors, res)
        })
        .await?;
        arena = returned_arena;
        res.map_err(|e| PipelineError::Embed(e.0))?;
        PipelineMetrics::add(&m.embed_nanos, t0.elapsed().as_nanos() as u64);
        PipelineMetrics::add(&m.embedded, records.len() as u64);
        tx.send(EmbeddedBatch { records, vectors })
            .await
            .map_err(|_| PipelineError::StageClosed("writer"))?;
    }
    Ok(())
}

fn write_batch(
    writer: &mut SegmentWriter,
    b: &EmbeddedBatch,
    dim: usize,
    buf: &mut String,
) -> Result<()> {
    for (r, v) in b.records.iter().zip(b.vectors.chunks_exact(dim)) {
        buf.clear();
        payload::encode(buf, &r.doc.path, r.start, r.end, r.heading(), r.text());
        writer.push(r.seq, v, Some(buf))?;
    }
    Ok(())
}
