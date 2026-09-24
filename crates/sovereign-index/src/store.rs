//! A directory of immutable segments with lock-free, RCU-style snapshot swaps.
//!
//! # Concurrency model
//!
//! ```text
//!            readers (any number, any thread)                 writer (one at a time)
//!   ┌──────────────────────────────────────────┐     ┌────────────────────────────────┐
//!   │ let snap = store.load();  // ArcSwap     │     │ write seg-N.srag (tmp+rename)  │
//!   │ snap.search(q)            // lock-free   │     │ open + validate (mmap)         │
//!   │ drop(snap)                               │     │ lock publish mutex             │
//!   └──────────────────────────────────────────┘     │ write MANIFEST (tmp+rename)    │
//!                    │                               │ current.store(Arc::new(new))   │◄─ atomic
//!                    ▼                               └────────────────────────────────┘
//!     ArcSwap<Snapshot> ──► Snapshot { generation, segments: [Arc<Segment>, ...] }
//! ```
//!
//! * Readers never block and are never blocked: `ArcSwap::load` is a handful of atomic ops with no
//!   locks, and a reader keeps its snapshot alive for exactly as long as it holds the guard.
//! * A segment is unmapped only when the last snapshot referencing it is dropped, so a reader can
//!   never observe an unmapped (or deleted) region — even while compaction deletes the file: on
//!   POSIX an unlinked-but-mapped file lives on until `munmap`.
//! * Writers serialize on a mutex held only for the manifest write + pointer swap; the expensive
//!   parts (HNSW build, file write) happen outside it, so ingestion and compaction overlap.
//! * Cross-process: an advisory lock on `LOCK` (taken on first write) enforces a single writer
//!   process; any number of reader processes may open the same directory.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use arc_swap::{ArcSwap, Guard};
use parking_lot::Mutex;
use sovereign_core::{padded_stride, Metric, TopK};

use crate::error::{IndexError, Result};
use crate::query::{SearchParams, SearchScratch};
use crate::segment::{OpenOptions, Segment};
use crate::writer::{sync_dir, SegmentConfig, SegmentWriter, WriteOptions, WriteReport};

const MANIFEST: &str = "MANIFEST";
const LOCK: &str = "LOCK";
const MANIFEST_MAGIC: &str = "sovereign-manifest v1";

/// Store-wide configuration, recorded in the manifest.
pub type StoreConfig = SegmentConfig;

/// A live segment and its store-assigned id.
#[derive(Debug, Clone)]
pub struct SegmentEntry {
    /// Monotonic segment id (also encoded in the file name).
    pub id: u64,
    /// The mapped segment.
    pub segment: Arc<Segment>,
}

/// An immutable, consistent view of the store at one generation.
#[derive(Debug)]
pub struct Snapshot {
    generation: u64,
    config: StoreConfig,
    segments: Vec<SegmentEntry>,
}

/// One search result from a store.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StoreHit {
    /// Id of the segment that produced the hit.
    pub segment_id: u64,
    /// Position of that segment in the snapshot.
    pub ordinal: u32,
    /// Row inside the segment.
    pub row: u32,
    /// External id.
    pub id: u64,
    /// Similarity score (higher is better).
    pub score: f32,
}

/// Search results pinned to the snapshot they came from, so payloads can be borrowed zero-copy
/// even if the store is swapped in the meantime.
#[derive(Debug)]
pub struct SearchResults {
    snapshot: Arc<Snapshot>,
    /// Hits, best first.
    pub hits: Vec<StoreHit>,
}

impl SearchResults {
    /// Payload text of a hit (borrowed from the mapped segment).
    ///
    /// # Errors
    /// Payload corruption or invalid UTF-8.
    pub fn payload(&self, hit: &StoreHit) -> Result<Option<&str>> {
        match self.snapshot.segments.get(hit.ordinal as usize) {
            Some(e) => e.segment.payload(hit.row as usize),
            None => Ok(None),
        }
    }

    /// The snapshot these results were computed against.
    #[must_use]
    pub fn snapshot(&self) -> &Arc<Snapshot> {
        &self.snapshot
    }
}

impl Snapshot {
    /// Monotonic generation number (bumped on every publish/compaction).
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Store configuration.
    #[must_use]
    pub fn config(&self) -> &StoreConfig {
        &self.config
    }

    /// Live segments.
    #[must_use]
    pub fn segments(&self) -> &[SegmentEntry] {
        &self.segments
    }

    /// Total rows across segments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.iter().map(|e| e.segment.len()).sum()
    }

    /// `true` if the snapshot has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total mapped bytes.
    #[must_use]
    pub fn mapped_bytes(&self) -> usize {
        self.segments.iter().map(|e| e.segment.mapped_bytes()).sum()
    }

    /// Searches every segment and merges the per-segment top-k (keys: `ordinal << 32 | row`).
    ///
    /// # Errors
    /// Query dimension mismatch / non-finite / zero norm.
    pub fn search_with(
        &self,
        scratch: &mut SearchScratch,
        query: &[f32],
        params: &SearchParams,
    ) -> Result<Vec<StoreHit>> {
        let stride = padded_stride(self.config.dim)?;
        scratch.prepare(
            query,
            self.config.dim,
            stride,
            self.config.metric,
            sovereign_core::kernels(),
        )?;
        scratch.topk.reset(params.k);
        let SearchScratch { query: q, topk, layer } = scratch;
        for (ordinal, e) in self.segments.iter().enumerate() {
            e.segment.search_prepared(q, params, layer, topk, (ordinal as u64) << 32);
        }
        Ok(self.resolve(topk))
    }

    fn resolve(&self, topk: &mut TopK) -> Vec<StoreHit> {
        topk.sort_best_first()
            .iter()
            .filter_map(|n| {
                let ordinal = (n.id >> 32) as u32;
                let row = (n.id & u64::from(u32::MAX)) as u32;
                let e = self.segments.get(ordinal as usize)?;
                Some(StoreHit {
                    segment_id: e.id,
                    ordinal,
                    row,
                    id: e.segment.id(row as usize)?,
                    score: n.score,
                })
            })
            .collect()
    }
}

/// Resets the compaction flag on scope exit.
struct FlagGuard<'a>(&'a AtomicBool);
impl Drop for FlagGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// A directory of segments. See the [module docs](self).
#[derive(Debug)]
pub struct IndexStore {
    dir: PathBuf,
    config: StoreConfig,
    current: ArcSwap<Snapshot>,
    next_segment_id: AtomicU64,
    publish: Mutex<()>,
    compacting: AtomicBool,
    writer_lock: OnceLock<File>,
    open_options: OpenOptions,
}

impl IndexStore {
    /// Opens an existing store, or creates an empty one with `config`.
    ///
    /// # Errors
    /// I/O errors, a corrupt manifest/segment, or a config mismatch with an existing store.
    pub fn open_or_create(dir: impl AsRef<Path>, config: StoreConfig) -> Result<Self> {
        let dir = dir.as_ref();
        if dir.join(MANIFEST).exists() {
            let store = Self::open(dir, OpenOptions::default())?;
            if store.config != config {
                return Err(IndexError::Incompatible(format!(
                    "store at {} has dim={} metric={} fingerprint={:016x}, requested dim={} metric={} fingerprint={:016x}",
                    dir.display(),
                    store.config.dim,
                    store.config.metric,
                    store.config.fingerprint,
                    config.dim,
                    config.metric,
                    config.fingerprint
                )));
            }
            return Ok(store);
        }
        fs::create_dir_all(dir).map_err(|e| IndexError::io(dir, e))?;
        SegmentWriter::new(config)?; // validates the config
        let snapshot = Snapshot { generation: 0, config, segments: Vec::new() };
        let store =
            Self::from_parts(dir.to_path_buf(), config, snapshot, 0, OpenOptions::default());
        store.acquire_writer_lock()?;
        store.write_manifest(&store.current.load())?;
        Ok(store)
    }

    /// Opens an existing store read-write (the writer lock is taken lazily on first write).
    ///
    /// # Errors
    /// Missing/corrupt manifest or segments.
    pub fn open(dir: impl AsRef<Path>, open_options: OpenOptions) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let manifest = read_manifest(&dir)?;
        let mut segments = Vec::with_capacity(manifest.segments.len());
        for (id, name) in &manifest.segments {
            let seg = Segment::open_with(dir.join(name), &open_options)?;
            check_compatible(&manifest.config, &seg)?;
            segments.push(SegmentEntry { id: *id, segment: Arc::new(seg) });
        }
        let snapshot =
            Snapshot { generation: manifest.generation, config: manifest.config, segments };
        Ok(Self::from_parts(dir, manifest.config, snapshot, manifest.next_segment, open_options))
    }

    fn from_parts(
        dir: PathBuf,
        config: StoreConfig,
        snapshot: Snapshot,
        next_segment: u64,
        open_options: OpenOptions,
    ) -> Self {
        Self {
            dir,
            config,
            current: ArcSwap::from_pointee(snapshot),
            next_segment_id: AtomicU64::new(next_segment),
            publish: Mutex::new(()),
            compacting: AtomicBool::new(false),
            writer_lock: OnceLock::new(),
            open_options,
        }
    }

    /// Store directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Store configuration.
    #[must_use]
    pub fn config(&self) -> &StoreConfig {
        &self.config
    }

    /// Cheap, lock-free access to the current snapshot (does not bump the refcount in the common
    /// case). Hold it only for the duration of one operation.
    #[must_use]
    pub fn load(&self) -> Guard<Arc<Snapshot>> {
        self.current.load()
    }

    /// An owned handle to the current snapshot, safe to keep indefinitely.
    #[must_use]
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.current.load_full()
    }

    /// A writer configured for this store.
    ///
    /// # Errors
    /// Never in practice (the config was validated at creation).
    pub fn writer(&self) -> Result<SegmentWriter> {
        SegmentWriter::new(self.config)
    }

    /// Searches the current snapshot.
    ///
    /// # Errors
    /// Query validation errors.
    pub fn search(&self, query: &[f32], params: &SearchParams) -> Result<SearchResults> {
        let mut scratch = SearchScratch::new();
        self.search_with(&mut scratch, query, params)
    }

    /// Searches the current snapshot with caller-provided scratch.
    ///
    /// # Errors
    /// Query validation errors.
    pub fn search_with(
        &self,
        scratch: &mut SearchScratch,
        query: &[f32],
        params: &SearchParams,
    ) -> Result<SearchResults> {
        let snapshot = self.snapshot();
        let hits = snapshot.search_with(scratch, query, params)?;
        Ok(SearchResults { snapshot, hits })
    }

    fn acquire_writer_lock(&self) -> Result<()> {
        if self.writer_lock.get().is_some() {
            return Ok(());
        }
        let path = self.dir.join(LOCK);
        let file = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(|e| IndexError::io(&path, e))?;
        match file.try_lock() {
            Ok(()) => {
                let _ = self.writer_lock.set(file);
                Ok(())
            }
            Err(std::fs::TryLockError::WouldBlock) => Err(IndexError::InvalidConfig(format!(
                "store {} is locked by another writer process",
                self.dir.display()
            ))),
            Err(std::fs::TryLockError::Error(e)) => Err(IndexError::io(&path, e)),
        }
    }

    fn segment_path(&self, id: u64) -> PathBuf {
        self.dir.join(format!("seg-{id:010}.srag"))
    }

    /// Writes `writer` as a new segment and atomically publishes it.
    ///
    /// # Errors
    /// Config mismatch, empty writer, I/O or validation failures. On error the store is unchanged.
    pub fn add_segment(&self, writer: &SegmentWriter, opts: &WriteOptions) -> Result<WriteReport> {
        if *writer.config() != self.config {
            return Err(IndexError::Incompatible("writer config differs from store config".into()));
        }
        if writer.is_empty() {
            return Err(IndexError::InvalidConfig("refusing to publish an empty segment".into()));
        }
        self.acquire_writer_lock()?;
        let id = self.next_segment_id.fetch_add(1, Ordering::Relaxed);
        let path = self.segment_path(id);
        let report = writer.write(&path, opts)?;
        let entry = SegmentEntry { id, segment: Arc::new(self.open_new(&path)?) };

        let _g = self.publish.lock();
        let cur = self.current.load_full();
        let mut segments = cur.segments.clone();
        segments.push(entry);
        self.publish_locked(&cur, segments, &[path])?;
        tracing::info!(segment = id, rows = report.rows, "published segment");
        Ok(report)
    }

    fn open_new(&self, path: &Path) -> Result<Segment> {
        let seg = Segment::open_with(path, &self.open_options);
        if seg.is_err() {
            let _ = fs::remove_file(path);
        }
        seg
    }

    /// Writes the manifest for `segments` and swaps it in. Caller holds `publish`.
    /// `new_files` are deleted if the manifest write fails.
    fn publish_locked(
        &self,
        cur: &Snapshot,
        segments: Vec<SegmentEntry>,
        new_files: &[PathBuf],
    ) -> Result<()> {
        let next = Snapshot { generation: cur.generation + 1, config: self.config, segments };
        if let Err(e) = self.write_manifest(&next) {
            for f in new_files {
                let _ = fs::remove_file(f);
            }
            return Err(e);
        }
        self.current.store(Arc::new(next));
        Ok(())
    }

    /// Merges every live segment into one (rebuilding the graph) and swaps it in atomically.
    /// Segments published concurrently while compaction runs are preserved.
    ///
    /// Returns `None` if there is nothing to compact (fewer than two segments).
    ///
    /// # Errors
    /// A compaction already in progress, I/O or validation failures. On error the store is
    /// unchanged.
    pub fn compact(&self, opts: &WriteOptions) -> Result<Option<WriteReport>> {
        if self.compacting.swap(true, Ordering::AcqRel) {
            return Err(IndexError::InvalidConfig("a compaction is already running".into()));
        }
        let _flag = FlagGuard(&self.compacting);
        self.acquire_writer_lock()?;

        let base = self.snapshot();
        if base.segments.len() < 2 {
            return Ok(None);
        }
        let mut writer = SegmentWriter::with_capacity(self.config, base.len())?;
        for e in &base.segments {
            let seg = &e.segment;
            let m = seg.vectors();
            for (row, &id) in seg.ids().iter().enumerate() {
                writer.push_prepared(id, m.row_padded(row), seg.payload(row)?)?;
            }
        }
        let id = self.next_segment_id.fetch_add(1, Ordering::Relaxed);
        let path = self.segment_path(id);
        let report = writer.write(&path, opts)?;
        let merged = SegmentEntry { id, segment: Arc::new(self.open_new(&path)?) };

        let replaced: HashSet<u64> = base.segments.iter().map(|e| e.id).collect();
        let old_paths: Vec<PathBuf> =
            base.segments.iter().map(|e| e.segment.path().to_path_buf()).collect();
        {
            let _g = self.publish.lock();
            let cur = self.current.load_full();
            let mut segments = vec![merged];
            segments.extend(cur.segments.iter().filter(|e| !replaced.contains(&e.id)).cloned());
            self.publish_locked(&cur, segments, &[path])?;
        }
        // Old files may still be mapped by in-flight readers; unlinking is safe on POSIX (the inode
        // lives until the last munmap). On platforms that refuse, the orphan is merely left behind.
        for p in old_paths {
            if let Err(e) = fs::remove_file(&p) {
                tracing::warn!(path = %p.display(), error = %e, "could not remove compacted segment");
            }
        }
        tracing::info!(segment = id, rows = report.rows, "compaction published");
        Ok(Some(report))
    }

    fn write_manifest(&self, snap: &Snapshot) -> Result<()> {
        let mut text = String::new();
        let c = &self.config;
        // Writing into a String cannot fail.
        let _ = writeln!(text, "{MANIFEST_MAGIC}");
        let _ = writeln!(text, "dim {}", c.dim);
        let _ = writeln!(text, "metric {}", c.metric);
        let _ = writeln!(text, "fingerprint {:016x}", c.fingerprint);
        let _ = writeln!(text, "generation {}", snap.generation);
        let _ = writeln!(text, "next-segment {}", self.next_segment_id.load(Ordering::Relaxed));
        for e in &snap.segments {
            let name = e
                .segment
                .path()
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let _ = writeln!(text, "segment {} {name}", e.id);
        }
        let _ = writeln!(text, "checksum {:08x}", crc32fast::hash(text.as_bytes()));

        let path = self.dir.join(MANIFEST);
        let tmp = self.dir.join(format!(".{MANIFEST}.tmp-{}", std::process::id()));
        let io = |e| IndexError::io(&tmp, e);
        let mut f = File::create(&tmp).map_err(io)?;
        f.write_all(text.as_bytes()).map_err(io)?;
        f.sync_all().map_err(io)?;
        drop(f);
        fs::rename(&tmp, &path).map_err(|e| IndexError::io(&path, e))?;
        sync_dir(&self.dir)
    }
}

struct Manifest {
    config: StoreConfig,
    generation: u64,
    next_segment: u64,
    segments: Vec<(u64, String)>,
}

fn read_manifest(dir: &Path) -> Result<Manifest> {
    let path = dir.join(MANIFEST);
    let text = fs::read_to_string(&path).map_err(|e| IndexError::io(&path, e))?;
    let bad = |reason: &str| IndexError::Manifest { path: path.clone(), reason: reason.to_owned() };

    let body_end = text.rfind("checksum ").ok_or_else(|| bad("missing checksum line"))?;
    let stored =
        u32::from_str_radix(text[body_end + 9..].trim(), 16).map_err(|_| bad("bad checksum"))?;
    if crc32fast::hash(&text.as_bytes()[..body_end]) != stored {
        return Err(bad("checksum mismatch"));
    }

    let mut lines = text[..body_end].lines();
    if lines.next() != Some(MANIFEST_MAGIC) {
        return Err(bad("bad magic line"));
    }
    let (mut dim, mut metric, mut fingerprint, mut generation, mut next) =
        (None, None, None, None, None);
    let mut segments = Vec::new();
    for line in lines {
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next(), parts.next()) {
            (Some("dim"), Some(v), None) => dim = v.parse::<usize>().ok(),
            (Some("metric"), Some(v), None) => metric = v.parse::<Metric>().ok(),
            (Some("fingerprint"), Some(v), None) => fingerprint = u64::from_str_radix(v, 16).ok(),
            (Some("generation"), Some(v), None) => generation = v.parse::<u64>().ok(),
            (Some("next-segment"), Some(v), None) => next = v.parse::<u64>().ok(),
            (Some("segment"), Some(id), Some(name)) => {
                let id = id.parse::<u64>().map_err(|_| bad("bad segment id"))?;
                if name.contains('/') || name.contains('\\') || name.starts_with('.') {
                    return Err(bad("segment name must be a plain file name"));
                }
                segments.push((id, name.to_owned()));
            }
            (None, _, _) => {}
            _ => return Err(bad(&format!("unrecognized line `{line}`"))),
        }
    }
    let config = StoreConfig {
        dim: dim.ok_or_else(|| bad("missing dim"))?,
        metric: metric.ok_or_else(|| bad("missing metric"))?,
        fingerprint: fingerprint.ok_or_else(|| bad("missing fingerprint"))?,
    };
    let next_segment = next.ok_or_else(|| bad("missing next-segment"))?;
    if segments.iter().any(|(id, _)| *id >= next_segment) {
        return Err(bad("segment id >= next-segment"));
    }
    Ok(Manifest {
        config,
        generation: generation.ok_or_else(|| bad("missing generation"))?,
        next_segment,
        segments,
    })
}

fn check_compatible(config: &StoreConfig, seg: &Segment) -> Result<()> {
    if seg.dim() != config.dim
        || seg.metric() != config.metric
        || seg.fingerprint() != config.fingerprint
    {
        return Err(IndexError::Incompatible(format!(
            "{}: dim={} metric={} fingerprint={:016x} does not match store",
            seg.path().display(),
            seg.dim(),
            seg.metric(),
            seg.fingerprint()
        )));
    }
    Ok(())
}
