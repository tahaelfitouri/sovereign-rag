//! Building and persisting immutable segments.
//!
//! Rows are staged in a 64-byte aligned buffer (already padded and, for cosine, normalized), so
//! the HNSW builder runs on exactly the bytes that will hit the disk, and serialization is a
//! straight `write_all` of the staging buffer — no per-row encoding.
//!
//! Durability protocol (the same one LMDB/RocksDB/SQLite rely on):
//!
//! 1. write everything to `.<name>.tmp-<pid>-<nanos>` in the destination directory;
//! 2. patch in the header (with both CRCs) and `fsync` the file;
//! 3. `rename` over the destination — atomic on POSIX filesystems;
//! 4. `fsync` the directory so the rename itself is durable.
//!
//! A crash at any point leaves either the old state or the new state, never a torn segment; a
//! stray temp file is the worst case, and those are ignored by the store.

use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytemuck::Zeroable;
use sovereign_core::{padded_stride, push_padded_row, AlignedVec, Kernels, MatrixRef, Metric};

use crate::error::{IndexError, Result};
use crate::format::{
    align_up, flags, Section, SegmentHeader, FORMAT_VERSION, HEADER_SIZE, MAGIC, MAX_DIM, MAX_ROWS,
    SECTION_ALIGN,
};
use crate::hnsw::{build_graph, encode, HnswParams};

/// Shape and identity of a segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Similarity metric.
    pub metric: Metric,
    /// Fingerprint of the embedding model/config that produced the vectors.
    pub fingerprint: u64,
}

/// Options for [`SegmentWriter::write`].
#[derive(Clone, Debug)]
pub struct WriteOptions {
    /// Build an HNSW graph with these parameters (`None` = flat-only segment).
    pub hnsw: Option<HnswParams>,
    /// `fsync` the file and directory (disable only for tests / throwaway data).
    pub sync: bool,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self { hnsw: Some(HnswParams::default()), sync: true }
    }
}

/// Statistics about a written segment.
#[derive(Clone, Debug)]
pub struct WriteReport {
    /// Final path.
    pub path: PathBuf,
    /// Rows written.
    pub rows: usize,
    /// File size in bytes.
    pub bytes: u64,
    /// Whether an HNSW graph was built.
    pub has_graph: bool,
    /// Time spent building the graph.
    pub build_time: Duration,
    /// Time spent writing and syncing.
    pub write_time: Duration,
}

/// Accumulates rows in memory and writes them as one immutable segment.
#[derive(Debug)]
pub struct SegmentWriter {
    config: SegmentConfig,
    stride: usize,
    kernels: &'static Kernels,
    vectors: AlignedVec<f32>,
    ids: Vec<u64>,
    payload_offsets: Vec<u64>,
    payload_data: Vec<u8>,
    has_payload: bool,
}

impl SegmentWriter {
    /// Creates an empty writer.
    ///
    /// # Errors
    /// [`IndexError::InvalidConfig`] if `dim` is 0 or exceeds [`MAX_DIM`].
    pub fn new(config: SegmentConfig) -> Result<Self> {
        if config.dim == 0 || config.dim > MAX_DIM {
            return Err(IndexError::InvalidConfig(format!(
                "dim={} must be in 1..={MAX_DIM}",
                config.dim
            )));
        }
        let stride = padded_stride(config.dim)?;
        Ok(Self {
            config,
            stride,
            kernels: sovereign_core::kernels(),
            vectors: AlignedVec::new(),
            ids: Vec::new(),
            payload_offsets: vec![0],
            payload_data: Vec::new(),
            has_payload: false,
        })
    }

    /// Creates an empty writer with room for `rows` rows.
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn with_capacity(config: SegmentConfig, rows: usize) -> Result<Self> {
        let mut w = Self::new(config)?;
        w.vectors.reserve(rows.saturating_mul(w.stride));
        w.ids.reserve(rows);
        w.payload_offsets.reserve(rows);
        Ok(w)
    }

    /// Segment configuration.
    #[must_use]
    pub fn config(&self) -> &SegmentConfig {
        &self.config
    }

    /// Rows staged so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// `true` if no rows are staged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Physical row length.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Approximate heap bytes held by the staging buffers.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.vectors.capacity() * 4
            + self.ids.capacity() * 8
            + self.payload_offsets.capacity() * 8
            + self.payload_data.capacity()
    }

    /// View of the staged (padded, normalized) rows.
    #[must_use]
    pub fn matrix(&self) -> MatrixRef<'_> {
        MatrixRef::new(&self.vectors, self.len(), self.config.dim, self.stride)
            .expect("staging buffer shape is maintained by push")
    }

    /// Stages one row. Returns its row index.
    ///
    /// # Errors
    /// * dimension mismatch, non-finite components, zero norm (cosine), or the row limit.
    pub fn push(&mut self, id: u64, vector: &[f32], payload: Option<&str>) -> Result<u32> {
        if vector.len() != self.config.dim {
            return Err(sovereign_core::CoreError::DimensionMismatch {
                expected: self.config.dim,
                actual: vector.len(),
            }
            .into());
        }
        if let Some(index) = vector.iter().position(|x| !x.is_finite()) {
            return Err(IndexError::NonFinite { id, index });
        }
        self.push_row(id, vector, payload, self.config.metric.normalizes())
    }

    /// Stages a row that is already normalized/padded (used by compaction to copy rows verbatim).
    pub(crate) fn push_prepared(
        &mut self,
        id: u64,
        padded: &[f32],
        payload: Option<&str>,
    ) -> Result<u32> {
        self.push_row(id, &padded[..self.config.dim.min(padded.len())], payload, false)
    }

    fn push_row(
        &mut self,
        id: u64,
        vector: &[f32],
        payload: Option<&str>,
        normalize: bool,
    ) -> Result<u32> {
        let row = self.len();
        if row >= MAX_ROWS {
            return Err(IndexError::TooManyRows { limit: MAX_ROWS });
        }
        let start = self.vectors.len();
        push_padded_row(&mut self.vectors, vector, self.stride)?;
        if normalize {
            if let Err(e) = self.kernels.normalize(&mut self.vectors[start..start + self.stride]) {
                self.vectors.truncate(start);
                return Err(e.into());
            }
        }
        self.ids.push(id);
        if let Some(p) = payload {
            self.has_payload = true;
            self.payload_data.extend_from_slice(p.as_bytes());
        }
        self.payload_offsets.push(self.payload_data.len() as u64);
        Ok(row as u32)
    }

    /// Builds the optional graph and writes the segment atomically to `path`.
    ///
    /// # Errors
    /// I/O failures and invalid HNSW parameters.
    pub fn write(&self, path: &Path, opts: &WriteOptions) -> Result<WriteReport> {
        let t0 = Instant::now();
        let graph = match &opts.hnsw {
            Some(p) if !self.is_empty() => {
                Some(encode(&build_graph(self.matrix(), self.config.metric, p, self.kernels)?))
            }
            _ => None,
        };
        let build_time = t0.elapsed();

        let t1 = Instant::now();
        let mut header = SegmentHeader::zeroed();
        header.magic = MAGIC;
        header.version = FORMAT_VERSION;
        header.metric = self.config.metric.tag();
        header.dim = self.config.dim as u32;
        header.stride = self.stride as u32;
        header.count = self.len() as u64;
        header.fingerprint = self.config.fingerprint;
        if self.config.metric.normalizes() {
            header.flags |= flags::NORMALIZED;
        }

        let vec_bytes = self.vectors.as_bytes();
        let id_bytes: &[u8] = bytemuck::cast_slice(&self.ids);
        let off_bytes: &[u8] = bytemuck::cast_slice(&self.payload_offsets);

        let mut cursor = HEADER_SIZE as u64;
        let mut place = |len: usize| {
            let offset = align_up(cursor, SECTION_ALIGN);
            cursor = offset + len as u64;
            Section { offset, len: len as u64 }
        };
        header.vectors = place(vec_bytes.len());
        header.ids = place(id_bytes.len());
        if self.has_payload {
            header.flags |= flags::HAS_PAYLOAD;
            header.payload_offsets = place(off_bytes.len());
            header.payload_data = place(self.payload_data.len());
        }
        if let Some(g) = &graph {
            header.flags |= flags::HAS_GRAPH;
            header.graph = place(g.len());
        }

        let mut sections: Vec<(Section, &[u8])> =
            vec![(header.vectors, vec_bytes), (header.ids, id_bytes)];
        if self.has_payload {
            sections.push((header.payload_offsets, off_bytes));
            sections.push((header.payload_data, &self.payload_data));
        }
        if let Some(g) = &graph {
            sections.push((header.graph, g));
        }

        let bytes = write_atomically(path, &mut header, &sections, opts.sync)?;
        Ok(WriteReport {
            path: path.to_path_buf(),
            rows: self.len(),
            bytes,
            has_graph: graph.is_some(),
            build_time,
            write_time: t1.elapsed(),
        })
    }
}

/// Deletes the temp file on drop unless disarmed (i.e. on every error path).
struct TempGuard(Option<PathBuf>);

impl Drop for TempGuard {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            let _ = fs::remove_file(p);
        }
    }
}

fn temp_path(path: &Path) -> PathBuf {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    path.with_file_name(format!(".{name}.tmp-{}-{nanos}", std::process::id()))
}

pub(crate) fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(dir).and_then(|d| d.sync_all()).map_err(|e| IndexError::io(dir, e))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

fn write_atomically(
    path: &Path,
    header: &mut SegmentHeader,
    sections: &[(Section, &[u8])],
    sync: bool,
) -> Result<u64> {
    static ZEROS: [u8; SECTION_ALIGN as usize] = [0; SECTION_ALIGN as usize];

    let tmp = temp_path(path);
    let mut guard = TempGuard(Some(tmp.clone()));
    let io = |e| IndexError::io(&tmp, e);

    let file = File::create(&tmp).map_err(io)?;
    let mut w = BufWriter::with_capacity(1 << 20, file);
    w.write_all(&[0u8; HEADER_SIZE]).map_err(io)?;

    let mut crc = crc32fast::Hasher::new();
    let mut pos = HEADER_SIZE as u64;
    for (section, bytes) in sections {
        while pos < section.offset {
            let n = (section.offset - pos).min(ZEROS.len() as u64) as usize;
            w.write_all(&ZEROS[..n]).map_err(io)?;
            crc.update(&ZEROS[..n]);
            pos += n as u64;
        }
        w.write_all(bytes).map_err(io)?;
        crc.update(bytes);
        pos += bytes.len() as u64;
    }

    header.body_crc32 = crc.finalize();
    header.header_crc32 = header.compute_crc();

    let mut file = w.into_inner().map_err(|e| io(e.into_error()))?;
    file.seek(SeekFrom::Start(0)).map_err(io)?;
    file.write_all(bytemuck::bytes_of(header)).map_err(io)?;
    if sync {
        file.sync_all().map_err(io)?;
    }
    drop(file);

    fs::rename(&tmp, path).map_err(|e| IndexError::io(path, e))?;
    guard.0 = None;
    if sync {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            sync_dir(dir)?;
        }
    }
    Ok(pos)
}
