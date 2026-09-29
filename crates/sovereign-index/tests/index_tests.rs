//! End-to-end tests: file format round-trips, search correctness, HNSW recall, corruption
//! handling and lock-free snapshot swaps under concurrent load.

use std::collections::HashSet;
use std::fs::{self, OpenOptions as FsOpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use sovereign_core::{kernels, Metric, Xoshiro256pp};
use sovereign_index::format::HEADER_SIZE;
use sovereign_index::{
    HnswParams, IndexError, IndexStore, SearchMode, SearchParams, SearchScratch, Segment,
    SegmentConfig, SegmentWriter, WriteOptions,
};

fn cfg(dim: usize, metric: Metric) -> SegmentConfig {
    SegmentConfig { dim, metric, fingerprint: 0xFEED_F00D }
}

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xoshiro256pp::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut v = vec![0.0; dim];
            rng.fill_gaussian(&mut v);
            v
        })
        .collect()
}

/// Gaussian blobs: more realistic than uniform noise for embeddings.
fn clustered_vectors(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let centers = random_vectors(clusters, dim, seed ^ 0xABCD);
    let mut rng = Xoshiro256pp::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let c = &centers[i % clusters];
            c.iter().map(|&x| x + 0.35 * rng.next_gaussian()).collect()
        })
        .collect()
}

fn fast_write() -> WriteOptions {
    WriteOptions { hnsw: Some(HnswParams::default()), sync: false }
}

fn flat_write() -> WriteOptions {
    WriteOptions { hnsw: None, sync: false }
}

fn write_segment(path: &Path, data: &[Vec<f32>], metric: Metric, opts: &WriteOptions) -> Segment {
    let mut w = SegmentWriter::new(cfg(data[0].len(), metric)).unwrap();
    for (i, v) in data.iter().enumerate() {
        w.push(1000 + i as u64, v, Some(&format!("doc-{i}"))).unwrap();
    }
    w.write(path, opts).unwrap();
    Segment::open(path).unwrap()
}

/// Ground truth computed in plain `f64` arithmetic on the *raw* vectors — deliberately independent
/// of the SIMD kernels, the stored normalization and the index code under test.
fn f64_score(metric: Metric, q: &[f32], v: &[f32]) -> f64 {
    let dot: f64 = q.iter().zip(v).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
    match metric {
        Metric::Dot => dot,
        Metric::Cosine => {
            let nq: f64 = q.iter().map(|&a| f64::from(a).powi(2)).sum();
            let nv: f64 = v.iter().map(|&b| f64::from(b).powi(2)).sum();
            dot / (nq.sqrt() * nv.sqrt())
        }
        Metric::L2 => {
            -q.iter().zip(v).map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2)).sum::<f64>()
        }
    }
}

fn brute_force(data: &[Vec<f32>], q: &[f32], metric: Metric, k: usize) -> Vec<u64> {
    let mut scored: Vec<(f64, u64)> =
        data.iter().enumerate().map(|(i, v)| (f64_score(metric, q, v), 1000 + i as u64)).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

#[test]
fn roundtrip_preserves_everything() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.srag");
    let data = random_vectors(300, 100, 1);
    let seg = write_segment(&path, &data, Metric::Cosine, &fast_write());
    seg.verify().unwrap();

    assert_eq!(seg.len(), 300);
    assert_eq!(seg.dim(), 100);
    assert_eq!(seg.stride(), 112);
    assert_eq!(seg.fingerprint(), 0xFEED_F00D);
    assert!(seg.has_graph());
    assert!(seg.vectors().is_cache_aligned(), "mmap sections must be 64-byte aligned");
    assert_eq!(seg.ids()[7], 1007);
    assert_eq!(seg.payload(299).unwrap(), Some("doc-299"));
    assert!(seg.payload(300).is_err());

    // Stored rows are normalized copies with zeroed padding.
    let k = kernels();
    for (i, v) in data.iter().enumerate().step_by(37) {
        let row = seg.vectors().row_padded(i);
        assert!((k.norm_sq(row) - 1.0).abs() < 1e-5);
        assert!(row[100..].iter().all(|&x| x == 0.0));
        assert!((k.cosine(&row[..100], v) - 1.0).abs() < 1e-5);
    }
}

#[test]
fn exact_search_matches_brute_force_for_all_metrics() {
    let dir = tempfile::tempdir().unwrap();
    let data = random_vectors(700, 37, 2);
    let queries = random_vectors(20, 37, 3);
    for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
        let path = dir.path().join(format!("{metric}.srag"));
        let seg = write_segment(&path, &data, metric, &flat_write());
        for q in &queries {
            let got: Vec<u64> = seg
                .search(q, &SearchParams::top(10).mode(SearchMode::Exact))
                .unwrap()
                .iter()
                .map(|h| h.id)
                .collect();
            assert_eq!(got, brute_force(&data, q, metric, 10), "{metric}");
        }
    }
}

#[test]
fn hnsw_recall_is_high() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("hnsw.srag");
    let (n, dim, k) = (6000, 48, 10);
    let data = clustered_vectors(n, dim, 40, 4);
    let seg = write_segment(&path, &data, Metric::Cosine, &fast_write());
    let queries = clustered_vectors(200, dim, 40, 5);

    let mut scratch = SearchScratch::new();
    let mut out = Vec::new();
    let mut hits = 0usize;
    for q in &queries {
        seg.search_into(
            &mut scratch,
            q,
            &SearchParams::top(k).ef(128).mode(SearchMode::Approximate),
            &mut out,
        )
        .unwrap();
        let truth: HashSet<u64> = brute_force(&data, q, Metric::Cosine, k).into_iter().collect();
        hits += out.iter().filter(|h| truth.contains(&h.id)).count();
    }
    let recall = hits as f64 / (queries.len() * k) as f64;
    eprintln!(
        "hnsw_recall_is_high: recall@{k} = {recall:.4} (n={n}, dim={dim}, ef=128, 200 queries)"
    );
    assert!(recall >= 0.95, "recall@{k} = {recall:.3}");
}

#[test]
fn every_vector_finds_itself() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("self.srag");
    let data = random_vectors(3000, 32, 6);
    let seg = write_segment(&path, &data, Metric::Cosine, &fast_write());
    let params = SearchParams::top(1).ef(64).mode(SearchMode::Approximate);
    let found = data
        .iter()
        .enumerate()
        .filter(|(i, v)| seg.search(v, &params).unwrap()[0].id == 1000 + *i as u64)
        .count();
    eprintln!("every_vector_finds_itself: {found}/{} (ef=64)", data.len());
    assert!(found as f64 / data.len() as f64 > 0.99, "self-recall {found}/3000");
}

#[test]
fn writer_rejects_bad_input() {
    let mut w = SegmentWriter::new(cfg(4, Metric::Cosine)).unwrap();
    assert!(matches!(w.push(1, &[1.0, 2.0], None), Err(IndexError::Core(_))));
    assert!(matches!(
        w.push(2, &[1.0, f32::NAN, 0.0, 0.0], None),
        Err(IndexError::NonFinite { id: 2, index: 1 })
    ));
    assert!(w.push(3, &[0.0; 4], None).is_err(), "zero vector cannot be normalized");
    assert_eq!(w.len(), 0, "failed pushes must not leave partial rows");
    w.push(4, &[1.0, 0.0, 0.0, 0.0], None).unwrap();
    assert_eq!(w.matrix().rows(), 1);
    assert!(SegmentWriter::new(cfg(0, Metric::Dot)).is_err());

    let seg_dir = tempfile::tempdir().unwrap();
    let path = seg_dir.path().join("x.srag");
    w.write(&path, &flat_write()).unwrap();
    let seg = Segment::open(&path).unwrap();
    assert!(seg.search(&[1.0, 0.0], &SearchParams::top(1)).is_err());
    assert!(seg.search(&[f32::INFINITY, 0.0, 0.0, 0.0], &SearchParams::top(1)).is_err());
    assert_eq!(seg.payload(0).unwrap(), None, "no payloads were stored");
    assert!(seg.search(&[1.0, 0.0, 0.0, 0.0], &SearchParams::top(0)).unwrap().is_empty());
}

fn patch_byte(path: &Path, offset: u64) {
    let mut f = FsOpenOptions::new().read(true).write(true).open(path).unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    let mut b = [0u8; 1];
    std::io::Read::read_exact(&mut f, &mut b).unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    f.write_all(&[b[0] ^ 0xFF]).unwrap();
}

#[test]
fn corruption_is_detected() {
    let dir = tempfile::tempdir().unwrap();
    let data = random_vectors(50, 16, 7);

    let p = dir.path().join("hdr.srag");
    write_segment(&p, &data, Metric::Cosine, &fast_write());
    patch_byte(&p, 16); // dim field
    assert!(matches!(Segment::open(&p), Err(IndexError::Checksum { what: "header", .. })));

    let p = dir.path().join("magic.srag");
    write_segment(&p, &data, Metric::Cosine, &fast_write());
    patch_byte(&p, 0);
    assert!(matches!(Segment::open(&p), Err(IndexError::BadMagic { .. })));

    let p = dir.path().join("body.srag");
    write_segment(&p, &data, Metric::Cosine, &fast_write());
    patch_byte(&p, 4096 + 5);
    let seg = Segment::open(&p).expect("body corruption is only caught by verify()");
    assert!(matches!(seg.verify(), Err(IndexError::Checksum { what: "body", .. })));
    let strict = sovereign_index::OpenOptions { verify_checksum: true, ..Default::default() };
    assert!(Segment::open_with(&p, &strict).is_err());

    let p = dir.path().join("trunc.srag");
    write_segment(&p, &data, Metric::Cosine, &fast_write());
    let len = fs::metadata(&p).unwrap().len();
    FsOpenOptions::new().write(true).open(&p).unwrap().set_len(len - 100).unwrap();
    assert!(matches!(Segment::open(&p), Err(IndexError::Corrupt { .. })));

    let p = dir.path().join("tiny.srag");
    fs::write(&p, [0u8; HEADER_SIZE - 1]).unwrap();
    assert!(matches!(Segment::open(&p), Err(IndexError::Corrupt { .. })));
}

#[test]
fn store_publish_search_compact_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let config = cfg(24, Metric::Cosine);
    let data = random_vectors(3000, 24, 8);
    let queries = random_vectors(10, 24, 9);
    let exact = SearchParams::top(5).mode(SearchMode::Exact);

    let store = IndexStore::open_or_create(dir.path(), config).unwrap();
    for (s, chunk) in data.chunks(1000).enumerate() {
        let mut w = store.writer().unwrap();
        for (i, v) in chunk.iter().enumerate() {
            let id = 1000 + (s * 1000 + i) as u64;
            w.push(id, v, Some(&format!("doc-{}", id - 1000))).unwrap();
        }
        store.add_segment(&w, &fast_write()).unwrap();
    }
    assert_eq!(store.load().segments().len(), 3);
    assert_eq!(store.load().generation(), 3);

    let before: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| store.search(q, &exact).unwrap().hits.iter().map(|h| h.id).collect())
        .collect();
    for (q, got) in queries.iter().zip(&before) {
        assert_eq!(got, &brute_force(&data, q, Metric::Cosine, 5));
    }
    let r = store.search(&queries[0], &exact).unwrap();
    let top = r.hits[0];
    assert_eq!(r.payload(&top).unwrap(), Some(format!("doc-{}", top.id - 1000).as_str()));

    let pinned = store.snapshot();
    let report = store.compact(&fast_write()).unwrap().expect("3 segments to compact");
    assert_eq!(report.rows, 3000);
    assert_eq!(store.load().segments().len(), 1);
    // The pinned pre-compaction snapshot is still fully usable (files unlinked but mapped).
    let mut scratch = SearchScratch::new();
    assert_eq!(pinned.search_with(&mut scratch, &queries[0], &exact).unwrap().len(), 5);

    let after: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| store.search(q, &exact).unwrap().hits.iter().map(|h| h.id).collect())
        .collect();
    assert_eq!(before, after, "compaction must not change exact results");
    assert!(store.compact(&fast_write()).unwrap().is_none(), "single segment: nothing to do");
    let generation = store.load().generation();
    drop(store);

    let reopened = IndexStore::open(dir.path(), Default::default()).unwrap();
    assert_eq!(reopened.load().generation(), generation);
    assert_eq!(reopened.load().len(), 3000);
    let again: Vec<u64> =
        reopened.search(&queries[3], &exact).unwrap().hits.iter().map(|h| h.id).collect();
    assert_eq!(again, before[3]);

    let other = cfg(25, Metric::Cosine);
    assert!(matches!(
        IndexStore::open_or_create(dir.path(), other),
        Err(IndexError::Incompatible(_))
    ));
}

#[test]
fn store_rejects_second_writer_process() {
    let dir = tempfile::tempdir().unwrap();
    let config = cfg(8, Metric::Dot);
    let a = IndexStore::open_or_create(dir.path(), config).unwrap();
    let b = IndexStore::open(dir.path(), Default::default()).unwrap();
    let mut w = b.writer().unwrap();
    w.push(1, &[1.0; 8], None).unwrap();
    assert!(b.add_segment(&w, &flat_write()).is_err(), "LOCK is held by `a`");
    a.add_segment(&w, &flat_write()).unwrap();
}

#[test]
fn manifest_tampering_is_detected() {
    let dir = tempfile::tempdir().unwrap();
    IndexStore::open_or_create(dir.path(), cfg(8, Metric::Dot)).unwrap();
    let path = dir.path().join("MANIFEST");
    let text = fs::read_to_string(&path).unwrap().replace("dim 8", "dim 9");
    fs::write(&path, text).unwrap();
    assert!(matches!(
        IndexStore::open(dir.path(), Default::default()),
        Err(IndexError::Manifest { .. })
    ));
}

#[test]
fn readers_never_block_or_fail_during_swaps() {
    let dir = tempfile::tempdir().unwrap();
    let dim = 32;
    let store = Arc::new(IndexStore::open_or_create(dir.path(), cfg(dim, Metric::Cosine)).unwrap());
    let seed_data = random_vectors(500, dim, 10);
    let mut w = store.writer().unwrap();
    for (i, v) in seed_data.iter().enumerate() {
        w.push(i as u64, v, None).unwrap();
    }
    store.add_segment(&w, &fast_write()).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..4)
        .map(|t| {
            let store = Arc::clone(&store);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let queries = random_vectors(16, dim, 100 + t);
                let mut scratch = SearchScratch::new();
                let (mut n, mut last_gen, mut last_len) = (0u64, 0u64, 0usize);
                while !stop.load(Ordering::Relaxed) {
                    let snap = store.snapshot();
                    assert!(snap.generation() >= last_gen, "generations must be monotonic");
                    assert!(snap.len() >= last_len, "rows are only ever added");
                    last_gen = snap.generation();
                    last_len = snap.len();
                    let q = &queries[(n % 16) as usize];
                    let hits = snap.search_with(&mut scratch, q, &SearchParams::top(5)).unwrap();
                    assert_eq!(hits.len(), 5);
                    n += 1;
                }
                n
            })
        })
        .collect();

    // Writer: keep publishing segments and compacting underneath the readers.
    for round in 0..6u64 {
        let batch = random_vectors(400, dim, 1000 + round);
        let mut w = store.writer().unwrap();
        for (i, v) in batch.iter().enumerate() {
            w.push(10_000 * (round + 1) + i as u64, v, None).unwrap();
        }
        store.add_segment(&w, &fast_write()).unwrap();
        if round % 2 == 1 {
            store.compact(&fast_write()).unwrap();
        }
    }
    stop.store(true, Ordering::Relaxed);
    let total: u64 = readers.into_iter().map(|h| h.join().expect("reader panicked")).sum();
    assert!(total > 0);
    assert_eq!(store.load().len(), 500 + 6 * 400);
}
