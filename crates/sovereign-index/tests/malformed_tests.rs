//! Malformed-file and edge-case tests.
//!
//! The corruption tests in `index_tests.rs` flip header bytes, which the header CRC rejects before
//! any structural validation runs. These tests forge headers with a *recomputed, valid* CRC so the
//! structural checks in `Segment::open` and the HNSW graph parser are exercised directly, and they
//! corrupt body regions (not covered by the open-time CRC) to check that search and payload access
//! degrade to errors or wrong-but-safe results instead of panics or UB.

use std::fs;
use std::path::{Path, PathBuf};

use sovereign_core::{Metric, Xoshiro256pp};
use sovereign_index::format::{flags, GraphHeader, SegmentHeader, GRAPH_HEADER_SIZE, HEADER_SIZE};
use sovereign_index::{
    HnswParams, IndexError, IndexStore, SearchMode, SearchParams, Segment, SegmentConfig,
    SegmentWriter, WriteOptions,
};

const DIM: usize = 20;

fn cfg() -> SegmentConfig {
    SegmentConfig { dim: DIM, metric: Metric::Cosine, fingerprint: 7 }
}

fn vectors(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xoshiro256pp::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut v = vec![0.0; DIM];
            rng.fill_gaussian(&mut v);
            v
        })
        .collect()
}

/// Writes a valid segment (graph + payloads) of `n` rows and returns its path.
fn valid_segment(dir: &Path, name: &str, n: usize) -> PathBuf {
    let mut w = SegmentWriter::new(cfg()).unwrap();
    for (i, v) in vectors(n, 1).iter().enumerate() {
        w.push(i as u64, v, Some(&format!("p{i}"))).unwrap();
    }
    let path = dir.join(name);
    let hnsw = (n > 0).then(|| HnswParams { parallel: false, ..HnswParams::default() });
    w.write(&path, &WriteOptions { hnsw, sync: false }).unwrap();
    path
}

fn read_header(path: &Path) -> SegmentHeader {
    bytemuck::pod_read_unaligned(&fs::read(path).unwrap()[..HEADER_SIZE])
}

/// Applies `edit` to the header, recomputes the header CRC and writes it back.
fn forge_header(path: &Path, edit: impl FnOnce(&mut SegmentHeader)) {
    let mut bytes = fs::read(path).unwrap();
    let mut h: SegmentHeader = bytemuck::pod_read_unaligned(&bytes[..HEADER_SIZE]);
    edit(&mut h);
    h.header_crc32 = h.compute_crc();
    bytes[..HEADER_SIZE].copy_from_slice(bytemuck::bytes_of(&h));
    fs::write(path, bytes).unwrap();
}

/// Applies `edit` to the graph header inside the body (no CRC is checked for the body at open).
fn forge_graph_header(path: &Path, edit: impl FnOnce(&mut GraphHeader)) {
    let mut bytes = fs::read(path).unwrap();
    let off = read_header(path).graph.offset as usize;
    let mut g: GraphHeader = bytemuck::pod_read_unaligned(&bytes[off..off + GRAPH_HEADER_SIZE]);
    edit(&mut g);
    bytes[off..off + GRAPH_HEADER_SIZE].copy_from_slice(bytemuck::bytes_of(&g));
    fs::write(path, bytes).unwrap();
}

fn expect_corrupt(path: &Path, what: &str) {
    match Segment::open(path) {
        Err(IndexError::Corrupt { reason, .. }) => {
            assert!(reason.contains(what), "expected `{what}`, got `{reason}`")
        }
        other => panic!("expected Corrupt(`{what}`), got {other:?}"),
    }
}

#[test]
fn forged_headers_with_valid_crc_are_rejected_structurally() {
    let dir = tempfile::tempdir().unwrap();
    type Edit = fn(&mut SegmentHeader);
    let cases: &[(&str, Edit, &str)] = &[
        ("dim0", |h| h.dim = 0, "dimension out of range"),
        ("dimhuge", |h| h.dim = 1 << 20, "dimension out of range"),
        ("stride", |h| h.stride += 16, "stride does not match"),
        ("count", |h| h.count = u64::MAX, "row count exceeds limit"),
        ("flags", |h| h.flags |= 1 << 40, "unknown header flags"),
        ("veclen", |h| h.vectors.len -= 4, "vectors section has wrong size"),
        ("idsoob", |h| h.ids.offset = 1 << 40, "ids section extends past end of file"),
        ("overflow", |h| h.ids.offset = u64::MAX - 1, "ids section overflows"),
        ("misaligned", |h| h.ids.offset += 8, "ids section is not page aligned"),
        ("nopayflag", |h| h.flags &= !flags::HAS_PAYLOAD, "payload sections present without flag"),
        ("nographflag", |h| h.flags &= !flags::HAS_GRAPH, "graph section present without flag"),
        ("countmismatch", |h| h.count -= 1, "wrong size"),
    ];
    for (name, edit, what) in cases {
        let p = valid_segment(dir.path(), &format!("{name}.srag"), 50);
        forge_header(&p, *edit);
        expect_corrupt(&p, what);
    }

    let p = valid_segment(dir.path(), "metric.srag", 50);
    forge_header(&p, |h| h.metric = 9);
    assert!(matches!(Segment::open(&p), Err(IndexError::Core(_))));

    let p = valid_segment(dir.path(), "version.srag", 50);
    forge_header(&p, |h| h.version = 2);
    assert!(matches!(Segment::open(&p), Err(IndexError::UnsupportedVersion { found: 2, .. })));
}

#[test]
fn forged_graph_headers_are_rejected_at_open() {
    let dir = tempfile::tempdir().unwrap();
    type Edit = fn(&mut GraphHeader);
    let cases: &[(&str, Edit, &str)] = &[
        ("magic", |g| g.magic[0] ^= 0xFF, "bad graph magic"),
        ("m", |g| g.m = 1, "m/m0 out of range"),
        ("m0", |g| g.m0 = 5000, "m/m0 out of range"),
        ("level", |g| g.max_level = 200, "max_level out of range"),
        ("nodes", |g| g.node_count += 1, "node count does not match"),
        ("entry", |g| g.entry_point = 1_000_000, "entry point out of range"),
        ("sub", |g| g.layer0.len += 4, "sub-section"),
        ("suboob", |g| g.upper.offset = u64::MAX - 2, "sub-section"),
    ];
    for (name, edit, what) in cases {
        let p = valid_segment(dir.path(), &format!("g{name}.srag"), 200);
        forge_graph_header(&p, *edit);
        expect_corrupt(&p, what);
    }
}

#[test]
fn corrupt_body_degrades_without_panicking() {
    let dir = tempfile::tempdir().unwrap();
    let p = valid_segment(dir.path(), "body.srag", 1500);
    let h = read_header(&p);
    let mut bytes = fs::read(&p).unwrap();

    // Point every layer-0 neighbor id far out of range and break payload offsets.
    let g: GraphHeader = bytemuck::pod_read_unaligned(
        &bytes[h.graph.offset as usize..h.graph.offset as usize + GRAPH_HEADER_SIZE],
    );
    let l0 = (h.graph.offset + g.layer0.offset) as usize;
    let words = g.layer0.len as usize / 4;
    let m0 = g.m0 as usize;
    for i in 0..words {
        if i % (m0 + 1) != 0 {
            bytes[l0 + 4 * i..l0 + 4 * i + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        }
    }
    let po = h.payload_offsets.offset as usize;
    bytes[po + 8..po + 16].copy_from_slice(&u64::MAX.to_le_bytes());
    fs::write(&p, &bytes).unwrap();

    let seg = Segment::open(&p).expect("body corruption is not checked at open");
    assert!(seg.verify().is_err(), "body CRC must catch it");
    let q = vectors(1, 99).remove(0);
    let hits = seg.search(&q, &SearchParams::top(5).mode(SearchMode::Approximate)).unwrap();
    assert!(hits.iter().all(|h| (h.row as usize) < seg.len()), "only valid rows may be returned");
    assert!(matches!(seg.payload(0), Err(IndexError::Corrupt { .. })));
    assert!(matches!(seg.payload(1), Err(IndexError::Corrupt { .. })));
}

#[test]
fn empty_segment_roundtrips_and_searches() {
    let dir = tempfile::tempdir().unwrap();
    let p = valid_segment(dir.path(), "empty.srag", 0);
    let seg = Segment::open(&p).unwrap();
    seg.verify().unwrap();
    assert!(seg.is_empty());
    assert!(!seg.has_graph());
    assert!(seg.vectors().is_empty());
    assert!(seg.ids().is_empty());
    let q = vec![1.0; DIM];
    for mode in [SearchMode::Auto, SearchMode::Exact, SearchMode::Approximate] {
        assert!(seg.search(&q, &SearchParams::top(10).mode(mode)).unwrap().is_empty());
    }
}

#[test]
fn empty_store_searches_and_huge_k_is_clamped() {
    let dir = tempfile::tempdir().unwrap();
    let store = IndexStore::open_or_create(dir.path(), cfg()).unwrap();
    let q = vec![1.0; DIM];
    assert!(store.search(&q, &SearchParams::top(10)).unwrap().hits.is_empty());

    let mut w = store.writer().unwrap();
    for (i, v) in vectors(30, 3).iter().enumerate() {
        w.push(i as u64, v, None).unwrap();
    }
    store.add_segment(&w, &WriteOptions { hnsw: None, sync: false }).unwrap();

    // A caller-controlled `k` must not translate into a giant up-front allocation.
    let huge = SearchParams::top(usize::MAX).mode(SearchMode::Exact);
    assert_eq!(store.search(&q, &huge).unwrap().hits.len(), 30);
    let snap = store.load();
    let seg = &snap.segments()[0].segment;
    assert_eq!(seg.search(&q, &huge).unwrap().len(), 30);
    assert_eq!(seg.search(&q, &SearchParams::top(1 << 40).ef(1 << 40)).unwrap().len(), 30);
}

#[test]
fn recovery_ignores_crash_leftovers_and_reports_lost_segments() {
    let dir = tempfile::tempdir().unwrap();
    let store = IndexStore::open_or_create(dir.path(), cfg()).unwrap();
    let mut w = store.writer().unwrap();
    for (i, v) in vectors(10, 4).iter().enumerate() {
        w.push(i as u64, v, None).unwrap();
    }
    store.add_segment(&w, &WriteOptions { hnsw: None, sync: false }).unwrap();
    drop(store);

    // Simulated crash leftovers: an unpublished segment (written + renamed, manifest not yet
    // updated) and half-written temp files. The manifest is the source of truth: all ignored.
    fs::write(dir.path().join("seg-0000000001.srag"), b"garbage").unwrap();
    fs::write(dir.path().join(".seg-0000000001.srag.tmp-1-2"), b"partial").unwrap();
    fs::write(dir.path().join(".MANIFEST.tmp-1"), b"partial").unwrap();
    let store = IndexStore::open(dir.path(), Default::default()).unwrap();
    assert_eq!(store.load().len(), 10);

    // The next publish reuses id 1 (next-segment was never advanced) and atomically replaces
    // the orphan via rename.
    store.add_segment(&w, &WriteOptions { hnsw: None, sync: false }).unwrap();
    assert_eq!(store.load().len(), 20);
    Segment::open(dir.path().join("seg-0000000001.srag")).unwrap();
    drop(store);

    // A manifest that references a missing segment (e.g. `--no-sync` + power loss) is a hard
    // error with the offending path; there is no automatic repair.
    fs::remove_file(dir.path().join("seg-0000000000.srag")).unwrap();
    match IndexStore::open(dir.path(), Default::default()) {
        Err(IndexError::Io { path, .. }) => assert!(path.ends_with("seg-0000000000.srag")),
        other => panic!("expected Io error for the missing segment, got {other:?}"),
    }
}
