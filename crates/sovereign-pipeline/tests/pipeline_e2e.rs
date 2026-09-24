//! End-to-end: files on disk → pipeline → segment store → query → decoded payload.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use sovereign_core::Metric;
use sovereign_index::{IndexStore, SearchParams, SegmentConfig, WriteOptions};
use sovereign_pipeline::{
    payload, Embedder, HashEmbedder, IngestConfig, IngestPipeline, PipelineError,
};

fn corpus(dir: &std::path::Path) {
    fs::create_dir_all(dir.join("docs/nested")).unwrap();
    fs::create_dir_all(dir.join("target")).unwrap();
    fs::create_dir_all(dir.join(".git")).unwrap();
    fs::write(
        dir.join("docs/hnsw.md"),
        "# HNSW\n\nHierarchical navigable small world graphs give logarithmic search.\n\n\
         ## Parameters\n\nThe parameter m controls graph degree and ef controls the beam width.\n",
    )
    .unwrap();
    fs::write(
        dir.join("docs/nested/mmap.md"),
        "# Memory mapping\n\nSegments are memory mapped read-only; the page cache serves vectors \
         with zero copies.\n\n```rust\nlet mmap = unsafe { Mmap::map(&file)? };\n```\n",
    )
    .unwrap();
    fs::write(
        dir.join("docs/cooking.txt"),
        "Whisk eggs and sugar, then fold in flour for a light sponge cake.\n",
    )
    .unwrap();
    fs::write(
        dir.join("docs/cooking-copy.txt"),
        "Whisk eggs and sugar, then fold in flour for a light sponge cake.\n",
    )
    .unwrap();
    fs::write(dir.join("src.rs"), "fn simd_dot(a: &[f32], b: &[f32]) -> f32 {\n    a.iter().zip(b).map(|(x, y)| x * y).sum()\n}\n").unwrap();
    fs::write(dir.join("binary.md"), [0xFFu8, 0xFE, 0x00, 0x41]).unwrap();
    fs::write(dir.join("image.png"), [0u8; 16]).unwrap();
    fs::write(dir.join("target/ignored.md"), "# should never be indexed\n").unwrap();
    fs::write(dir.join(".git/ignored.md"), "# should never be indexed\n").unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingest_publish_and_query() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("corpus");
    corpus(&src);

    let embedder = Arc::new(HashEmbedder::new(256).unwrap());
    let config =
        SegmentConfig { dim: 256, metric: Metric::Cosine, fingerprint: embedder.fingerprint() };
    let store = IndexStore::open_or_create(tmp.path().join("index"), config).unwrap();

    let cfg = IngestConfig {
        batch_size: 2,
        embed_workers: 3,
        queue_depth: 1,
        doc_queue: 1,
        ..Default::default()
    };
    let pipeline = IngestPipeline::new(embedder.clone(), cfg).unwrap();
    let (writer, report) =
        pipeline.run(std::slice::from_ref(&src), store.writer().unwrap()).await.unwrap();

    assert_eq!(report.files_total, 6, "5 text files + 1 binary .md; target/.git/png excluded");
    assert_eq!(report.files_skipped, 1, "binary file skipped");
    assert_eq!(report.files_indexed, 5);
    assert_eq!(report.duplicates, 1, "the copied cooking file is deduplicated");
    assert_eq!(report.vectors, report.chunks - report.duplicates);
    assert!(report.tokens > 0 && report.bytes > 0);

    let written = writer.len();
    store.add_segment(&writer, &WriteOptions { hnsw: None, sync: false }).unwrap();
    assert_eq!(store.load().len(), written);

    let ask = |q: &str| {
        let mut v = vec![0.0; 256];
        embedder.embed_batch(&[q], &mut v).unwrap();
        let res = store.search(&v, &SearchParams::top(3)).unwrap();
        let top = res.hits[0];
        let raw = res.payload(&top).unwrap().expect("payloads stored").to_owned();
        raw
    };

    let p = ask("what does the ef parameter control in the graph?");
    let d = payload::decode(&p).unwrap();
    assert!(d.path.ends_with("hnsw.md"), "got {}", d.path);
    assert_eq!(d.heading, Some("Parameters"));
    assert!(d.text.contains("beam width"));

    let p = ask("memory mapped page cache zero copies");
    assert!(payload::decode(&p).unwrap().path.ends_with("mmap.md"));

    let p = ask("sponge cake with eggs and flour");
    assert!(payload::decode(&p).unwrap().text.contains("Whisk eggs"));

    let p = ask("simd_dot function");
    assert!(payload::decode(&p).unwrap().path.ends_with("src.rs"));
}

#[tokio::test]
async fn mismatched_embedder_and_missing_root_are_errors() {
    let embedder = Arc::new(HashEmbedder::new(64).unwrap());
    let pipeline = IngestPipeline::new(embedder.clone(), IngestConfig::default()).unwrap();

    let wrong =
        SegmentConfig { dim: 32, metric: Metric::Cosine, fingerprint: embedder.fingerprint() };
    let w = sovereign_index::SegmentWriter::new(wrong).unwrap();
    assert!(matches!(pipeline.run(&[PathBuf::from(".")], w).await, Err(PipelineError::Config(_))));

    let ok = SegmentConfig { dim: 64, metric: Metric::Cosine, fingerprint: embedder.fingerprint() };
    let w = sovereign_index::SegmentWriter::new(ok).unwrap();
    let missing = PathBuf::from("/definitely/not/here");
    assert!(matches!(pipeline.run(&[missing], w).await, Err(PipelineError::Io { .. })));

    let bad = IngestConfig { batch_size: 0, ..Default::default() };
    assert!(IngestPipeline::new(embedder, bad).is_err());
}
