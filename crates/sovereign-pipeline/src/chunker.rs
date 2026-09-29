//! Structure-aware, zero-allocation text chunking.
//!
//! [`Chunker`] is an iterator over [`Chunk`]s that **borrow** from the source string: chunk text is
//! a `&str` slice, never a copy, and the iterator itself holds only a few `usize`s of state. Lines
//! are located with `memchr` (SIMD-accelerated), and no regex engine is involved.
//!
//! # Strategy
//!
//! The source is split into *blocks* — Markdown headings, fenced code blocks, and paragraphs
//! (runs of non-blank lines) — and blocks are packed greedily into chunks of about
//! `target_bytes`:
//!
//! * a heading always starts a new chunk once the current one has body text, and becomes the
//!   chunk's `heading` context (so a chunk deep inside a section still knows which section);
//! * fenced code blocks are atomic unless they exceed `max_bytes`;
//! * an oversized block is split at the best boundary within budget: newline, then sentence end,
//!   then whitespace, then a UTF-8 character boundary (never inside a code point).
//!
//! For source code, blank-line-separated blocks approximate top-level items; this keeps functions
//! intact in the common case. A tree-sitter chunker can be slotted in behind the same `Chunk`
//! type for exact AST boundaries.

use std::path::Path;

/// What kind of document is being chunked (selects the block grammar).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceKind {
    /// CommonMark-ish: ATX headings and fenced code blocks are recognized.
    Markdown,
    /// Source code: blank-line separated blocks.
    Code,
    /// Plain prose: blank-line separated paragraphs.
    Text,
}

impl SourceKind {
    /// Classifies a file by extension; `None` for unsupported files.
    #[must_use]
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        Some(match ext.as_str() {
            "md" | "markdown" | "mdx" => Self::Markdown,
            "txt" | "rst" | "adoc" | "org" | "tex" => Self::Text,
            "rs" | "py" | "js" | "mjs" | "ts" | "tsx" | "jsx" | "go" | "c" | "h" | "cc" | "cpp"
            | "cxx" | "hpp" | "hh" | "cu" | "cuh" | "java" | "kt" | "swift" | "rb" | "sh"
            | "bash" | "zsh" | "toml" | "yaml" | "yml" | "sql" | "zig" | "cs" | "scala" | "lua"
            | "hs" | "ml" | "php" | "proto" | "cmake" => Self::Code,
            _ => return None,
        })
    }
}

/// Chunk size limits in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkerConfig {
    /// Preferred chunk size; blocks are packed until the next one would exceed it.
    pub target_bytes: usize,
    /// Hard limit; larger blocks are split.
    pub max_bytes: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self { target_bytes: 1200, max_bytes: 2400 }
    }
}

impl ChunkerConfig {
    /// Validates the limits.
    ///
    /// # Errors
    /// A message describing the invalid combination.
    pub fn validate(&self) -> Result<(), String> {
        if self.target_bytes < 64 {
            return Err(format!("target_bytes={} must be >= 64", self.target_bytes));
        }
        if self.max_bytes < self.target_bytes {
            return Err(format!(
                "max_bytes={} must be >= target_bytes={}",
                self.max_bytes, self.target_bytes
            ));
        }
        Ok(())
    }
}

/// Dominant content of a chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChunkKind {
    /// Prose / headings.
    Prose,
    /// Code (fenced block or source file).
    Code,
    /// Both prose and code.
    Mixed,
}

/// A chunk borrowed from its source document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk<'a> {
    /// Chunk text (a slice of the source, trailing whitespace trimmed).
    pub text: &'a str,
    /// Byte offset of `text` in the source.
    pub start: usize,
    /// Exclusive end offset of `text` in the source.
    pub end: usize,
    /// Nearest enclosing Markdown heading, if any.
    pub heading: Option<&'a str>,
    /// Dominant content kind.
    pub kind: ChunkKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKind<'a> {
    Blank,
    Heading(&'a str),
    Fence,
    Para,
}

#[derive(Clone, Copy, Debug)]
struct Block<'a> {
    start: usize,
    end: usize,
    kind: BlockKind<'a>,
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    start: usize,
    end: usize,
    kind: ChunkKind,
}

/// Iterator over [`Chunk`]s of a document. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct Chunker<'a> {
    src: &'a str,
    kind: SourceKind,
    cfg: ChunkerConfig,
    pos: usize,
    heading: Option<&'a str>,
    pending: Option<Pending>,
}

impl<'a> Chunker<'a> {
    /// Creates a chunker over `src`.
    #[must_use]
    pub fn new(src: &'a str, kind: SourceKind, cfg: ChunkerConfig) -> Self {
        Self { src, kind, cfg, pos: 0, heading: None, pending: None }
    }

    /// Returns `(end of line content, start of next line)` for the line starting at `at`.
    #[inline]
    fn line(&self, at: usize) -> (usize, usize) {
        match memchr::memchr(b'\n', &self.src.as_bytes()[at..]) {
            Some(i) => (at + i, at + i + 1),
            None => (self.src.len(), self.src.len()),
        }
    }

    fn block_kind(&self, b: &Block<'_>) -> ChunkKind {
        match (b.kind, self.kind) {
            (BlockKind::Fence, _) | (_, SourceKind::Code) => ChunkKind::Code,
            _ => ChunkKind::Prose,
        }
    }

    fn scan_block(&self, at: usize) -> Block<'a> {
        let src = self.src;
        let (ce, next) = self.line(at);
        let line = &src[at..ce];
        if is_blank(line) {
            return Block { start: at, end: next, kind: BlockKind::Blank };
        }
        let md = self.kind == SourceKind::Markdown;
        if md {
            if let Some(h) = heading_text(line) {
                return Block { start: at, end: next, kind: BlockKind::Heading(h) };
            }
            if let Some((ch, n)) = fence_open(line) {
                let mut pos = next;
                while pos < src.len() {
                    let (ce2, nx2) = self.line(pos);
                    if is_fence_close(&src[pos..ce2], ch, n) {
                        return Block { start: at, end: nx2, kind: BlockKind::Fence };
                    }
                    pos = nx2;
                }
                return Block { start: at, end: src.len(), kind: BlockKind::Fence };
            }
        }
        let mut end = next;
        while end < src.len() {
            let (ce2, nx2) = self.line(end);
            let l = &src[end..ce2];
            if is_blank(l) || (md && (heading_text(l).is_some() || fence_open(l).is_some())) {
                break;
            }
            end = nx2;
        }
        Block { start: at, end, kind: BlockKind::Para }
    }

    fn make(
        &self,
        start: usize,
        end: usize,
        heading: Option<&'a str>,
        kind: ChunkKind,
    ) -> Chunk<'a> {
        let text = self.src[start..end].trim_end();
        Chunk { text, start, end: start + text.len(), heading, kind }
    }

    fn emit_pending(&mut self, p: Pending) -> Chunk<'a> {
        let cut = p.start + split_point(&self.src[p.start..p.end], self.cfg.max_bytes);
        if cut < p.end {
            self.pending = Some(Pending { start: cut, ..p });
        }
        self.make(p.start, cut, self.heading, p.kind)
    }

    fn next_chunk(&mut self) -> Option<Chunk<'a>> {
        if let Some(p) = self.pending.take() {
            return Some(self.emit_pending(p));
        }
        let len = self.src.len();
        // Skip leading blank lines.
        loop {
            if self.pos >= len {
                return None;
            }
            let b = self.scan_block(self.pos);
            if b.kind == BlockKind::Blank {
                self.pos = b.end;
            } else {
                break;
            }
        }

        let start = self.pos;
        let mut content_end = start;
        let mut cursor = start;
        let mut has_body = false;
        let (mut prose, mut code) = (false, false);
        let mut heading = self.heading;

        while cursor < len {
            let b = self.scan_block(cursor);
            match b.kind {
                BlockKind::Blank => cursor = b.end,
                BlockKind::Heading(h) => {
                    if has_body {
                        break; // new section: flush what we have
                    }
                    self.heading = Some(h);
                    heading = Some(h);
                    prose = true;
                    content_end = b.end;
                    cursor = b.end;
                }
                BlockKind::Fence | BlockKind::Para => {
                    let bk = self.block_kind(&b);
                    let block_len = b.end - b.start;
                    if has_body && (content_end - start) + block_len > self.cfg.target_bytes {
                        break;
                    }
                    if block_len > self.cfg.max_bytes {
                        if has_body {
                            break;
                        }
                        // Oversized block at the start of a chunk: emit the first piece (with any
                        // headings accumulated so far) and queue the remainder.
                        let prefix = b.start - start;
                        let budget =
                            self.cfg.max_bytes.saturating_sub(prefix).max(self.cfg.max_bytes / 2);
                        let cut = b.start + split_point(&self.src[b.start..b.end], budget);
                        if cut < b.end {
                            self.pending = Some(Pending { start: cut, end: b.end, kind: bk });
                        }
                        self.pos = b.end;
                        let kind =
                            if prose && bk == ChunkKind::Code { ChunkKind::Mixed } else { bk };
                        return Some(self.make(start, cut, heading, kind));
                    }
                    match bk {
                        ChunkKind::Code => code = true,
                        _ => prose = true,
                    }
                    has_body = true;
                    content_end = b.end;
                    cursor = b.end;
                }
            }
        }
        self.pos = content_end.max(start);
        if content_end == start {
            self.pos = cursor; // only blank lines remained
            return None;
        }
        let kind = match (prose, code) {
            (true, true) => ChunkKind::Mixed,
            (false, true) => ChunkKind::Code,
            _ => ChunkKind::Prose,
        };
        Some(self.make(start, content_end, heading, kind))
    }
}

impl<'a> Iterator for Chunker<'a> {
    type Item = Chunk<'a>;

    fn next(&mut self) -> Option<Chunk<'a>> {
        loop {
            let c = self.next_chunk()?;
            if !c.text.is_empty() {
                return Some(c);
            }
        }
    }
}

#[inline]
fn is_blank(line: &str) -> bool {
    line.bytes().all(|b| b.is_ascii_whitespace())
}

/// Strips up to three leading spaces (CommonMark indentation allowance).
#[inline]
fn strip_indent(line: &str) -> &str {
    let spaces = line.bytes().take(3).take_while(|&b| b == b' ').count();
    &line[spaces..]
}

/// ATX heading text (`## Title ##` → `Title`).
fn heading_text(line: &str) -> Option<&str> {
    let s = strip_indent(line);
    let hashes = s.bytes().take_while(|&b| b == b'#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &s[hashes..];
    if !rest.is_empty() && !rest.starts_with([' ', '\t']) {
        return None; // "#hashtag" is not a heading
    }
    let text = rest.trim().trim_end_matches('#').trim_end();
    Some(text)
}

/// Opening code fence: returns the fence character and its run length.
fn fence_open(line: &str) -> Option<(u8, usize)> {
    let s = strip_indent(line).as_bytes();
    let ch = *s.first()?;
    if ch != b'`' && ch != b'~' {
        return None;
    }
    let n = s.iter().take_while(|&&b| b == ch).count();
    if n < 3 || (ch == b'`' && s[n..].contains(&b'`')) {
        return None;
    }
    Some((ch, n))
}

fn is_fence_close(line: &str, ch: u8, n: usize) -> bool {
    let s = strip_indent(line).as_bytes();
    let run = s.iter().take_while(|&&b| b == ch).count();
    run >= n && s[run..].iter().all(|b| b.is_ascii_whitespace())
}

/// Best split offset in `text` (`0 < result <= text.len()`, on a char boundary) within `budget`.
#[must_use]
pub fn split_point(text: &str, budget: usize) -> usize {
    if text.len() <= budget {
        return text.len();
    }
    let mut window = budget.min(text.len());
    while window > 0 && !text.is_char_boundary(window) {
        window -= 1;
    }
    if window == 0 {
        // Budget smaller than the first character: take exactly one character (progress).
        return text.chars().next().map_or(text.len(), char::len_utf8);
    }
    let bytes = &text.as_bytes()[..window];
    let floor = window / 2;
    if let Some(i) = memchr::memrchr(b'\n', bytes).filter(|&i| i + 1 > floor) {
        return i + 1;
    }
    if let Some(i) = memchr::memmem::rfind(bytes, b". ").filter(|&i| i + 2 > floor) {
        return i + 2;
    }
    if let Some(i) = bytes.iter().rposition(|&b| b == b' ' || b == b'\t').filter(|&i| i + 1 > floor)
    {
        return i + 1;
    }
    window
}

/// Rough BPE token estimate without allocating: one token per word start, one per additional
/// six word characters, and one per punctuation mark. Within ~20% of GPT-style tokenizers on
/// English prose and code — good enough for throughput metrics, not for billing.
#[must_use]
pub fn estimate_tokens(s: &str) -> usize {
    let (mut tokens, mut run) = (0usize, 0usize);
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80 {
            if run % 6 == 0 {
                tokens += 1;
            }
            run += 1;
        } else {
            run = 0;
            if !b.is_ascii_whitespace() {
                tokens += 1;
            }
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(src: &str, kind: SourceKind, target: usize, max: usize) -> Vec<Chunk<'_>> {
        Chunker::new(src, kind, ChunkerConfig { target_bytes: target, max_bytes: max }).collect()
    }

    fn assert_zero_copy(src: &str, cs: &[Chunk<'_>]) {
        for c in cs {
            assert_eq!(&src[c.start..c.end], c.text, "chunk must be a slice of the source");
            let base = src.as_ptr() as usize;
            assert_eq!(c.text.as_ptr() as usize - base, c.start, "chunk must borrow, not copy");
        }
    }

    #[test]
    fn markdown_sections_and_headings() {
        let src = "# Intro\n\nHello world.\n\n## Install\n\nRun `cargo build`.\n\nThen test.\n\n## Usage\n\nCall it.\n";
        let cs = chunks(src, SourceKind::Markdown, 64, 256);
        assert_zero_copy(src, &cs);
        assert_eq!(cs.len(), 3);
        assert_eq!(cs[0].heading, Some("Intro"));
        assert!(cs[0].text.starts_with("# Intro"));
        assert_eq!(cs[1].heading, Some("Install"));
        assert!(cs[1].text.contains("Then test."));
        assert_eq!(cs[2].heading, Some("Usage"));
    }

    #[test]
    fn heading_only_runs_merge_into_body() {
        let src = "# A\n## B\n\nbody text\n";
        let cs = chunks(src, SourceKind::Markdown, 64, 256);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].heading, Some("B"));
        assert_eq!(cs[0].text, "# A\n## B\n\nbody text");
    }

    #[test]
    fn fences_are_atomic_and_hide_fake_headings() {
        let src =
            "Intro para.\n\n```python\n# not a heading\n\ndef f():\n    return 1\n```\n\nAfter.\n";
        let cs = chunks(src, SourceKind::Markdown, 16, 1024);
        assert_zero_copy(src, &cs);
        let fence = cs.iter().find(|c| c.text.starts_with("```")).expect("fence chunk");
        assert!(fence.text.ends_with("```"));
        assert!(fence.text.contains("# not a heading"));
        assert_eq!(fence.kind, ChunkKind::Code);
        assert!(cs.iter().all(|c| c.heading.is_none()));
    }

    #[test]
    fn packing_respects_target_and_max() {
        let para = "lorem ipsum dolor sit amet. ".repeat(4);
        let src = (0..40).map(|_| para.trim_end()).collect::<Vec<_>>().join("\n\n");
        let cs = chunks(&src, SourceKind::Text, 400, 800);
        assert_zero_copy(&src, &cs);
        assert!(cs.len() > 5);
        assert!(cs.iter().all(|c| c.text.len() <= 800));
        let covered: usize = cs.iter().map(|c| c.text.len()).sum();
        assert!(covered > src.len() * 9 / 10, "chunks must cover the text");
    }

    #[test]
    fn oversized_blocks_split_on_char_boundaries() {
        let src = "ü".repeat(3000) + "\n\nshort tail";
        let cs = chunks(&src, SourceKind::Text, 100, 500);
        assert_zero_copy(&src, &cs);
        assert!(cs.iter().all(|c| c.text.len() <= 500 && !c.text.is_empty()));
        let total: usize = cs.iter().map(|c| c.text.len()).sum();
        assert_eq!(total, 6000 + "short tail".len());
    }

    #[test]
    fn code_splits_at_blank_lines() {
        let src = "fn a() {\n    1\n}\n\nfn b() {\n    2\n}\n\nfn c() {\n    3\n}\n";
        let cs = chunks(src, SourceKind::Code, 20, 200);
        assert_eq!(cs.len(), 3);
        assert!(cs.iter().all(|c| c.kind == ChunkKind::Code && c.text.starts_with("fn ")));
        // In code, '#' lines are not headings.
        let py = "# comment\nx = 1\n";
        assert_eq!(chunks(py, SourceKind::Code, 64, 128)[0].heading, None);
    }

    #[test]
    fn degenerate_inputs() {
        assert!(chunks("", SourceKind::Markdown, 64, 128).is_empty());
        assert!(chunks("\n\n  \n\t\n", SourceKind::Markdown, 64, 128).is_empty());
        let one = chunks("no newline at end", SourceKind::Text, 64, 128);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].text, "no newline at end");
        assert_eq!(
            chunks("```\nunterminated", SourceKind::Markdown, 64, 128)[0].kind,
            ChunkKind::Code
        );
    }

    #[test]
    fn helpers() {
        assert_eq!(heading_text("### Title ###"), Some("Title"));
        assert_eq!(heading_text("#hashtag"), None);
        assert_eq!(heading_text("####### seven"), None);
        assert_eq!(heading_text("#"), Some(""));
        assert_eq!(fence_open("```rust"), Some((b'`', 3)));
        assert_eq!(fence_open("~~~~"), Some((b'~', 4)));
        assert_eq!(fence_open("`` no"), None);
        assert!(is_fence_close("````", b'`', 3));
        assert!(!is_fence_close("```x", b'`', 3));
        assert_eq!(split_point("hello world foo", 13), 12);
        assert_eq!(split_point("line one\nline two", 12), 9);
        assert_eq!(split_point("日本語", 2), 3, "must make progress by one whole char");
        assert_eq!(estimate_tokens("Hello, world!"), 4);
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(SourceKind::from_path(Path::new("a/b.RS")), Some(SourceKind::Code));
        assert_eq!(SourceKind::from_path(Path::new("x.png")), None);
    }
}
