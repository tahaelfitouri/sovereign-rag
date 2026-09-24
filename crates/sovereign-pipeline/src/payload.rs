//! Compact payload encoding stored next to each vector.
//!
//! ```text
//!   <path> \t <start> \t <end> \t <heading> \n <chunk text ...>
//! ```
//!
//! The header line is tab-separated with tabs/newlines in `path`/`heading` replaced by spaces, so
//! decoding is a `memchr` for the first newline plus three splits — and returns borrowed `&str`s
//! straight from the memory-mapped payload section (zero-copy).

/// Borrowed view of a decoded payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadRef<'a> {
    /// Source file path.
    pub path: &'a str,
    /// Byte offset of the chunk in the source file.
    pub start: usize,
    /// Exclusive end offset.
    pub end: usize,
    /// Enclosing heading, if any.
    pub heading: Option<&'a str>,
    /// Chunk text.
    pub text: &'a str,
}

fn push_sanitized(out: &mut String, s: &str) {
    for c in s.chars() {
        out.push(if c == '\t' || c == '\n' || c == '\r' { ' ' } else { c });
    }
}

/// Appends the encoded payload to `out` (callers reuse one `String` across rows).
pub fn encode(
    out: &mut String,
    path: &str,
    start: usize,
    end: usize,
    heading: Option<&str>,
    text: &str,
) {
    use std::fmt::Write as _;
    push_sanitized(out, path);
    // Writing to a String is infallible.
    let _ = write!(out, "\t{start}\t{end}\t");
    push_sanitized(out, heading.unwrap_or(""));
    out.push('\n');
    out.push_str(text);
}

/// Decodes a payload produced by [`encode`]; `None` if it is not in that format.
#[must_use]
pub fn decode(s: &str) -> Option<PayloadRef<'_>> {
    let nl = memchr::memchr(b'\n', s.as_bytes())?;
    let (head, text) = (&s[..nl], &s[nl + 1..]);
    let mut it = head.splitn(4, '\t');
    let path = it.next()?;
    let start = it.next()?.parse().ok()?;
    let end = it.next()?.parse().ok()?;
    let heading = it.next()?;
    Some(PayloadRef { path, start, end, heading: (!heading.is_empty()).then_some(heading), text })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_sanitization() {
        let mut s = String::new();
        encode(&mut s, "docs/a\tb.md", 10, 42, Some("Intro\nPart"), "line1\nline2\tx");
        let p = decode(&s).unwrap();
        assert_eq!(p.path, "docs/a b.md");
        assert_eq!((p.start, p.end), (10, 42));
        assert_eq!(p.heading, Some("Intro Part"));
        assert_eq!(p.text, "line1\nline2\tx");

        s.clear();
        encode(&mut s, "x.rs", 0, 1, None, "");
        assert_eq!(decode(&s).unwrap().heading, None);
        assert!(decode("no header").is_none());
        assert!(decode("a\tb\tc\td\nx").is_none());
    }
}
