//! Embedding backends.
//!
//! [`Embedder`] is the seam for real models (ONNX Runtime, candle, llama.cpp, an HTTP API...).
//! The bundled [`HashEmbedder`] is a dependency-free, deterministic *feature-hashing* embedder:
//! lowercase word unigrams and bigrams are hashed into signed buckets ("the hashing trick",
//! Weinberger et al. 2009). It captures lexical overlap, not semantics — but it is fast, works
//! offline, and makes the whole engine demonstrable end-to-end without a GPU or model download.

use std::fmt;

/// Error reported by an embedder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedError(pub String);

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EmbedError {}

/// A batch text-embedding model.
///
/// Implementations must be deterministic for a given [`fingerprint`](Self::fingerprint): the
/// fingerprint is stored in every segment header and checked at query time, so an index can never
/// be queried with vectors from a different model.
pub trait Embedder: Send + Sync + 'static {
    /// Human-readable model name.
    fn name(&self) -> &str;
    /// Output dimensionality.
    fn dim(&self) -> usize;
    /// Stable hash of model identity + configuration.
    fn fingerprint(&self) -> u64;
    /// Embeds `texts` into `out` (row-major, `texts.len() * dim()` floats). CPU-heavy; the
    /// pipeline calls it from a blocking thread.
    ///
    /// # Errors
    /// Model-specific failures.
    fn embed_batch(&self, texts: &[&str], out: &mut [f32]) -> Result<(), EmbedError>;
}

/// 64-bit FNV-1a followed by the MurmurHash3 finalizer (FNV alone has weak low bits).
#[inline]
#[must_use]
pub fn hash_bytes(bytes: &[u8], seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    fmix64(h)
}

#[inline]
fn fmix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "of", "to", "is", "and", "in", "for", "on", "with", "that", "this", "it",
    "as", "be", "are", "by", "or", "at", "from", "was", "were", "has", "have", "its", "into",
];

/// Deterministic feature-hashing embedder. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct HashEmbedder {
    dim: usize,
    seed: u64,
}

impl HashEmbedder {
    /// Default output dimensionality (matches MiniLM-class models).
    pub const DEFAULT_DIM: usize = 384;

    /// Creates an embedder with `dim` buckets.
    ///
    /// # Errors
    /// If `dim < 8`.
    pub fn new(dim: usize) -> Result<Self, EmbedError> {
        if dim < 8 {
            return Err(EmbedError(format!("hash embedder dim={dim} must be >= 8")));
        }
        Ok(Self { dim, seed: 0x0005_EED0_F4A5 })
    }

    #[inline]
    fn bucket(&self, h: u64) -> (usize, f32) {
        // Lemire's fast range reduction on the high bits; sign from the low bit.
        let idx = ((u128::from(h) * self.dim as u128) >> 64) as usize;
        (idx, if h & 1 == 0 { 1.0 } else { -1.0 })
    }

    fn embed_one(&self, text: &str, out: &mut [f32]) {
        out.fill(0.0);
        let bytes = text.as_bytes();
        let mut prev: Option<u64> = None;
        let mut i = 0;
        while i < bytes.len() {
            // Token = maximal run of ASCII alphanumerics, '_' or non-ASCII bytes.
            let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80;
            if !is_word(bytes[i]) {
                i += 1;
                continue;
            }
            let start = i;
            while i < bytes.len() && is_word(bytes[i]) {
                i += 1;
            }
            let tok = &bytes[start..i];
            if tok.len() < 2 && tok[0] < 0x80 {
                continue;
            }
            if STOPWORDS.iter().any(|s| s.as_bytes().eq_ignore_ascii_case(tok)) {
                continue;
            }
            // Case-insensitive hash without allocating a lowercase copy.
            let mut h = 0xcbf2_9ce4_8422_2325u64 ^ self.seed;
            for &b in tok {
                h = (h ^ u64::from(b.to_ascii_lowercase())).wrapping_mul(0x0000_0100_0000_01B3);
            }
            let h = fmix64(h);
            let (idx, sign) = self.bucket(h);
            out[idx] += sign;
            if let Some(p) = prev {
                let (idx, sign) = self.bucket(fmix64(p.rotate_left(17) ^ h));
                out[idx] += 0.5 * sign;
            }
            prev = Some(h);
        }
        // Sublinear term frequency: sign(x)·sqrt(|x|) damps repeated terms.
        let mut any = false;
        for x in out.iter_mut() {
            if *x != 0.0 {
                any = true;
                *x = x.signum() * x.abs().sqrt();
            }
        }
        if !any {
            // Token-free text (e.g. "---"): fall back to one bucket from the raw bytes so the
            // vector is never all-zero (cosine would reject it).
            let (idx, sign) = self.bucket(hash_bytes(bytes, self.seed));
            out[idx] = sign;
        }
        let _ = sovereign_core::kernels().normalize(out);
    }
}

impl Embedder for HashEmbedder {
    fn name(&self) -> &str {
        "hash-embedder-v1"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn fingerprint(&self) -> u64 {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&(self.dim as u64).to_le_bytes());
        buf[8..].copy_from_slice(&self.seed.to_le_bytes());
        hash_bytes(&buf, hash_bytes(self.name().as_bytes(), 0))
    }

    fn embed_batch(&self, texts: &[&str], out: &mut [f32]) -> Result<(), EmbedError> {
        if out.len() != texts.len() * self.dim {
            return Err(EmbedError(format!(
                "output buffer has {} floats, expected {}",
                out.len(),
                texts.len() * self.dim
            )));
        }
        for (text, row) in texts.iter().zip(out.chunks_exact_mut(self.dim)) {
            self.embed_one(text, row);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embed(e: &HashEmbedder, t: &str) -> Vec<f32> {
        let mut v = vec![0.0; e.dim()];
        e.embed_batch(&[t], &mut v).unwrap();
        v
    }

    #[test]
    fn deterministic_normalized_and_lexically_meaningful() {
        let e = HashEmbedder::new(256).unwrap();
        let k = sovereign_core::kernels();
        let a = embed(&e, "Memory-mapped vector index with SIMD search");
        let b = embed(&e, "a SIMD search over a memory mapped vector index");
        let c = embed(&e, "chocolate cake recipe with strawberries");
        assert_eq!(a, embed(&e, "Memory-mapped vector index with SIMD search"));
        assert!((k.norm_sq(&a) - 1.0).abs() < 1e-5);
        assert!(k.dot(&a, &b) > 0.6, "paraphrase similarity {}", k.dot(&a, &b));
        assert!(k.dot(&a, &c).abs() < 0.3, "unrelated similarity {}", k.dot(&a, &c));
        assert_eq!(embed(&e, "SIMD"), embed(&e, "simd"), "case-insensitive");
    }

    #[test]
    fn never_zero_and_validates_buffers() {
        let e = HashEmbedder::new(64).unwrap();
        for t in ["", "---", "a the of", "?!"] {
            let v = embed(&e, t);
            assert!((sovereign_core::kernels().norm_sq(&v) - 1.0).abs() < 1e-5, "{t:?}");
        }
        let mut small = vec![0.0; 10];
        assert!(e.embed_batch(&["x"], &mut small).is_err());
        assert!(HashEmbedder::new(4).is_err());
        assert_ne!(e.fingerprint(), HashEmbedder::new(65).unwrap().fingerprint());
    }
}
