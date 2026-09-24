//! Similarity metrics.
//!
//! Internally every search maximizes a **score** (higher = more similar). HNSW works with a
//! **distance** (lower = closer), defined as `distance = -score`, so the two views never disagree:
//!
//! | metric   | score                  | stored vectors      |
//! |----------|------------------------|---------------------|
//! | `Cosine` | `x̂ · q̂`                | L2-normalized       |
//! | `Dot`    | `x · q`                | as-is               |
//! | `L2`     | `-‖x − q‖²`            | as-is               |
//!
//! Cosine is implemented by normalizing once at ingest and once per query, which turns the
//! three-accumulator cosine kernel into a single-accumulator dot product on the hot path.

use core::fmt;
use core::str::FromStr;

use crate::error::CoreError;

/// A similarity metric. The discriminant is the on-disk tag.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Metric {
    /// Cosine similarity (vectors are normalized at ingest; scored by dot product).
    #[default]
    Cosine = 0,
    /// Raw inner product (maximum inner-product search).
    Dot = 1,
    /// Squared Euclidean distance (score is its negation).
    L2 = 2,
}

impl Metric {
    /// Decodes an on-disk tag.
    ///
    /// # Errors
    /// [`CoreError::InvalidMetric`] for unknown tags.
    pub fn from_tag(tag: u32) -> Result<Self, CoreError> {
        match tag {
            0 => Ok(Self::Cosine),
            1 => Ok(Self::Dot),
            2 => Ok(Self::L2),
            other => Err(CoreError::InvalidMetric(other.to_string())),
        }
    }

    /// On-disk tag.
    #[must_use]
    pub const fn tag(self) -> u32 {
        self as u32
    }

    /// Canonical lowercase name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::Dot => "dot",
            Self::L2 => "l2",
        }
    }

    /// Whether stored vectors and queries are L2-normalized for this metric.
    #[must_use]
    pub const fn normalizes(self) -> bool {
        matches!(self, Self::Cosine)
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Metric {
    type Err = CoreError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cosine" | "cos" => Ok(Self::Cosine),
            "dot" | "ip" | "inner" => Ok(Self::Dot),
            "l2" | "euclidean" => Ok(Self::L2),
            _ => Err(CoreError::InvalidMetric(s.to_owned())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for m in [Metric::Cosine, Metric::Dot, Metric::L2] {
            assert_eq!(Metric::from_tag(m.tag()).unwrap(), m);
            assert_eq!(m.as_str().parse::<Metric>().unwrap(), m);
        }
        assert!(Metric::from_tag(9).is_err());
        assert!("manhattan".parse::<Metric>().is_err());
    }
}
