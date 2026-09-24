//! Lock-free pipeline counters.
//!
//! Each counter is written by a different stage on a different core, so each gets its own
//! `CachePadded` slot (128 bytes on x86_64 — see the note on adjacent-line prefetching in
//! `sovereign_core::aligned`). Without padding, eight hot `AtomicU64`s would share one cache line
//! and every increment would bounce it between cores (false sharing).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crossbeam_utils::CachePadded;

macro_rules! counters {
    ($($(#[$doc:meta])* $name:ident),* $(,)?) => {
        /// Live counters updated by the pipeline stages.
        #[derive(Debug)]
        pub struct PipelineMetrics {
            started: Instant,
            $($(#[$doc])* pub $name: CachePadded<AtomicU64>,)*
        }

        impl PipelineMetrics {
            /// Creates zeroed counters; the throughput clock starts now.
            #[must_use]
            pub fn new() -> Self {
                Self { started: Instant::now(), $($name: CachePadded::new(AtomicU64::new(0)),)* }
            }

            /// Consistent-enough point-in-time copy of every counter (each load is `Relaxed`).
            #[must_use]
            pub fn snapshot(&self) -> MetricsSnapshot {
                MetricsSnapshot {
                    elapsed: self.started.elapsed(),
                    $($name: self.$name.load(Ordering::Relaxed),)*
                }
            }
        }

        /// Plain-data copy of [`PipelineMetrics`].
        #[derive(Clone, Copy, Debug, Default)]
        pub struct MetricsSnapshot {
            /// Time since the pipeline started.
            pub elapsed: Duration,
            $($(#[$doc])* pub $name: u64,)*
        }
    };
}

counters! {
    /// Files discovered.
    files_total,
    /// Files read and chunked.
    files_done,
    /// Files skipped (binary, unreadable, too large).
    files_skipped,
    /// Bytes of text read.
    bytes,
    /// Chunks produced by the chunker.
    chunks,
    /// Estimated tokens across all chunks.
    tokens,
    /// Chunks dropped as exact duplicates.
    duplicates,
    /// Vectors embedded.
    embedded,
    /// Vectors staged into the segment writer.
    written,
    /// Total nanoseconds spent inside `Embedder::embed_batch` (summed across workers).
    embed_nanos,
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineMetrics {
    /// Adds `n` to a counter.
    #[inline]
    pub fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Ordering::Relaxed);
    }
}

impl MetricsSnapshot {
    fn rate(&self, n: u64) -> f64 {
        let s = self.elapsed.as_secs_f64();
        if s > 0.0 {
            n as f64 / s
        } else {
            0.0
        }
    }

    /// Estimated tokens per second.
    #[must_use]
    pub fn tokens_per_sec(&self) -> f64 {
        self.rate(self.tokens)
    }

    /// Chunks per second.
    #[must_use]
    pub fn chunks_per_sec(&self) -> f64 {
        self.rate(self.chunks)
    }

    /// Input MiB per second.
    #[must_use]
    pub fn mib_per_sec(&self) -> f64 {
        self.rate(self.bytes) / (1024.0 * 1024.0)
    }

    /// Mean embedding latency per vector.
    #[must_use]
    pub fn embed_us_per_vector(&self) -> f64 {
        if self.embedded == 0 {
            0.0
        } else {
            self.embed_nanos as f64 / 1e3 / self.embedded as f64
        }
    }
}
