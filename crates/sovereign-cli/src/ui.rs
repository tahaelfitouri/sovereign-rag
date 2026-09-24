//! Terminal formatting helpers and process introspection.

use std::time::Duration;

use console::style;
use indicatif::{ProgressBar, ProgressStyle};

/// Formats a byte count with binary units.
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// Formats a duration with an adaptive unit (ns/µs/ms/s).
pub fn duration(d: Duration) -> String {
    let ns = d.as_nanos() as f64;
    if ns < 1e3 {
        format!("{ns:.0} ns")
    } else if ns < 1e6 {
        format!("{:.1} µs", ns / 1e3)
    } else if ns < 1e9 {
        format!("{:.2} ms", ns / 1e6)
    } else {
        format!("{:.2} s", ns / 1e9)
    }
}

/// Formats a large count with thousands separators.
pub fn count(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Resident set size of this process (Linux `/proc/self/status`), if available.
pub fn rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

/// CPU model string (Linux `/proc/cpuinfo`), if available.
pub fn cpu_model() -> Option<String> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    info.lines()
        .find(|l| l.starts_with("model name") || l.starts_with("Model"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_owned())
}

/// Percentile (nearest-rank) of an ascending-sorted slice.
pub fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Section header.
pub fn header(title: &str) {
    println!("\n{}", style(title).bold().cyan());
}

/// Aligned `key  value` line.
pub fn kv(key: &str, value: impl std::fmt::Display) {
    println!("  {:<22} {}", style(key).dim(), value);
}

/// Spinner with a steady tick (runs on indicatif's own thread, so it animates during blocking work).
pub fn spinner(msg: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg} {elapsed:.dim}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    pb.set_message(msg.into());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

/// Share of SIMD lanes doing useful work: rows are zero-padded to 16 lanes (64 B), so kernels
/// process `padded_stride(dim)` elements of which `dim` are real.
pub fn lane_efficiency(dim: usize) -> f64 {
    match sovereign_core::padded_stride(dim) {
        Ok(stride) if stride > 0 => dim as f64 / stride as f64,
        _ => 0.0,
    }
}
