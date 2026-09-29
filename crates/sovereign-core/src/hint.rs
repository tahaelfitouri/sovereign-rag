//! Branch-prediction and cache hints that work on **stable** Rust.
//!
//! `core::intrinsics::likely`/`unlikely` are permanently unstable (and `core::hint::likely` is
//! still feature-gated), so we use the same trick as `hashbrown`: route the unexpected side of a
//! branch through a `#[cold]` function. LLVM propagates the cold attribute to the calling basic
//! block, which moves it out of the fall-through path and biases block layout — exactly what the
//! intrinsic would do.

/// Marker function. Calling it tells LLVM "this path is rarely taken".
#[inline]
#[cold]
pub fn cold_path() {}

/// Hint that `b` is almost always `true`.
#[inline(always)]
#[must_use]
pub fn likely(b: bool) -> bool {
    if !b {
        cold_path();
    }
    b
}

/// Hint that `b` is almost always `false`.
#[inline(always)]
#[must_use]
pub fn unlikely(b: bool) -> bool {
    if b {
        cold_path();
    }
    b
}

/// Issue a software prefetch of the cache line containing `ptr` into all cache levels (T0).
///
/// Prefetches are pure hints: they never fault, never dereference in the abstract-machine sense
/// and never change program semantics, so this is a *safe* function even for dangling pointers.
/// On architectures without a stable prefetch intrinsic this compiles to nothing.
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        // SAFETY: SSE is part of the x86_64 baseline, and `prefetcht0` cannot fault on any
        // address (invalid addresses are silently dropped by the hardware).
        unsafe { _mm_prefetch::<_MM_HINT_T0>(ptr.cast::<i8>()) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

/// Prefetch `lines` consecutive 64-byte cache lines starting at `ptr`.
///
/// Useful before touching a vector whose address was just discovered (e.g. an HNSW neighbor):
/// the hardware stream prefetcher needs a couple of misses to lock on, so we prime the first few.
#[inline(always)]
pub fn prefetch_lines<T>(ptr: *const T, lines: usize) {
    let base = ptr.cast::<u8>();
    for i in 0..lines {
        prefetch_read(base.wrapping_add(i * crate::CACHE_LINE));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hints_are_transparent() {
        assert!(likely(true));
        assert!(!likely(false));
        assert!(unlikely(true));
        assert!(!unlikely(false));
    }

    #[test]
    fn prefetch_accepts_any_pointer() {
        let data = [0u8; 256];
        prefetch_lines(data.as_ptr(), 4);
        prefetch_read(core::ptr::null::<u8>());
        prefetch_read(usize::MAX as *const u8);
    }
}
