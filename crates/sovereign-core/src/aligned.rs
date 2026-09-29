//! Cache-line aligned storage.
//!
//! # Why 64 bytes?
//!
//! * A cache line is 64 bytes on every x86_64 and on the vast majority of AArch64 cores.
//! * A ZMM register (AVX-512) is exactly 64 bytes, so a 64-byte aligned row means every
//!   full-width load hits exactly one cache line — no split loads, which cost an extra
//!   L1 access (and a full extra line fill when they straddle a page boundary).
//!
//! ```text
//!   row start (64-byte aligned)
//!   v
//!   |<------- cache line 0 ------->|<------- cache line 1 ------->|
//!   [ zmm0: f32 x 16 (64 B)        ][ zmm1: f32 x 16 (64 B)        ] ...
//!
//!   misaligned row (offset 4 B): every zmm load touches TWO lines
//!   |<------- cache line 0 ------->|<------- cache line 1 ------->|
//!       [ zmm0 ------------------------ ][ zmm1 ------------------ ...
//! ```
//!
//! For *false-sharing* avoidance between atomics written by different cores, 64 bytes is not
//! always enough: Intel's L2 "spatial prefetcher" fetches lines in adjacent pairs, so hot
//! counters should be padded to 128 bytes (`crossbeam_utils::CachePadded` does this on
//! x86_64). [`CacheAligned`] is about *data alignment* for SIMD, not contention.

use core::alloc::Layout;
use core::fmt;
use core::mem::size_of;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use std::alloc::{alloc, alloc_zeroed, dealloc, handle_alloc_error, realloc};

use bytemuck::Pod;

use crate::error::{CoreError, Result};

/// Size of one cache line (and of one AVX-512 register) in bytes.
pub const CACHE_LINE: usize = 64;

/// Wrapper forcing its contents onto a cache-line boundary.
///
/// `size_of::<CacheAligned<T>>()` is rounded up to a multiple of 64, so arrays of
/// `CacheAligned<T>` place every element on its own line(s).
#[repr(C, align(64))]
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CacheAligned<T>(pub T);

impl<T> CacheAligned<T> {
    /// Wraps `value`.
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Unwraps the inner value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> Deref for CacheAligned<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for CacheAligned<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for CacheAligned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A growable, contiguous buffer of POD values whose first element is always 64-byte aligned.
///
/// This is `Vec<T>` with a stronger alignment guarantee. It is restricted to [`Pod`] types,
/// which buys three things:
///
/// 1. No destructors to run — `truncate`/`clear`/`drop` are O(1) and panic-free.
/// 2. `alloc_zeroed` produces valid values ([`AlignedVec::zeroed`] is a single `calloc`, which the
///    OS can satisfy with copy-on-write zero pages).
/// 3. Safe byte views (`as_bytes`) for zero-copy serialization.
///
/// # Memory layout
///
/// ```text
///   ptr (addr % 64 == 0)
///   v
///   [ T0 | T1 | ... | T(len-1) | uninit ... | ] <- cap * size_of::<T>() bytes
///   \______________ len ______/
/// ```
///
/// # Invariants
///
/// * If `cap > 0`, `ptr` was returned by the global allocator for `Self::layout(cap)` and
///   `ptr as usize % 64 == 0`.
/// * If `cap == 0`, `ptr` is a dangling, 64-aligned, non-null pointer and nothing is allocated.
/// * `len <= cap`, and elements `[0, len)` are initialized.
pub struct AlignedVec<T: Pod> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

// SAFETY: `AlignedVec<T>` uniquely owns its allocation (like `Vec<T>`), so it is `Send`/`Sync`
// exactly when `T` is.
unsafe impl<T: Pod + Send> Send for AlignedVec<T> {}
// SAFETY: see above; shared access only hands out `&T`.
unsafe impl<T: Pod + Sync> Sync for AlignedVec<T> {}

impl<T: Pod> AlignedVec<T> {
    /// Compile-time guard: ZSTs and over-aligned types are rejected.
    const TYPE_OK: () = {
        assert!(size_of::<T>() != 0, "AlignedVec does not support zero-sized types");
        assert!(core::mem::align_of::<T>() <= CACHE_LINE, "element alignment exceeds 64 bytes");
    };

    #[inline]
    fn dangling() -> NonNull<T> {
        // SAFETY: CACHE_LINE is non-zero, so the pointer is non-null. It carries no provenance
        // and is never dereferenced while `cap == 0`.
        unsafe { NonNull::new_unchecked(core::ptr::without_provenance_mut::<T>(CACHE_LINE)) }
    }

    #[inline]
    fn layout(cap: usize) -> Result<Layout> {
        let bytes = cap.checked_mul(size_of::<T>()).ok_or(CoreError::CapacityOverflow)?;
        Layout::from_size_align(bytes, CACHE_LINE).map_err(|_| CoreError::CapacityOverflow)
    }

    /// Creates an empty buffer without allocating.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        let () = Self::TYPE_OK;
        Self { ptr: Self::dangling(), len: 0, cap: 0 }
    }

    /// Creates an empty buffer with room for at least `cap` elements.
    ///
    /// # Panics
    /// Panics if the byte size overflows `isize::MAX`; aborts on allocation failure.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        let mut v = Self::new();
        v.reserve_exact(cap);
        v
    }

    /// Creates a buffer of `len` zero-initialized elements with a single `alloc_zeroed` call.
    ///
    /// # Panics
    /// Panics on capacity overflow; aborts on allocation failure.
    #[must_use]
    pub fn zeroed(len: usize) -> Self {
        let () = Self::TYPE_OK;
        if len == 0 {
            return Self::new();
        }
        let layout = Self::layout(len).unwrap_or_else(|_| capacity_overflow());
        // SAFETY: `layout` has non-zero size because `len > 0` and `T` is not a ZST.
        let raw = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw.cast::<T>()) else { handle_alloc_error(layout) };
        // All-zero bytes are a valid `T` because `T: Pod` (which implies `Zeroable`).
        Self { ptr, len, cap: len }
    }

    /// Creates a buffer holding a copy of `src`.
    #[must_use]
    pub fn from_slice(src: &[T]) -> Self {
        let mut v = Self::with_capacity(src.len());
        v.extend_from_slice(src);
        v
    }

    /// Number of initialized elements.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the buffer holds no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of elements the buffer can hold without reallocating.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Raw pointer to the first element (64-byte aligned whenever `capacity() > 0`).
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Mutable raw pointer to the first element.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Borrows the initialized prefix as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr` is valid for `len` initialized elements (struct invariant); when
        // `len == 0` a dangling, aligned, non-null pointer is valid for an empty slice.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutably borrows the initialized prefix as a slice.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as in `as_slice`, and `&mut self` guarantees uniqueness.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Views the contents as raw bytes (zero-copy serialization).
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_slice())
    }

    /// Ensures room for `additional` more elements, growing geometrically.
    ///
    /// # Panics
    /// Panics on capacity overflow; aborts on allocation failure.
    pub fn reserve(&mut self, additional: usize) {
        if let Err(e) = self.try_reserve(additional) {
            match e {
                CoreError::CapacityOverflow => capacity_overflow(),
                _ => unreachable!("try_reserve only reports capacity overflow"),
            }
        }
    }

    /// Ensures room for exactly `additional` more elements (no geometric slack).
    ///
    /// # Panics
    /// Panics on capacity overflow; aborts on allocation failure.
    pub fn reserve_exact(&mut self, additional: usize) {
        let needed = self.len.checked_add(additional).unwrap_or_else(|| capacity_overflow());
        if needed > self.cap {
            self.grow_to(needed).unwrap_or_else(|_| capacity_overflow());
        }
    }

    /// Fallible variant of [`reserve`](Self::reserve).
    ///
    /// # Errors
    /// [`CoreError::CapacityOverflow`] if the new byte size cannot be represented.
    /// Allocation failure itself aborts via [`handle_alloc_error`], matching `Vec`.
    pub fn try_reserve(&mut self, additional: usize) -> Result<()> {
        let needed = self.len.checked_add(additional).ok_or(CoreError::CapacityOverflow)?;
        if needed <= self.cap {
            return Ok(());
        }
        // Amortized doubling; start at 4 cache lines worth of elements.
        let min_cap = (4 * CACHE_LINE / size_of::<T>()).max(1);
        let new_cap = needed.max(self.cap.saturating_mul(2)).max(min_cap);
        self.grow_to(new_cap)
    }

    #[cold]
    #[inline(never)]
    fn grow_to(&mut self, new_cap: usize) -> Result<()> {
        debug_assert!(new_cap > self.cap);
        let new_layout = Self::layout(new_cap)?;
        if new_layout.size() > isize::MAX as usize {
            return Err(CoreError::CapacityOverflow);
        }
        let raw = if self.cap == 0 {
            // SAFETY: `new_layout` has non-zero size (`new_cap > 0`, `T` is not a ZST).
            unsafe { alloc(new_layout) }
        } else {
            let old_layout = Self::layout(self.cap)?;
            // SAFETY: `ptr` was allocated with `old_layout` (invariant); the new size is non-zero
            // and does not overflow `isize`. `realloc` preserves the layout's 64-byte alignment.
            unsafe { realloc(self.ptr.as_ptr().cast::<u8>(), old_layout, new_layout.size()) }
        };
        let Some(ptr) = NonNull::new(raw.cast::<T>()) else { handle_alloc_error(new_layout) };
        self.ptr = ptr;
        self.cap = new_cap;
        Ok(())
    }

    /// Appends one element.
    #[inline]
    pub fn push(&mut self, value: T) {
        if crate::hint::unlikely(self.len == self.cap) {
            self.reserve(1);
        }
        // SAFETY: `len < cap` after the reserve, so the slot is inside the allocation.
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
    }

    /// Appends all elements of `src` with a single `memcpy`.
    #[inline]
    pub fn extend_from_slice(&mut self, src: &[T]) {
        self.reserve(src.len());
        // SAFETY: `reserve` guarantees `cap - len >= src.len()`; `src` cannot alias our
        // uninitialized tail because we hold `&mut self`.
        unsafe {
            core::ptr::copy_nonoverlapping(
                src.as_ptr(),
                self.ptr.as_ptr().add(self.len),
                src.len(),
            );
        }
        self.len += src.len();
    }

    /// Resizes to `new_len`, filling new slots with `value`.
    pub fn resize(&mut self, new_len: usize, value: T) {
        if new_len <= self.len {
            self.len = new_len;
            return;
        }
        self.reserve(new_len - self.len);
        for i in self.len..new_len {
            // SAFETY: `i < new_len <= cap`.
            unsafe { self.ptr.as_ptr().add(i).write(value) };
        }
        self.len = new_len;
    }

    /// Shortens the buffer to `len` elements (no-op if already shorter). O(1): `T` has no drop glue.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// Removes all elements, keeping the allocation.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

#[cold]
#[inline(never)]
#[track_caller]
fn capacity_overflow() -> ! {
    panic!("AlignedVec capacity overflow")
}

impl<T: Pod> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.cap != 0 {
            let layout = Self::layout(self.cap).expect("layout was valid at allocation time");
            // SAFETY: `ptr` was allocated by the global allocator with exactly this layout.
            unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), layout) };
        }
    }
}

impl<T: Pod> Default for AlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Pod> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl<T: Pod> Deref for AlignedVec<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Pod> DerefMut for AlignedVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Pod + fmt::Debug> fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedVec")
            .field("len", &self.len)
            .field("cap", &self.cap)
            .field("data", &self.as_slice())
            .finish()
    }
}

impl<T: Pod + PartialEq> PartialEq for AlignedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Pod> FromIterator<T> for AlignedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut v = Self::with_capacity(iter.size_hint().0);
        for x in iter {
            v.push(x);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_aligned<T: Pod>(v: &AlignedVec<T>) {
        if v.capacity() > 0 {
            assert_eq!(v.as_ptr() as usize % CACHE_LINE, 0, "buffer not 64-byte aligned");
        }
    }

    #[test]
    fn cache_aligned_layout() {
        assert_eq!(core::mem::align_of::<CacheAligned<u8>>(), 64);
        assert_eq!(core::mem::size_of::<CacheAligned<u8>>(), 64);
        assert_eq!(core::mem::size_of::<CacheAligned<[u8; 65]>>(), 128);
    }

    #[test]
    fn push_grow_keeps_alignment_and_data() {
        let mut v = AlignedVec::<f32>::new();
        assert!(v.is_empty());
        for i in 0..10_000 {
            v.push(i as f32);
            assert_aligned(&v);
        }
        assert_eq!(v.len(), 10_000);
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as f32));
    }

    #[test]
    fn zeroed_extend_resize_truncate() {
        let mut v = AlignedVec::<u64>::zeroed(33);
        assert_aligned(&v);
        assert!(v.iter().all(|&x| x == 0));
        v.extend_from_slice(&[1, 2, 3]);
        assert_eq!(&v[33..], &[1, 2, 3]);
        v.resize(40, 7);
        assert_eq!(v.len(), 40);
        assert_eq!(v[39], 7);
        v.truncate(2);
        assert_eq!(v.len(), 2);
        v.clear();
        assert!(v.is_empty());
        assert!(v.capacity() >= 40);
    }

    #[test]
    fn clone_eq_and_bytes() {
        let v: AlignedVec<u32> = (0..100u32).collect();
        let w = v.clone();
        assert_eq!(v, w);
        assert_aligned(&w);
        assert_eq!(v.as_bytes().len(), 400);
        assert_eq!(&v.as_bytes()[4..8], &1u32.to_ne_bytes());
    }

    #[test]
    fn try_reserve_overflow_is_reported() {
        let mut v = AlignedVec::<f32>::new();
        v.push(1.0);
        assert_eq!(v.try_reserve(usize::MAX), Err(CoreError::CapacityOverflow));
        assert_eq!(v.try_reserve(usize::MAX / 2), Err(CoreError::CapacityOverflow));
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn empty_slice_is_valid() {
        let v = AlignedVec::<f32>::new();
        assert_eq!(v.as_slice(), &[] as &[f32]);
        assert_eq!(AlignedVec::<f32>::zeroed(0).capacity(), 0);
    }
}
