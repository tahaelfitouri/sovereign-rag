//! A single-threaded bump ("region") allocator.
//!
//! Allocation is a pointer bump plus an alignment round-up — typically 2–3 instructions on the
//! fast path — and freeing is wholesale: [`Arena::reset`] rewinds the cursor in O(1) while
//! keeping the largest chunk for reuse. This is the classic pattern for *per-batch* scratch
//! memory: the ingestion pipeline builds contextualized chunk texts for a whole embedding batch in
//! an arena, hands `&str`s to the embedder, then resets — zero `malloc`/`free` pairs per chunk.
//!
//! # Memory layout
//!
//! ```text
//!   chunks: [ chunk 0 (full) ][ chunk 1 (full) ][ chunk 2 (current)                ]
//!                                                ^base          ^base+offset    ^base+cap
//!                                                [ a | pad | b  | free ........... ]
//! ```
//!
//! Every chunk is allocated with 64-byte alignment. Chunk sizes double (up to a cap) so the
//! number of chunks stays logarithmic in the total allocated bytes.
//!
//! # Safety model
//!
//! * Allocations return `&mut T` borrowed from `&self`. This is sound because every allocation is
//!   a fresh, disjoint region, and chunks are never freed or moved while `&self` borrows exist:
//!   the only operations that free memory ([`Arena::reset`], `Drop`) take `&mut self` / `self`.
//! * Only `T: Copy` values may be stored, so the arena never needs to run destructors.
//! * The arena uses `Cell`, so it is `!Sync`: one arena per thread / task.

use core::alloc::Layout;
use core::cell::{Cell, RefCell};
use core::fmt;
use core::ptr::NonNull;
use std::alloc::{alloc, dealloc, handle_alloc_error};

use crate::aligned::CACHE_LINE;

/// Default size of the first chunk.
pub const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;
/// Chunks stop doubling beyond this size (oversized requests still get a dedicated chunk).
const MAX_CHUNK_SIZE: usize = 16 * 1024 * 1024;

struct Chunk {
    ptr: NonNull<u8>,
    layout: Layout,
}

/// Bump allocator for `Copy` data. See the [module docs](self).
pub struct Arena {
    /// Every chunk ever allocated since the last reset; the last one is current.
    chunks: RefCell<Vec<Chunk>>,
    /// Base of the current chunk (null before the first allocation).
    base: Cell<*mut u8>,
    /// Bytes used in the current chunk.
    offset: Cell<usize>,
    /// Capacity of the current chunk.
    cap: Cell<usize>,
    /// Size of the next chunk to allocate.
    next_chunk: Cell<usize>,
    /// Payload bytes handed out since the last reset (excludes alignment padding).
    used: Cell<usize>,
}

// SAFETY: the arena exclusively owns its chunks; the raw pointers never alias memory owned by
// anything else, and all outstanding borrows are tied to `&self`, which cannot cross threads
// because `Arena: !Sync`. Moving the whole arena to another thread is therefore sound.
unsafe impl Send for Arena {}

impl Arena {
    /// Creates an empty arena; the first chunk is allocated lazily.
    #[must_use]
    pub fn new() -> Self {
        Self::with_chunk_size(DEFAULT_CHUNK_SIZE)
    }

    /// Creates an empty arena whose first chunk will hold `chunk_size` bytes.
    #[must_use]
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunks: RefCell::new(Vec::new()),
            base: Cell::new(core::ptr::null_mut()),
            offset: Cell::new(0),
            cap: Cell::new(0),
            next_chunk: Cell::new(chunk_size.max(CACHE_LINE)),
            used: Cell::new(0),
        }
    }

    /// Bytes handed out since the last reset (excluding padding).
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.used.get()
    }

    /// Bytes reserved from the system allocator across all live chunks.
    #[must_use]
    pub fn capacity_bytes(&self) -> usize {
        self.chunks.borrow().iter().map(|c| c.layout.size()).sum()
    }

    /// Moves `value` into the arena.
    #[inline]
    #[allow(clippy::mut_from_ref)] // disjoint fresh allocation; see module-level safety model
    pub fn alloc<T: Copy>(&self, value: T) -> &mut T {
        let p = self.alloc_layout(Layout::new::<T>()).cast::<T>();
        // SAFETY: `p` is a fresh, properly aligned, exclusively owned region of
        // `size_of::<T>()` bytes that lives as long as `&self`.
        unsafe {
            p.as_ptr().write(value);
            &mut *p.as_ptr()
        }
    }

    /// Copies `src` into the arena and returns the copy.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice_copy<T: Copy>(&self, src: &[T]) -> &mut [T] {
        if src.is_empty() {
            return &mut [];
        }
        let layout = Layout::array::<T>(src.len()).unwrap_or_else(|_| layout_overflow());
        let p = self.alloc_layout(layout).cast::<T>();
        // SAFETY: fresh region sized and aligned for `src.len()` values of `T`; cannot overlap
        // `src`, which lives outside the region we just carved.
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), p.as_ptr(), src.len());
            core::slice::from_raw_parts_mut(p.as_ptr(), src.len())
        }
    }

    /// Copies a string into the arena.
    #[inline]
    pub fn alloc_str(&self, s: &str) -> &str {
        let bytes = self.alloc_slice_copy(s.as_bytes());
        // SAFETY: the bytes are an exact copy of a valid UTF-8 string.
        unsafe { core::str::from_utf8_unchecked(bytes) }
    }

    /// Concatenates `parts` into one contiguous arena string with a single bump.
    pub fn alloc_str_concat(&self, parts: &[&str]) -> &str {
        let total = parts
            .iter()
            .try_fold(0usize, |acc, p| acc.checked_add(p.len()))
            .unwrap_or_else(|| layout_overflow());
        if total == 0 {
            return "";
        }
        let layout = Layout::array::<u8>(total).unwrap_or_else(|_| layout_overflow());
        let dst = self.alloc_layout(layout).as_ptr();
        let mut at = 0;
        for p in parts {
            // SAFETY: `at + p.len() <= total`, so the write stays inside the fresh region.
            unsafe { core::ptr::copy_nonoverlapping(p.as_ptr(), dst.add(at), p.len()) };
            at += p.len();
        }
        // SAFETY: the region is fully initialized with the concatenation of valid UTF-8
        // strings, which is valid UTF-8.
        unsafe { core::str::from_utf8_unchecked(core::slice::from_raw_parts(dst, total)) }
    }

    /// Allocates raw, uninitialized memory for `layout`.
    ///
    /// The returned pointer is valid for reads and writes of `layout.size()` bytes until the
    /// arena is reset or dropped.
    #[inline]
    pub fn alloc_layout(&self, layout: Layout) -> NonNull<u8> {
        if layout.size() == 0 {
            // SAFETY: alignment is a non-zero power of two, hence non-null. Zero-sized accesses
            // through a dangling aligned pointer are valid.
            return unsafe {
                NonNull::new_unchecked(core::ptr::without_provenance_mut(layout.align()))
            };
        }
        // Fast path: bump inside the current chunk. `alloc_slow` is `#[cold]`, which already
        // biases the branch layout toward this path.
        if let Some(p) = self.try_bump(layout) {
            return p;
        }
        self.alloc_slow(layout)
    }

    #[inline(always)]
    fn try_bump(&self, layout: Layout) -> Option<NonNull<u8>> {
        let base = self.base.get();
        if base.is_null() {
            return None;
        }
        let addr = (base as usize).checked_add(self.offset.get())?;
        let aligned = addr.checked_add(layout.align() - 1)? & !(layout.align() - 1);
        let start = aligned - base as usize;
        let end = start.checked_add(layout.size())?;
        if end > self.cap.get() {
            return None;
        }
        self.offset.set(end);
        self.used.set(self.used.get() + layout.size());
        // SAFETY: `start <= end <= cap`, so `base + start` is inside (or one past) the current
        // chunk; `base` is non-null, so the result is non-null.
        Some(unsafe { NonNull::new_unchecked(base.add(start)) })
    }

    #[cold]
    #[inline(never)]
    fn alloc_slow(&self, layout: Layout) -> NonNull<u8> {
        let want = layout.size().checked_add(layout.align()).unwrap_or_else(|| layout_overflow());
        let size = self.next_chunk.get().max(want);
        let chunk_layout = Layout::from_size_align(size, layout.align().max(CACHE_LINE))
            .unwrap_or_else(|_| layout_overflow());
        // SAFETY: `size >= layout.size() + layout.align() > 0`.
        let raw = unsafe { alloc(chunk_layout) };
        let Some(ptr) = NonNull::new(raw) else { handle_alloc_error(chunk_layout) };
        self.chunks.borrow_mut().push(Chunk { ptr, layout: chunk_layout });
        self.base.set(ptr.as_ptr());
        self.offset.set(0);
        self.cap.set(size);
        self.next_chunk
            .set((size.saturating_mul(2)).min(MAX_CHUNK_SIZE).max(self.next_chunk.get()));
        self.try_bump(layout).expect("a fresh chunk always satisfies the request that sized it")
    }

    /// Frees everything allocated so far in O(chunks), keeping the largest chunk for reuse.
    ///
    /// Requires `&mut self`, which statically guarantees no allocation is still borrowed.
    pub fn reset(&mut self) {
        let chunks = self.chunks.get_mut();
        if let Some(keep_idx) =
            chunks.iter().enumerate().max_by_key(|(_, c)| c.layout.size()).map(|(i, _)| i)
        {
            let keep = chunks.swap_remove(keep_idx);
            for c in chunks.drain(..) {
                // SAFETY: each chunk was allocated with exactly its recorded layout and no
                // borrows into it can exist (we hold `&mut self`).
                unsafe { dealloc(c.ptr.as_ptr(), c.layout) };
            }
            self.base.set(keep.ptr.as_ptr());
            self.cap.set(keep.layout.size());
            chunks.push(keep);
        }
        self.offset.set(0);
        self.used.set(0);
    }
}

#[cold]
#[inline(never)]
fn layout_overflow() -> ! {
    panic!("arena allocation size overflow")
}

impl Drop for Arena {
    fn drop(&mut self) {
        for c in self.chunks.get_mut().drain(..) {
            // SAFETY: allocated with exactly this layout; no borrows outlive `self`.
            unsafe { dealloc(c.ptr.as_ptr(), c.layout) };
        }
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Arena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena")
            .field("chunks", &self.chunks.borrow().len())
            .field("allocated_bytes", &self.allocated_bytes())
            .field("capacity_bytes", &self.capacity_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_and_slices_are_disjoint_and_aligned() {
        let arena = Arena::with_chunk_size(128);
        let a = arena.alloc(1u8);
        let b = arena.alloc(0xDEAD_BEEF_u64);
        let c = arena.alloc_slice_copy(&[1.0f32, 2.0, 3.0]);
        assert_eq!(b as *mut u64 as usize % 8, 0);
        assert_eq!(c.as_ptr() as usize % 4, 0);
        *a = 9;
        c[1] = 42.0;
        assert_eq!(*a, 9);
        assert_eq!(*b, 0xDEAD_BEEF);
        assert_eq!(c, &[1.0, 42.0, 3.0]);
    }

    #[test]
    fn grows_across_chunks_and_strings_survive() {
        let arena = Arena::with_chunk_size(64);
        let mut strs = Vec::new();
        for i in 0..1000 {
            strs.push(arena.alloc_str(&format!("chunk-{i}")));
        }
        for (i, s) in strs.iter().enumerate() {
            assert_eq!(*s, format!("chunk-{i}"));
        }
        assert!(arena.chunks.borrow().len() > 1);
    }

    #[test]
    fn concat_and_empty() {
        let arena = Arena::new();
        assert_eq!(arena.alloc_str_concat(&["a", "", "bc", "δ"]), "abcδ");
        assert_eq!(arena.alloc_str_concat(&[]), "");
        assert!(arena.alloc_slice_copy::<u64>(&[]).is_empty());
        assert_eq!(arena.alloc_str(""), "");
    }

    #[test]
    fn oversized_request_gets_dedicated_chunk() {
        let arena = Arena::with_chunk_size(64);
        let big = arena.alloc_slice_copy(&vec![7u8; 1 << 20]);
        assert_eq!(big.len(), 1 << 20);
        assert!(big.iter().all(|&b| b == 7));
    }

    #[test]
    fn over_aligned_layout() {
        let arena = Arena::with_chunk_size(64);
        let _ = arena.alloc(1u8);
        let p = arena.alloc_layout(Layout::from_size_align(16, 256).unwrap());
        assert_eq!(p.as_ptr() as usize % 256, 0);
    }

    #[test]
    fn reset_reuses_largest_chunk() {
        let mut arena = Arena::with_chunk_size(64);
        for i in 0..10_000u32 {
            let _ = arena.alloc(i);
        }
        let cap_before = arena.capacity_bytes();
        arena.reset();
        assert_eq!(arena.allocated_bytes(), 0);
        assert_eq!(arena.chunks.borrow().len(), 1);
        assert!(arena.capacity_bytes() <= cap_before);
        let x = arena.alloc(5u64);
        assert_eq!(*x, 5);
    }
}
