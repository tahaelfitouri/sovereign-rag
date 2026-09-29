//! Zero-copy, row-major matrix views with cache-line padded rows.
//!
//! # Row padding
//!
//! Every row is padded with zeros to a multiple of 16 `f32`s (64 bytes):
//!
//! ```text
//!  dim = 100, stride = padded_stride(100) = 112
//!
//!  row i: [ x0 x1 ... x99 | 0 0 0 0 0 0 0 0 0 0 0 0 ]   <- 448 bytes = 7 cache lines
//!         \___ dim ______/ \______ padding _______/
//!         \_______________ stride ________________/
//! ```
//!
//! Zero padding is *neutral* for all three metrics (`0·0 = 0`, `(0−0)² = 0`), so SIMD kernels run
//! over `stride` elements with no scalar tail and no masking, and — combined with a 64-byte
//! aligned base — every row starts on a cache-line boundary.

use crate::aligned::{AlignedVec, CACHE_LINE};
use crate::error::{CoreError, Result};

/// Number of `f32` lanes in one cache line.
pub const LANES_PER_LINE: usize = CACHE_LINE / core::mem::size_of::<f32>();

/// Rounds `dim` up to a multiple of [`LANES_PER_LINE`] (16).
///
/// # Errors
/// [`CoreError::CapacityOverflow`] if the rounded value does not fit in `usize`.
pub const fn padded_stride(dim: usize) -> Result<usize> {
    match dim.checked_add(LANES_PER_LINE - 1) {
        Some(v) => Ok(v & !(LANES_PER_LINE - 1)),
        None => Err(CoreError::CapacityOverflow),
    }
}

/// Borrowed view over `rows × stride` `f32`s, of which the first `dim` of each row are live.
///
/// The view is `Copy` and borrows the underlying storage (an [`AlignedVec`] or an `mmap`).
#[derive(Clone, Copy, Debug)]
pub struct MatrixRef<'a> {
    data: &'a [f32],
    rows: usize,
    dim: usize,
    stride: usize,
}

impl<'a> MatrixRef<'a> {
    /// Creates a view, validating the shape.
    ///
    /// # Errors
    /// * [`CoreError::InvalidLayout`] if `stride < dim` or `data.len() != rows * stride`.
    /// * [`CoreError::CapacityOverflow`] if `rows * stride` overflows.
    pub fn new(data: &'a [f32], rows: usize, dim: usize, stride: usize) -> Result<Self> {
        if stride < dim {
            return Err(CoreError::InvalidLayout("stride is smaller than dim"));
        }
        let expected = rows.checked_mul(stride).ok_or(CoreError::CapacityOverflow)?;
        if data.len() != expected {
            return Err(CoreError::InvalidLayout("data length != rows * stride"));
        }
        Ok(Self { data, rows, dim, stride })
    }

    /// Reinterprets raw bytes (e.g. a region of a memory-mapped file) as a matrix, zero-copy.
    ///
    /// # Errors
    /// * [`CoreError::Misaligned`] if `bytes` is not 4-byte aligned.
    /// * [`CoreError::InvalidLayout`] if the length is not a whole number of `f32`s or does not
    ///   match the shape.
    pub fn from_bytes(bytes: &'a [u8], rows: usize, dim: usize, stride: usize) -> Result<Self> {
        let data: &[f32] = bytemuck::try_cast_slice(bytes).map_err(|e| match e {
            bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                CoreError::Misaligned { addr: bytes.as_ptr() as usize, required: 4 }
            }
            _ => CoreError::InvalidLayout("byte length is not a multiple of 4"),
        })?;
        Self::new(data, rows, dim, stride)
    }

    /// Number of rows.
    #[inline]
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Logical dimensionality.
    #[inline]
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Physical row length in `f32`s (≥ `dim`, multiple of 16 when built by this crate).
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// `true` if the view has no rows.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// `true` if the base pointer is cache-line aligned (every row then is, when
    /// `stride % 16 == 0`).
    #[must_use]
    pub fn is_cache_aligned(&self) -> bool {
        (self.data.as_ptr() as usize).is_multiple_of(CACHE_LINE)
    }

    /// The whole backing slice.
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &'a [f32] {
        self.data
    }

    /// Row `i`, logical part only (`dim` elements).
    ///
    /// # Panics
    /// Panics if `i >= rows`.
    #[inline]
    #[must_use]
    pub fn row(&self, i: usize) -> &'a [f32] {
        &self.row_padded(i)[..self.dim]
    }

    /// Row `i` including zero padding (`stride` elements) — what SIMD kernels should consume.
    ///
    /// # Panics
    /// Panics if `i >= rows`.
    #[inline]
    #[must_use]
    pub fn row_padded(&self, i: usize) -> &'a [f32] {
        assert!(i < self.rows, "row {i} out of bounds ({} rows)", self.rows);
        // SAFETY: bounds checked just above.
        unsafe { self.row_padded_unchecked(i) }
    }

    /// Row `i` including padding, without bounds checks.
    ///
    /// # Safety
    /// `i < self.rows()` must hold.
    #[inline(always)]
    #[must_use]
    pub unsafe fn row_padded_unchecked(&self, i: usize) -> &'a [f32] {
        debug_assert!(i < self.rows);
        // SAFETY: the constructor proved `data.len() == rows * stride` without overflow; with
        // `i < rows`, `[i*stride, i*stride + stride)` lies within `data`.
        unsafe { self.data.get_unchecked(i * self.stride..i * self.stride + self.stride) }
    }

    /// Pointer to the start of row `i` (for prefetching). Never dereferenced by this call.
    #[inline(always)]
    #[must_use]
    pub fn row_ptr(&self, i: usize) -> *const f32 {
        self.data.as_ptr().wrapping_add(i.wrapping_mul(self.stride))
    }

    /// Iterates over padded rows.
    pub fn iter_padded(&self) -> impl ExactSizeIterator<Item = &'a [f32]> + 'a {
        let stride = self.stride.max(1);
        self.data.chunks_exact(stride).take(self.rows)
    }
}

/// Appends `row` to `buf` as one padded row of `stride` elements.
///
/// # Errors
/// [`CoreError::DimensionMismatch`] if `row.len() > stride`.
pub fn push_padded_row(buf: &mut AlignedVec<f32>, row: &[f32], stride: usize) -> Result<()> {
    if row.len() > stride {
        return Err(CoreError::DimensionMismatch { expected: stride, actual: row.len() });
    }
    buf.reserve(stride);
    buf.extend_from_slice(row);
    let new_len = buf.len() + (stride - row.len());
    buf.resize(new_len, 0.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_rounding() {
        assert_eq!(padded_stride(1).unwrap(), 16);
        assert_eq!(padded_stride(16).unwrap(), 16);
        assert_eq!(padded_stride(100).unwrap(), 112);
        assert_eq!(padded_stride(1536).unwrap(), 1536);
        assert_eq!(padded_stride(0).unwrap(), 0);
        assert!(padded_stride(usize::MAX).is_err());
    }

    #[test]
    fn view_rows_and_padding() {
        let dim = 3;
        let stride = padded_stride(dim).unwrap();
        let mut buf = AlignedVec::new();
        push_padded_row(&mut buf, &[1.0, 2.0, 3.0], stride).unwrap();
        push_padded_row(&mut buf, &[4.0, 5.0, 6.0], stride).unwrap();
        let m = MatrixRef::new(&buf, 2, dim, stride).unwrap();
        assert!(m.is_cache_aligned());
        assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
        assert_eq!(m.row_padded(0).len(), 16);
        assert!(m.row_padded(0)[3..].iter().all(|&x| x == 0.0));
        assert_eq!(m.iter_padded().count(), 2);
    }

    #[test]
    fn shape_validation() {
        let data = [0.0f32; 32];
        assert!(MatrixRef::new(&data, 2, 16, 16).is_ok());
        assert!(MatrixRef::new(&data, 3, 16, 16).is_err());
        assert!(MatrixRef::new(&data, 2, 17, 16).is_err());
        assert!(MatrixRef::new(&data, usize::MAX, 16, 16).is_err());
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        assert!(MatrixRef::from_bytes(bytes, 2, 16, 16).is_ok());
        assert!(MatrixRef::from_bytes(&bytes[1..65], 1, 16, 16).is_err());
        let mut buf = AlignedVec::new();
        assert!(push_padded_row(&mut buf, &[0.0; 17], 16).is_err());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn row_oob_panics() {
        let data = [0.0f32; 16];
        let m = MatrixRef::new(&data, 1, 16, 16).unwrap();
        let _ = m.row(1);
    }
}
