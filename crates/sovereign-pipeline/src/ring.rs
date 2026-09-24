//! Bounded, lock-free, async single-producer/single-consumer ring buffer.
//!
//! This is the transport between pipeline stages. Compared to `tokio::sync::mpsc` it trades
//! generality (exactly one sender, one receiver) for a much cheaper hot path: a push or pop is one
//! slot write/read plus one `Release` store, with no locks, no linked list and no per-message
//! allocation. Fan-out/fan-in is built from several rings (see `ingest`), the "shared-nothing"
//! topology used by LMAX Disruptor and Seastar.
//!
//! # Layout
//!
//! ```text
//!   head (consumer)                 tail (producer)
//!   CachePadded<AtomicUsize>        CachePadded<AtomicUsize>      ← separate 128 B lines: no
//!        │                               │                          false sharing between cores
//!        ▼                               ▼
//!   ┌────┬────┬────┬────┬────┬────┬────┬────┐
//!   │ T  │ T  │ T  │ T  │    │    │    │    │   capacity = 2^k, slot = index & (cap − 1)
//!   └────┴────┴────┴────┴────┴────┴────┴────┘
//! ```
//!
//! Indices increase monotonically (wrapping at `usize::MAX`), so `tail − head` is the fill level
//! and "full" vs "empty" is never ambiguous. Each side also keeps a *cached* copy of the other
//! side's index and only re-reads the shared atomic when the cache says full/empty, which keeps
//! the opposite core's cache line out of the fast path (Rigtorp's SPSC optimization).
//!
//! # Memory ordering
//!
//! * Producer: write slot → `tail.store(Release)`. Consumer: `tail.load(Acquire)` → read slot.
//!   The release/acquire pair makes the slot write visible before the index that publishes it.
//! * Symmetrically, `head.store(Release)` after reading a slot hands the slot back to the producer.
//!
//! # Async backpressure
//!
//! A full ring parks the producer (`AtomicWaker`) until the consumer frees a slot; an empty ring
//! parks the consumer until the producer pushes. Both sides follow the *register-then-recheck*
//! protocol, so a wakeup can never be lost between the failed attempt and parking.

use core::cell::UnsafeCell;
use core::fmt;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use core::task::{Context, Poll};
use std::sync::Arc;

use crossbeam_utils::CachePadded;
use futures::task::AtomicWaker;
use futures::Stream;

struct Shared<T> {
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
    buf: Box<[UnsafeCell<MaybeUninit<T>>]>,
    mask: usize,
    tx_closed: AtomicBool,
    rx_closed: AtomicBool,
    rx_waker: AtomicWaker,
    tx_waker: AtomicWaker,
}

// SAFETY: slots are handed between exactly one producer and one consumer through the
// acquire/release protocol above; a slot is never accessed by both sides at once. `T: Send` is
// required because values move across threads.
unsafe impl<T: Send> Send for Shared<T> {}
// SAFETY: see above — all shared mutation goes through atomics or through slot ownership.
unsafe impl<T: Send> Sync for Shared<T> {}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        // Both endpoints are gone (we hold the last `Arc`), so plain reads are race-free.
        let head = *self.head.get_mut();
        let tail = *self.tail.get_mut();
        let mut i = head;
        while i != tail {
            // SAFETY: slots in `[head, tail)` were written by the producer and not yet consumed.
            unsafe { (*self.buf[i & self.mask].get()).assume_init_drop() };
            i = i.wrapping_add(1);
        }
    }
}

/// Sending half. Not `Clone` — there is exactly one producer.
pub struct Producer<T> {
    shared: Arc<Shared<T>>,
    tail: usize,
    cached_head: usize,
}

/// Receiving half. Not `Clone` — there is exactly one consumer.
pub struct Consumer<T> {
    shared: Arc<Shared<T>>,
    head: usize,
    cached_tail: usize,
}

/// Error returned by [`Producer::send`] when the consumer is gone; carries the value back.
#[derive(Debug, PartialEq, Eq)]
pub struct SendError<T>(pub T);

impl<T> fmt::Display for SendError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ring consumer dropped")
    }
}

impl<T: fmt::Debug> std::error::Error for SendError<T> {}

/// Error returned by [`Producer::try_send`].
#[derive(Debug, PartialEq, Eq)]
pub enum TrySendError<T> {
    /// The ring is full.
    Full(T),
    /// The consumer is gone.
    Closed(T),
}

/// Creates a ring with room for at least `capacity` items (rounded up to a power of two, min 2).
///
/// # Panics
/// Panics if the rounded capacity overflows `usize`.
#[must_use]
pub fn channel<T>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    let cap = capacity.max(2).checked_next_power_of_two().expect("ring capacity overflow");
    let buf = (0..cap).map(|_| UnsafeCell::new(MaybeUninit::uninit())).collect();
    let shared = Arc::new(Shared {
        head: CachePadded::new(AtomicUsize::new(0)),
        tail: CachePadded::new(AtomicUsize::new(0)),
        buf,
        mask: cap - 1,
        tx_closed: AtomicBool::new(false),
        rx_closed: AtomicBool::new(false),
        rx_waker: AtomicWaker::new(),
        tx_waker: AtomicWaker::new(),
    });
    (
        Producer { shared: Arc::clone(&shared), tail: 0, cached_head: 0 },
        Consumer { shared, head: 0, cached_tail: 0 },
    )
}

impl<T> Producer<T> {
    /// Ring capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.shared.mask + 1
    }

    /// Items currently buffered (a racy snapshot, for metrics).
    #[must_use]
    pub fn len(&self) -> usize {
        self.tail.wrapping_sub(self.shared.head.load(Ordering::Relaxed))
    }

    /// `true` if nothing is buffered (racy snapshot).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `true` once the consumer has been dropped.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.shared.rx_closed.load(Ordering::Acquire)
    }

    /// Pushes without waiting.
    ///
    /// # Errors
    /// [`TrySendError::Full`] if there is no free slot, [`TrySendError::Closed`] if the consumer
    /// is gone. Both return the value.
    pub fn try_send(&mut self, value: T) -> Result<(), TrySendError<T>> {
        if self.is_closed() {
            return Err(TrySendError::Closed(value));
        }
        let cap = self.shared.mask + 1;
        if self.tail.wrapping_sub(self.cached_head) == cap {
            self.cached_head = self.shared.head.load(Ordering::Acquire);
            if self.tail.wrapping_sub(self.cached_head) == cap {
                return Err(TrySendError::Full(value));
            }
        }
        let slot = &self.shared.buf[self.tail & self.shared.mask];
        // SAFETY: `tail - head < cap`, so this slot is outside `[head, tail)`: the consumer has
        // released it (its `head` store was acquired above or earlier) and will not touch it until
        // we publish the new tail below. We are the only producer.
        unsafe { (*slot.get()).write(value) };
        self.tail = self.tail.wrapping_add(1);
        self.shared.tail.store(self.tail, Ordering::Release);
        self.shared.rx_waker.wake();
        Ok(())
    }

    fn poll_send(
        &mut self,
        cx: &mut Context<'_>,
        slot: &mut Option<T>,
    ) -> Poll<Result<(), SendError<T>>> {
        let value = slot.take().expect("poll_send polled after completion");
        match self.try_send(value) {
            Ok(()) => Poll::Ready(Ok(())),
            Err(TrySendError::Closed(v)) => Poll::Ready(Err(SendError(v))),
            Err(TrySendError::Full(v)) => {
                // Register, then re-check: if the consumer freed a slot between our failed attempt
                // and registration, the retry sees it; otherwise its next pop wakes us.
                self.shared.tx_waker.register(cx.waker());
                match self.try_send(v) {
                    Ok(()) => Poll::Ready(Ok(())),
                    Err(TrySendError::Closed(v)) => Poll::Ready(Err(SendError(v))),
                    Err(TrySendError::Full(v)) => {
                        *slot = Some(v);
                        Poll::Pending
                    }
                }
            }
        }
    }

    /// Pushes, waiting asynchronously while the ring is full (backpressure).
    ///
    /// # Errors
    /// [`SendError`] carrying the value if the consumer is gone.
    pub async fn send(&mut self, value: T) -> Result<(), SendError<T>> {
        let mut slot = Some(value);
        futures::future::poll_fn(|cx| self.poll_send(cx, &mut slot)).await
    }
}

impl<T> Drop for Producer<T> {
    fn drop(&mut self) {
        // Release: everything pushed before the close is visible to a consumer that observes it.
        self.shared.tx_closed.store(true, Ordering::Release);
        self.shared.rx_waker.wake();
    }
}

impl<T> Consumer<T> {
    /// Ring capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.shared.mask + 1
    }

    /// Pops without waiting. `None` means "empty right now" (check [`is_finished`](Self::is_finished)).
    pub fn try_recv(&mut self) -> Option<T> {
        if self.head == self.cached_tail {
            self.cached_tail = self.shared.tail.load(Ordering::Acquire);
            if self.head == self.cached_tail {
                return None;
            }
        }
        let slot = &self.shared.buf[self.head & self.shared.mask];
        // SAFETY: `head < tail` (acquired), so the producer has fully written this slot and
        // published it with a release store; it will not reuse the slot until we advance `head`.
        let value = unsafe { (*slot.get()).assume_init_read() };
        self.head = self.head.wrapping_add(1);
        self.shared.head.store(self.head, Ordering::Release);
        self.shared.tx_waker.wake();
        Some(value)
    }

    /// `true` once the producer is gone *and* the ring is drained.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.shared.tx_closed.load(Ordering::Acquire)
            && self.head == self.shared.tail.load(Ordering::Acquire)
    }

    /// Polls for the next item; `Ready(None)` once the producer is gone and the ring is drained.
    pub fn poll_recv(&mut self, cx: &mut Context<'_>) -> Poll<Option<T>> {
        if let Some(v) = self.try_recv() {
            return Poll::Ready(Some(v));
        }
        self.shared.rx_waker.register(cx.waker());
        if let Some(v) = self.try_recv() {
            return Poll::Ready(Some(v));
        }
        if self.shared.tx_closed.load(Ordering::Acquire) {
            // The acquire above synchronizes with the producer's final release, so this last pop
            // observes every item pushed before the close.
            return Poll::Ready(self.try_recv());
        }
        Poll::Pending
    }

    /// Receives the next item, waiting while the ring is empty.
    pub async fn recv(&mut self) -> Option<T> {
        futures::future::poll_fn(|cx| self.poll_recv(cx)).await
    }
}

impl<T> Drop for Consumer<T> {
    fn drop(&mut self) {
        self.shared.rx_closed.store(true, Ordering::Release);
        self.shared.tx_waker.wake();
    }
}

impl<T> Stream for Consumer<T> {
    type Item = T;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T>> {
        self.get_mut().poll_recv(cx)
    }
}

impl<T> fmt::Debug for Producer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Producer")
            .field("capacity", &self.capacity())
            .field("len", &self.len())
            .finish()
    }
}

impl<T> fmt::Debug for Consumer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Consumer").field("capacity", &self.capacity()).finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use tokio_stream::StreamExt;

    #[test]
    fn capacity_rounds_to_power_of_two() {
        assert_eq!(channel::<u8>(0).0.capacity(), 2);
        assert_eq!(channel::<u8>(5).0.capacity(), 8);
        assert_eq!(channel::<u8>(64).1.capacity(), 64);
    }

    #[test]
    fn fifo_full_and_wraparound() {
        let (mut tx, mut rx) = channel(4);
        for round in 0..10u32 {
            for i in 0..4 {
                tx.try_send(round * 4 + i).unwrap();
            }
            assert_eq!(tx.try_send(99), Err(TrySendError::Full(99)));
            for i in 0..4 {
                assert_eq!(rx.try_recv(), Some(round * 4 + i));
            }
            assert_eq!(rx.try_recv(), None);
        }
    }

    #[test]
    fn close_semantics() {
        let (mut tx, mut rx) = channel(4);
        tx.try_send(1).unwrap();
        drop(tx);
        assert!(!rx.is_finished());
        assert_eq!(rx.try_recv(), Some(1));
        assert!(rx.is_finished());

        let (mut tx, rx) = channel::<i32>(4);
        drop(rx);
        assert!(tx.is_closed());
        assert_eq!(tx.try_send(7), Err(TrySendError::Closed(7)));
    }

    #[test]
    fn undelivered_items_are_dropped_exactly_once() {
        #[derive(Debug)]
        struct Tracked(Arc<AtomicUsize>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let (mut tx, mut rx) = channel(8);
        for _ in 0..6 {
            tx.try_send(Tracked(Arc::clone(&drops))).unwrap();
        }
        drop(rx.try_recv());
        drop(rx.try_recv());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        drop(tx);
        drop(rx);
        assert_eq!(drops.load(Ordering::SeqCst), 6);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cross_thread_stream_with_backpressure() {
        const N: u64 = 200_000;
        let (mut tx, rx) = channel::<u64>(16);
        let producer = tokio::spawn(async move {
            for i in 0..N {
                tx.send(i).await.unwrap();
            }
        });
        let consumer = tokio::spawn(async move {
            let mut expected = 0u64;
            let mut rx = rx;
            while let Some(v) = rx.next().await {
                assert_eq!(v, expected, "FIFO order violated");
                expected += 1;
            }
            expected
        });
        producer.await.unwrap();
        assert_eq!(consumer.await.unwrap(), N);
    }

    #[tokio::test]
    async fn send_fails_when_consumer_dropped_while_waiting() {
        let (mut tx, rx) = channel::<u32>(2);
        tx.send(1).await.unwrap();
        tx.send(2).await.unwrap();
        let waiter = tokio::spawn(async move { tx.send(3).await });
        tokio::task::yield_now().await;
        drop(rx);
        assert_eq!(waiter.await.unwrap(), Err(SendError(3)));
    }
}
