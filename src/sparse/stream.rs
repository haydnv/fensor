use std::cmp::Ordering;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::ready;
use futures::stream::{Fuse, Stream, StreamExt};
use ha_ndarray::CDatatype;
use pin_project::pin_project;

use crate::Error;

// Based on: https://github.com/rust-lang/futures-rs/blob/master/futures-util/src/stream/select.rs
#[pin_project]
struct InnerJoin<L, R, T> {
    #[pin]
    left: Fuse<L>,
    #[pin]
    right: Fuse<R>,

    pending_left: Option<(u64, T)>,
    pending_right: Option<(u64, T)>,
}

impl<L, R, T> InnerJoin<L, R, T>
where
    L: Stream<Item = Result<(u64, T), Error>>,
    R: Stream<Item = Result<(u64, T), Error>>,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left: left.fuse(),
            right: right.fuse(),
            pending_left: None,
            pending_right: None,
        }
    }
}

impl<L, R, T> Stream for InnerJoin<L, R, T>
where
    L: Stream<Item = Result<(u64, T), Error>>,
    R: Stream<Item = Result<(u64, T), Error>>,
{
    type Item = Result<(u64, T, T), Error>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            let left_done = if this.left.is_done() {
                true
            } else if this.pending_left.is_none() {
                match ready!(this.left.as_mut().poll_next(cxt)) {
                    Some(Err(cause)) => break Some(Err(cause)),
                    Some(Ok(pending_left)) => {
                        *this.pending_left = Some(pending_left);
                        false
                    }
                    None => true,
                }
            } else {
                false
            };

            let right_done = if this.right.is_done() {
                true
            } else if this.pending_right.is_none() {
                match ready!(this.right.as_mut().poll_next(cxt)) {
                    Some(Err(cause)) => break Some(Err(cause)),
                    Some(Ok(pending_right)) => {
                        *this.pending_right = Some(pending_right);
                        false
                    }
                    None => true,
                }
            } else {
                false
            };

            if this.pending_left.is_some() && this.pending_right.is_some() {
                let l_offset = this.pending_left.as_ref().unwrap().0;
                let r_offset = this.pending_right.as_ref().unwrap().0;

                match l_offset.cmp(&r_offset) {
                    Ordering::Equal => {
                        let (l_offset, l_value) = this.pending_left.take().unwrap();
                        let (_r_offset, r_value) = this.pending_left.take().unwrap();
                        break Some(Ok((l_offset, l_value, r_value)));
                    }
                    Ordering::Less => {
                        this.pending_left.take();
                    }
                    Ordering::Greater => {
                        this.pending_right.take();
                    }
                }
            } else if left_done || right_done {
                break None;
            }
        })
    }
}

// Based on: https://github.com/rust-lang/futures-rs/blob/master/futures-util/src/stream/select.rs
#[pin_project]
struct OuterJoin<L, R, T> {
    #[pin]
    left: Fuse<L>,
    #[pin]
    right: Fuse<R>,

    pending_left: Option<(u64, T)>,
    pending_right: Option<(u64, T)>,
}

impl<L, R, T> OuterJoin<L, R, T>
where
    L: Stream<Item = Result<(u64, T), Error>>,
    R: Stream<Item = Result<(u64, T), Error>>,
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            left: left.fuse(),
            right: right.fuse(),
            pending_left: None,
            pending_right: None,
        }
    }
}

impl<L, R, T> Stream for OuterJoin<L, R, T>
where
    L: Stream<Item = Result<(u64, T), Error>>,
    R: Stream<Item = Result<(u64, T), Error>>,
    T: CDatatype,
{
    type Item = Result<(u64, T, T), Error>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            let left_done = if this.left.is_done() {
                true
            } else if this.pending_left.is_none() {
                match ready!(this.left.as_mut().poll_next(cxt)) {
                    Some(Err(cause)) => break Some(Err(cause)),
                    Some(Ok(pending_left)) => {
                        *this.pending_left = Some(pending_left);
                        false
                    }
                    None => true,
                }
            } else {
                false
            };

            let right_done = if this.right.is_done() {
                true
            } else if this.pending_right.is_none() {
                match ready!(this.right.as_mut().poll_next(cxt)) {
                    Some(Err(cause)) => break Some(Err(cause)),
                    Some(Ok(pending_right)) => {
                        *this.pending_right = Some(pending_right);
                        false
                    }
                    None => true,
                }
            } else {
                false
            };

            if this.pending_left.is_some() && this.pending_right.is_some() {
                let l_offset = this.pending_left.as_ref().unwrap().0;
                let r_offset = this.pending_right.as_ref().unwrap().0;

                break match l_offset.cmp(&r_offset) {
                    Ordering::Equal => {
                        let (offset, l_value) = this.pending_left.take().unwrap();
                        let (_offset, r_value) = this.pending_right.take().unwrap();
                        Some(Ok((offset, l_value, r_value)))
                    }
                    Ordering::Less => {
                        let (offset, l_value) = this.pending_left.take().unwrap();
                        Some(Ok((offset, l_value, T::zero())))
                    }
                    Ordering::Greater => {
                        let (offset, r_value) = this.pending_right.take().unwrap();
                        Some(Ok((offset, T::zero(), r_value)))
                    }
                };
            } else if right_done && this.pending_left.is_some() {
                let (offset, l_value) = this.pending_left.take().unwrap();
                break Some(Ok((offset, l_value, T::zero())));
            } else if left_done && this.pending_right.is_some() {
                let (offset, r_value) = this.pending_right.take().unwrap();
                break Some(Ok((offset, T::zero(), r_value)));
            } else if left_done && right_done {
                break None;
            }
        })
    }
}
