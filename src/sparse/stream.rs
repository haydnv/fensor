use std::cmp::Ordering;
use std::mem;
use std::pin::Pin;
use std::task::{ready, Context, Poll};

use futures::stream::{Fuse, Stream, StreamExt};
use ha_ndarray::{ArrayBase, CDatatype};
use pin_project::pin_project;

use crate::{Coord, Error, IDEAL_BLOCK_SIZE};

#[pin_project]
pub struct BlockCoords<S, T> {
    #[pin]
    source: Fuse<S>,
    pending_coords: Vec<u64>,
    pending_values: Vec<T>,
    ndim: usize,
}

impl<S, T> BlockCoords<S, T>
where
    S: Stream<Item = Result<(Coord, T), Error>>,
{
    pub fn new(source: S, ndim: usize) -> Self {
        Self {
            source: source.fuse(),
            pending_coords: Vec::with_capacity(IDEAL_BLOCK_SIZE),
            pending_values: Vec::with_capacity(IDEAL_BLOCK_SIZE),
            ndim,
        }
    }
}

impl<S, T> BlockCoords<S, T>
where
    T: CDatatype,
{
    fn block_cutoff(
        pending_coords: &mut Vec<u64>,
        pending_values: &mut Vec<T>,
        ndim: usize,
    ) -> Result<(ArrayBase<u64>, Vec<T>), Error> {
        debug_assert_eq!(pending_coords.len() % ndim, 0);

        let values = pending_values.drain(..).collect();
        let coords = ArrayBase::new(
            vec![pending_coords.len() / ndim, ndim],
            pending_coords.drain(..).collect(),
        )?;

        Ok((coords, values))
    }
}

impl<S, T> Stream for BlockCoords<S, T>
where
    S: Stream<Item = Result<(Coord, T), Error>> + Unpin,
    T: CDatatype,
{
    type Item = Result<(ArrayBase<u64>, Vec<T>), Error>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let ndim = self.ndim;
        let mut this = self.project();

        Poll::Ready(loop {
            debug_assert_eq!(this.pending_values.len() * ndim, this.pending_coords.len());

            match ready!(this.source.as_mut().poll_next(cxt)) {
                Some(Ok((coord, value))) => {
                    debug_assert_eq!(coord.len(), *this.ndim);

                    this.pending_coords.extend(coord);
                    this.pending_values.push(value);

                    if this.pending_values.len() == IDEAL_BLOCK_SIZE {
                        break Some(Self::block_cutoff(
                            this.pending_coords,
                            this.pending_values,
                            ndim,
                        ));
                    }
                }
                None if !this.pending_values.is_empty() => {
                    break Some(Self::block_cutoff(
                        this.pending_coords,
                        this.pending_values,
                        ndim,
                    ));
                }
                None => break None,
                Some(Err(cause)) => break Some(Err(cause)),
            }
        })
    }
}

#[pin_project]
pub struct FilledAt<S> {
    #[pin]
    source: S,

    pending: Option<Vec<u64>>,
    axes: Vec<usize>,
    ndim: usize,
}

impl<S> FilledAt<S> {
    pub fn new(source: S, axes: Vec<usize>, ndim: usize) -> Self {
        debug_assert!(!axes.iter().copied().any(|x| x >= ndim));

        Self {
            source,
            pending: None,
            axes,
            ndim,
        }
    }
}

impl<T, S: Stream<Item = Result<(Coord, T), Error>>> Stream for FilledAt<S> {
    type Item = Result<Vec<u64>, Error>;

    fn poll_next(self: Pin<&mut Self>, cxt: &mut Context) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        Poll::Ready(loop {
            match ready!(this.source.as_mut().poll_next(cxt)) {
                Some(Ok((coord, _))) => match this.pending.as_mut() {
                    None => {
                        let filled = this.axes.iter().copied().map(|x| coord[x]).collect();
                        *this.pending = Some(filled);
                    }
                    Some(pending) => {
                        if this
                            .axes
                            .iter()
                            .copied()
                            .map(|x| coord[x])
                            .zip(pending.iter().copied())
                            .any(|(l, r)| l != r)
                        {
                            let mut filled =
                                Some(this.axes.iter().copied().map(|x| coord[x]).collect());

                            mem::swap(&mut *this.pending, &mut filled);
                            break filled.map(Ok);
                        }
                    }
                },
                None => break None,
                Some(Err(cause)) => break Some(Err(cause)),
            }
        })
    }
}

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
