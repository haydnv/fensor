use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Bound;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use b_table::{Range, TableLock};
use freqfs::DirLock;
use futures::future::TryFutureExt;
use futures::stream::{Stream, StreamExt, TryStreamExt};
use ha_ndarray::*;
use number_general::{DType, Number, NumberCollator, NumberType};
use rayon::prelude::*;
use safecast::{AsType, CastInto};

use crate::{strides_for, Axes, AxisBound, Bounds, Coord, Error, Shape, Strides, TensorInstance};

use super::stream;
use super::{Elements, IndexSchema, Node, Schema};

#[async_trait]
pub trait SparseInstance: TensorInstance + fmt::Debug {
    type CoordBlock: NDArrayRead<DType = u64> + NDArrayMath + NDArrayTransform;
    type ValueBlock: NDArrayRead<DType = Self::DType>;
    type Blocks: Stream<Item = Result<(Self::CoordBlock, Self::ValueBlock), Error>>;
    type DType: CDatatype + DType;

    async fn blocks(self, bounds: Bounds) -> Result<Self::Blocks, Error>;

    async fn elements(self, bounds: Bounds) -> Result<Elements<Self::DType>, Error>;

    async fn filled_at(
        self,
        bounds: Bounds,
        axes: Axes,
    ) -> Result<stream::FilledAt<Elements<Self::DType>>, Error>
    where
        Self: Sized,
    {
        let ndim = self.ndim();

        self.elements(bounds)
            .map_ok(|elements| stream::FilledAt::new(elements, axes, ndim))
            .await
    }
}

#[derive(Clone)]
pub enum SparseAccess<FE, T> {
    Table(SparseTable<FE, T>),
    Broadcast(Box<SparseBroadcast<Self>>),
    Expand(Box<SparseExpand<Self>>),
    Reshape(Box<SparseReshape<Self>>),
    Slice(Box<SparseSlice<Self>>),
}

macro_rules! array_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::Table($var) => $call,
            Self::Broadcast($var) => $call,
            Self::Expand($var) => $call,
            Self::Reshape($var) => $call,
            Self::Slice($var) => $call,
        }
    };
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for SparseAccess<FE, T> {
    fn dtype(&self) -> NumberType {
        array_dispatch!(self, this, this.dtype())
    }

    fn shape(&self) -> &[u64] {
        array_dispatch!(self, this, this.shape())
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseAccess<FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType,
    Number: CastInto<T>,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Pin<Box<dyn Stream<Item = Result<(Array<u64>, Array<T>), Error>>>>;
    type DType = T;

    async fn blocks(self, bounds: Bounds) -> Result<Self::Blocks, Error> {
        match self {
            Self::Table(table) => {
                let blocks = table.blocks(bounds).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Broadcast(broadcast) => {
                let blocks = broadcast.blocks(bounds).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Expand(expand) => {
                let blocks = expand.blocks(bounds).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Reshape(reshape) => {
                let blocks = reshape.blocks(bounds).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Slice(slice) => {
                let blocks = slice.blocks(bounds).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
        }
    }

    async fn elements(self, bounds: Bounds) -> Result<Elements<Self::DType>, Error> {
        array_dispatch!(self, this, this.elements(bounds).await)
    }
}

impl<FE, T> fmt::Debug for SparseAccess<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        array_dispatch!(self, this, this.fmt(f))
    }
}

pub struct SparseTable<FE, T> {
    table: TableLock<Schema, IndexSchema, NumberCollator, FE>,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for SparseTable<FE, T> {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            dtype: PhantomData,
        }
    }
}

impl<FE: AsType<Node> + Send + Sync, T> SparseTable<FE, T> {
    pub async fn create(dir: DirLock<FE>, shape: Shape) -> Result<Self, Error> {
        let schema = Schema::new(shape);
        let collator = NumberCollator::default();
        let table = TableLock::create(schema, collator, dir)?;

        Ok(Self {
            table,
            dtype: PhantomData,
        })
    }
}

impl<FE, T> TensorInstance for SparseTable<FE, T>
where
    FE: Send + Sync + 'static,
    T: DType + Send + Sync + 'static,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.table.schema().shape
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseTable<FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType,
    Number: CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<T>, T>;
    type DType = T;

    async fn blocks(self, bounds: Bounds) -> Result<Self::Blocks, Error> {
        let ndim = self.ndim();
        let elements = self.elements(bounds).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, bounds: Bounds) -> Result<Elements<Self::DType>, Error> {
        let range = range_from_bounds(&bounds, self.shape())?;
        let rows = self.table.rows(range, &[], false).await?;
        let elements = rows.map_ok(|row| unwrap_row(row)).map_err(Error::from);
        Ok(Box::pin(elements))
    }
}

impl<FE, T> fmt::Debug for SparseTable<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "sparse table with shape {:?}", self.table.schema().shape)
    }
}

#[derive(Clone)]
pub struct SparseBroadcast<S> {
    source: S,
    axis: usize,
    dim: u64,
    shape: Shape,
}

impl<S: TensorInstance + fmt::Debug> SparseBroadcast<S> {
    fn new(source: S, axis: usize, dim: u64) -> Result<Self, Error> {
        let shape = if axis < source.ndim() {
            let mut shape = source.shape().to_vec();
            if shape[axis] == 1 {
                shape[axis] = dim;
                Ok(shape)
            } else {
                Err(Error::Bounds(format!(
                    "cannot broadcast dimension {} into {}",
                    shape[axis], dim
                )))
            }
        } else {
            Err(Error::Bounds(format!(
                "invalid axis for {:?}: {}",
                source, axis
            )))
        }?;

        Ok(Self {
            source,
            axis,
            dim,
            shape,
        })
    }
}

impl<S: TensorInstance> TensorInstance for SparseBroadcast<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseBroadcast<S> {
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = S::DType;

    async fn blocks(self, bounds: Bounds) -> Result<Self::Blocks, Error> {
        let ndim = self.ndim();
        let elements = self.elements(bounds).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, bounds: Bounds) -> Result<Elements<Self::DType>, Error> {
        let (source_bounds, dim) = if bounds.0.len() > self.axis {
            let bdim = match &bounds.0[self.axis] {
                AxisBound::At(i) if *i < self.dim => Ok(1),
                AxisBound::In(start, stop, 1) if stop > start => Ok(stop - start),
                bound => Err(Error::Bounds(format!(
                    "invalid bound for axis {}: {:?}",
                    self.axis, bound
                ))),
            }?;

            let mut source_bounds = bounds;
            source_bounds.0[self.axis] = AxisBound::At(0);
            (source_bounds, bdim)
        } else {
            (bounds, self.dim)
        };

        if self.axis == self.ndim() - 1 {
            let source_elements = self.source.elements(source_bounds).await?;

            let elements = source_elements
                .map_ok(move |(source_coord, value)| {
                    futures::stream::iter(0..dim).map(move |i| {
                        let mut coord = source_coord.to_vec();
                        *coord.last_mut().expect("x") = i;
                        Ok((coord, value))
                    })
                })
                .try_flatten();

            Ok(Box::pin(elements))
        } else {
            let axes = (0..self.axis).into_iter().collect();
            let _filled = self.source.filled_at(source_bounds, axes).await?;

            // TODO: for each filled prefix, slice the source tensor and repeat the slice dim times

            todo!()
        }
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseBroadcast<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "broadcast of {:?} axis {}", self.source, self.axis)
    }
}

#[derive(Clone)]
pub struct SparseExpand<S> {
    source: S,
    shape: Shape,
    axes: Axes,
}

impl<S: TensorInstance + fmt::Debug> SparseExpand<S> {
    fn new(source: S, mut axes: Axes) -> Result<Self, Error> {
        axes.sort();

        let mut shape = source.shape().to_vec();
        for x in axes.iter().rev().copied() {
            shape.insert(x, 1);
        }

        if Some(source.ndim()) > axes.last().copied() {
            Ok(Self {
                source,
                shape,
                axes,
            })
        } else {
            Err(Error::Bounds(format!(
                "cannot expand axes {:?} of {:?}",
                axes, source
            )))
        }
    }
}

impl<S: TensorInstance> TensorInstance for SparseExpand<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseExpand<S> {
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = S::DType;

    async fn blocks(self, bounds: Bounds) -> Result<Self::Blocks, Error> {
        let ndim = self.ndim();
        let elements = self.elements(bounds).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, bounds: Bounds) -> Result<Elements<Self::DType>, Error> {
        let mut source_bounds = bounds;
        for x in self.axes.iter().rev().copied() {
            if x < source_bounds.0.len() {
                source_bounds.0.remove(x);
            }
        }

        let ndim = self.ndim();
        let axes = self.axes;
        debug_assert_eq!(self.source.ndim() + 1, ndim);

        let source_elements = self.source.elements(source_bounds).await?;

        let elements = source_elements.map_ok(move |(source_coord, value)| {
            let mut coord = Coord::with_capacity(ndim);
            coord.extend(source_coord);
            for x in axes.iter().rev().copied() {
                coord.insert(x, 0);
            }

            debug_assert_eq!(coord.len(), ndim);
            (coord, value)
        });

        Ok(Box::pin(elements))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseExpand<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "expand axes {:?} of {:?}", self.axes, self.source)
    }
}

#[derive(Clone)]
pub struct SparseReshape<S> {
    source: S,
    source_strides: Strides,
    shape: Shape,
    strides: Strides,
}

impl<S: TensorInstance> SparseReshape<S> {
    fn new(source: S, shape: Shape) -> Self {
        let source_strides = strides_for(source.shape(), source.ndim());
        let strides = strides_for(&shape, shape.len());

        Self {
            source,
            source_strides,
            shape,
            strides,
        }
    }
}

impl<S: TensorInstance> TensorInstance for SparseReshape<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseReshape<S> {
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = S::ValueBlock;
    type Blocks = Pin<Box<dyn Stream<Item = Result<(Self::CoordBlock, Self::ValueBlock), Error>>>>;
    type DType = S::DType;

    async fn blocks(self, bounds: Bounds) -> Result<Self::Blocks, Error> {
        let source_bounds = if bounds.0.is_empty() {
            Ok(bounds)
        } else {
            Err(Error::Bounds(format!(
                "cannot slice a reshaped sparse tensor (consider making a copy first)"
            )))
        }?;

        let source_ndim = self.source.ndim();
        let source_blocks = self.source.blocks(source_bounds).await?;
        let source_strides =
            ArrayBase::<Arc<Vec<_>>>::new(vec![source_ndim], Arc::new(self.source_strides))?;

        let ndim = self.shape.len();
        let strides = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], Arc::new(self.strides))?;
        let shape = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], Arc::new(self.shape))?;

        let blocks = source_blocks.map(move |result| {
            let (source_coords, values) = result?;

            debug_assert_eq!(source_coords.size() % source_ndim, 0);
            debug_assert_eq!(source_coords.size() / source_ndim, values.size());

            let source_strides = source_strides
                .clone()
                .broadcast(vec![values.size(), source_ndim])?;

            let offsets = source_coords.mul(source_strides)?;
            let offsets = offsets.sum_axis(1)?;

            let broadcast = vec![offsets.size(), ndim];
            let strides = strides.clone().broadcast(broadcast.to_vec())?;
            let offsets = offsets
                .expand_dims(vec![1])?
                .broadcast(broadcast.to_vec())?;

            let dims = shape.clone().expand_dims(vec![0])?.broadcast(broadcast)?;
            let coords = (offsets / strides) % dims;

            let coords = ArrayBase::<Vec<_>>::copy(&coords)?;

            Result::<_, Error>::Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(self, bounds: Bounds) -> Result<Elements<Self::DType>, Error> {
        let ndim = self.shape.len();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let blocks = self.blocks(bounds).await?;

        let elements = blocks
            .map(move |result| {
                let (coords, values) = result?;
                let coords = coords.into_inner();
                let values = values.read(&queue)?.to_slice()?.into_vec();
                let coords = coords.into_par_iter().chunks(ndim).collect::<Vec<Coord>>();

                Result::<_, Error>::Ok(futures::stream::iter(
                    coords.into_iter().zip(values).map(Ok),
                ))
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseReshape<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "reshape {:?} into {:?}", self.source, self.shape)
    }
}

#[derive(Clone)]
pub struct SparseSlice<S> {
    source: S,
    bounds: Bounds,
    shape: Shape,
}

impl<S> SparseSlice<S>
where
    S: TensorInstance + fmt::Debug,
{
    fn new(source: S, bounds: Bounds) -> Result<Self, Error> {
        if bounds.0.len() > source.ndim() {
            return Err(Error::Bounds(format!(
                "invalid slice bounds for {:?}: {:?}",
                source, bounds.0
            )));
        }

        let mut shape = Shape::with_capacity(source.ndim());
        for (x, bound) in bounds.0.iter().enumerate() {
            match bound {
                AxisBound::At(_) => {} // no-op
                AxisBound::In(start, stop, 1) => {
                    shape.push(stop - start);
                }
                axis_bound => {
                    return Err(Error::Bounds(format!(
                        "invalid bound for sparse tensor axis {}: {:?}",
                        x, axis_bound
                    )));
                }
            }
        }

        shape.extend_from_slice(&source.shape()[bounds.0.len()..]);

        Ok(Self {
            source,
            bounds,
            shape,
        })
    }

    fn source_bounds(&self, bounds: Bounds) -> Result<Bounds, Error> {
        if bounds.0.len() > self.ndim() {
            Err(Error::Bounds(format!(
                "invalid bounds for {:?}: {:?}",
                self, bounds.0
            )))
        } else {
            let mut source_bounds = Vec::with_capacity(self.source.ndim());

            let mut source_axis = 0;
            let mut axis = 0;

            while source_bounds.len() < bounds.0.len() {
                match &self.bounds.0[source_axis] {
                    AxisBound::At(i) => {
                        source_bounds.push(AxisBound::At(*i));
                    }
                    AxisBound::In(start, stop, 1) => {
                        let source_bound = match &bounds.0[axis] {
                            AxisBound::At(i) => {
                                let i = start + i;
                                if i < *stop {
                                    Ok(AxisBound::At(i))
                                } else {
                                    Err(Error::Bounds(format!(
                                        "index {} is out of bounds for axis {}",
                                        i, axis
                                    )))
                                }
                            }
                            AxisBound::In(start, stop, 1) => {
                                let (source_start, source_stop) = (start + start, stop + start);
                                if source_stop <= *stop {
                                    Ok(AxisBound::In(source_start, source_stop, 1))
                                } else {
                                    Err(Error::Bounds(format!(
                                        "range [{}, {}) is out of bounds for axis {}",
                                        source_start, source_stop, axis
                                    )))
                                }
                            }
                            bound => Err(Error::Bounds(format!(
                                "invalid bound for axis {}: {:?}",
                                axis, bound
                            ))),
                        }?;

                        source_bounds.push(source_bound);
                        axis += 1;
                    }
                    bound => {
                        return Err(Error::Bounds(format!(
                            "invalid bound for sparse tensor: {:?}",
                            bound
                        )))
                    }
                }

                source_axis += 1;
            }

            source_bounds.extend(self.bounds.0.iter().skip(bounds.0.len()).cloned());

            Ok(Bounds(source_bounds))
        }
    }
}

impl<S> TensorInstance for SparseSlice<S>
where
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<S> SparseInstance for SparseSlice<S>
where
    S: SparseInstance,
{
    type CoordBlock = S::CoordBlock;
    type ValueBlock = S::ValueBlock;
    type Blocks = S::Blocks;
    type DType = S::DType;

    async fn blocks(self, bounds: Bounds) -> Result<Self::Blocks, Error> {
        let source_bounds = if bounds.0.is_empty() {
            self.bounds
        } else {
            self.source_bounds(bounds)?
        };

        self.source.blocks(source_bounds).await
    }

    async fn elements(self, bounds: Bounds) -> Result<Elements<Self::DType>, Error> {
        let source_bounds = if bounds.0.is_empty() {
            self.bounds
        } else {
            self.source_bounds(bounds)?
        };

        self.source.elements(source_bounds).await
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseSlice<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "slice of {:?} with bounds {:?}",
            self.source, self.bounds
        )
    }
}

#[derive(Clone)]
pub struct SparseTranspose<S> {
    source: S,
    permutation: Axes,
    shape: Shape,
}

impl<S> TensorInstance for SparseTranspose<S>
where
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<S> SparseInstance for SparseTranspose<S>
where
    S: SparseInstance,
    <S::CoordBlock as NDArrayTransform>::Transpose: NDArrayRead<DType = u64>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = S::ValueBlock;
    type Blocks = Pin<Box<dyn Stream<Item = Result<(Self::CoordBlock, Self::ValueBlock), Error>>>>;
    type DType = S::DType;

    async fn blocks(self, _bounds: Bounds) -> Result<Self::Blocks, Error> {
        todo!("support an order parameter in SparseInstance::blocks")
    }

    async fn elements(self, _bounds: Bounds) -> Result<Elements<Self::DType>, Error> {
        todo!("support an order parameter in SparseInstance::elements")
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseTranspose<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "transpose of {:?} with permutation {:?}",
            self.source, self.permutation
        )
    }
}

#[inline]
fn size_hint(size: u64) -> usize {
    size.try_into().ok().unwrap_or_else(|| usize::MAX)
}

#[inline]
fn unwrap_row<T>(mut row: Vec<Number>) -> (Coord, T)
where
    Number: CastInto<T> + CastInto<u64>,
{
    let n = row.pop().expect("n").cast_into();
    let coord = row.into_iter().map(|i| i.cast_into()).collect();
    (coord, n)
}

#[inline]
fn range_from_bounds(bounds: &Bounds, shape: &[u64]) -> Result<Range<usize, Number>, Error> {
    if bounds.0.is_empty() {
        return Ok(Range::default());
    } else if bounds.0.len() > shape.len() {
        return Err(Error::Bounds(format!(
            "invalid bounds for shape {:?}: {:?}",
            shape, bounds.0
        )));
    }

    let mut range = HashMap::new();

    for (x, bound) in bounds.0.iter().enumerate() {
        match bound {
            AxisBound::At(i) => {
                if *i >= shape[x] {
                    return Err(Error::Bounds(format!(
                        "invalid index for axis {}: {}",
                        x, i
                    )));
                }

                range.insert(x, b_table::ColumnRange::Eq(Number::from(*i)));
            }
            AxisBound::In(start, stop, step) => {
                if stop < start {
                    return Err(Error::Bounds(format!(
                        "invalid range for axis {}: [{}, {})",
                        x, start, stop
                    )));
                } else if *step != 1 {
                    return Err(Error::Bounds(format!(
                        "sparse tensor does not support stride {}",
                        step
                    )));
                } else if *stop > shape[x] {
                    return Err(Error::Bounds(format!(
                        "index {} is out of bounds for dimension {}",
                        stop, shape[x]
                    )));
                }

                let start = Bound::Included(Number::from(*start));
                let stop = Bound::Excluded(Number::from(*stop));
                range.insert(x, b_table::ColumnRange::In((start, stop)));
            }
            AxisBound::Of(indices) => {
                return Err(Error::Bounds(format!(
                    "sparse tensor does not support axis bound {:?}",
                    indices
                )));
            }
        }
    }

    Ok(range.into())
}
