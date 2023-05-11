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

use crate::{
    strides_for, validate_bounds, validate_order, Axes, AxisBound, Bounds, Coord, Error, Shape,
    Strides, TensorInstance,
};

use super::schema::{IndexSchema, Schema};
use super::stream;
use super::{Elements, Node};

#[async_trait]
pub trait SparseInstance: TensorInstance + fmt::Debug {
    type CoordBlock: NDArrayRead<DType = u64> + NDArrayMath + NDArrayTransform;
    type ValueBlock: NDArrayRead<DType = Self::DType>;
    type Blocks: Stream<Item = Result<(Self::CoordBlock, Self::ValueBlock), Error>>;
    type DType: CDatatype + DType;

    async fn blocks(self, bounds: Bounds, order: Axes) -> Result<Self::Blocks, Error>;

    async fn elements(self, bounds: Bounds, order: Axes) -> Result<Elements<Self::DType>, Error>;

    async fn filled_at(
        self,
        bounds: Bounds,
        axes: Axes,
    ) -> Result<stream::FilledAt<Elements<Self::DType>>, Error>
    where
        Self: Sized,
    {
        let ndim = self.ndim();

        let elided = (0..ndim).filter(|x| !axes.contains(x));

        let mut order = Vec::with_capacity(ndim);
        order.copy_from_slice(&axes);
        order.extend(elided);

        self.elements(bounds, order)
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

    async fn blocks(self, bounds: Bounds, order: Axes) -> Result<Self::Blocks, Error> {
        match self {
            Self::Table(table) => {
                let blocks = table.blocks(bounds, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Broadcast(broadcast) => {
                let blocks = broadcast.blocks(bounds, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Expand(expand) => {
                let blocks = expand.blocks(bounds, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Reshape(reshape) => {
                let blocks = reshape.blocks(bounds, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Slice(slice) => {
                let blocks = slice.blocks(bounds, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
        }
    }

    async fn elements(self, bounds: Bounds, order: Axes) -> Result<Elements<Self::DType>, Error> {
        array_dispatch!(self, this, this.elements(bounds, order).await)
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
        self.table.schema().shape()
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

    async fn blocks(self, bounds: Bounds, order: Axes) -> Result<Self::Blocks, Error> {
        let ndim = self.ndim();
        let elements = self.elements(bounds, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, bounds: Bounds, order: Axes) -> Result<Elements<Self::DType>, Error> {
        debug_assert!(validate_bounds(&bounds, self.shape()).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let range = range_from_bounds(&bounds, self.shape())?;
        let rows = self.table.rows(range, &order, false).await?;
        let elements = rows.map_ok(|row| unwrap_row(row)).map_err(Error::from);
        Ok(Box::pin(elements))
    }
}

impl<FE, T> fmt::Debug for SparseTable<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "sparse table with shape {:?}",
            self.table.schema().shape()
        )
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

    async fn blocks(self, bounds: Bounds, order: Axes) -> Result<Self::Blocks, Error> {
        let ndim = self.ndim();
        let elements = self.elements(bounds, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, bounds: Bounds, order: Axes) -> Result<Elements<Self::DType>, Error> {
        debug_assert!(validate_bounds(&bounds, self.shape()).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

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
            let source_elements = self.source.elements(source_bounds, order).await?;

            // TODO: write a range to a slice of a coordinate block instead
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

    async fn blocks(self, bounds: Bounds, order: Axes) -> Result<Self::Blocks, Error> {
        let ndim = self.ndim();
        let elements = self.elements(bounds, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, bounds: Bounds, order: Axes) -> Result<Elements<Self::DType>, Error> {
        debug_assert!(validate_bounds(&bounds, self.shape()).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let mut source_bounds = bounds;
        for x in self.axes.iter().rev().copied() {
            if x < source_bounds.0.len() {
                source_bounds.0.remove(x);
            }
        }

        let mut source_order = order;
        for x in self.axes.iter().rev().copied() {
            source_order.remove(x);
        }

        let ndim = self.ndim();
        let axes = self.axes;
        debug_assert_eq!(self.source.ndim() + 1, ndim);

        let source_elements = self.source.elements(source_bounds, source_order).await?;

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

    async fn blocks(self, bounds: Bounds, order: Axes) -> Result<Self::Blocks, Error> {
        debug_assert!(validate_bounds(&bounds, self.shape()).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let source_bounds = if bounds.0.is_empty() {
            Ok(bounds)
        } else {
            Err(Error::Bounds(format!(
                "cannot slice a reshaped sparse tensor (consider making a copy first)"
            )))
        }?;

        let source_order = if order
            .iter()
            .copied()
            .zip(0..self.ndim())
            .all(|(x, o)| x == o)
        {
            Ok(order)
        } else {
            Err(Error::Bounds(format!(
                "cannot transpose a reshaped sparse tensor (consider making a copy first)"
            )))
        }?;

        let source_ndim = self.source.ndim();
        let source_blocks = self.source.blocks(source_bounds, source_order).await?;
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

    async fn elements(self, bounds: Bounds, order: Axes) -> Result<Elements<Self::DType>, Error> {
        let ndim = self.shape.len();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let blocks = self.blocks(bounds, order).await?;

        let elements = blocks
            .map(move |result| {
                let (coords, values) = result?;
                let coords = coords.into_inner();
                let values = values.read(&queue)?.to_slice()?;
                let tuples = coords
                    .into_par_iter()
                    .chunks(ndim)
                    .zip(values.as_ref().into_par_iter().copied())
                    .map(Ok)
                    .collect::<Vec<_>>();

                Result::<_, Error>::Ok(futures::stream::iter(tuples))
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
        let mut source_bounds = Vec::with_capacity(self.source.ndim());
        let mut axis = 0;

        for source_bound in self.bounds.0.iter() {
            let source_bound = match source_bound {
                AxisBound::At(i) => AxisBound::At(*i),
                AxisBound::In(source_start, _stop, _step) => match &bounds.0[axis] {
                    AxisBound::At(i) => AxisBound::At(source_start + i),
                    AxisBound::In(start, stop, step) => {
                        let (source_start, source_stop, source_step) = (
                            source_start + start,
                            source_start + stop,
                            source_start * step,
                        );
                        AxisBound::In(source_start, source_stop, source_step)
                    }
                    AxisBound::Of(indices) => {
                        let indices = indices.iter().copied().map(|i| source_start + i).collect();
                        AxisBound::Of(indices)
                    }
                },
                AxisBound::Of(source_indices) => match &bounds.0[axis] {
                    AxisBound::At(i) => AxisBound::At(source_indices[*i as usize]),
                    AxisBound::In(start, stop, step) => {
                        let indices = source_indices[(*start as usize)..(*stop as usize)]
                            .iter()
                            .step_by(*step as usize)
                            .copied()
                            .collect();

                        AxisBound::Of(indices)
                    }
                    AxisBound::Of(indices) => {
                        let indices = indices
                            .iter()
                            .copied()
                            .map(|i| source_indices[i as usize])
                            .collect();

                        AxisBound::Of(indices)
                    }
                },
            };

            if !source_bound.is_index() {
                axis += 1;
            }

            source_bounds.push(source_bound);
        }

        source_bounds.extend(self.bounds.0.iter().skip(bounds.0.len()).cloned());

        Ok(Bounds(source_bounds))
    }

    fn source_order(&self, order: Axes) -> Result<Axes, Error> {
        debug_assert!(validate_order(&order, self.ndim()));

        let mut source_axes = Vec::with_capacity(self.ndim());
        for (x, bound) in self.bounds.0.iter().enumerate() {
            if !bound.is_index() {
                source_axes.push(x);
            }
        }

        Ok(order.into_iter().map(|x| source_axes[x]).collect())
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

    async fn blocks(self, bounds: Bounds, order: Axes) -> Result<Self::Blocks, Error> {
        debug_assert!(validate_bounds(&bounds, self.shape()).is_ok());

        let source_order = self.source_order(order)?;

        let source_bounds = if bounds.0.is_empty() {
            self.bounds
        } else {
            self.source_bounds(bounds)?
        };

        self.source.blocks(source_bounds, source_order).await
    }

    async fn elements(self, bounds: Bounds, order: Axes) -> Result<Elements<Self::DType>, Error> {
        debug_assert!(validate_bounds(&bounds, self.shape()).is_ok());

        let source_order = self.source_order(order)?;

        let source_bounds = if bounds.0.is_empty() {
            self.bounds
        } else {
            self.source_bounds(bounds)?
        };

        self.source.elements(source_bounds, source_order).await
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
    type CoordBlock = <S::CoordBlock as NDArrayTransform>::Transpose;
    type ValueBlock = S::ValueBlock;
    type Blocks = Pin<Box<dyn Stream<Item = Result<(Self::CoordBlock, Self::ValueBlock), Error>>>>;
    type DType = S::DType;

    async fn blocks(self, bounds: Bounds, order: Axes) -> Result<Self::Blocks, Error> {
        debug_assert!(validate_bounds(&bounds, self.shape()).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let bounds = bounds.normalize(self.shape());
        debug_assert_eq!(bounds.0.len(), self.ndim());

        let permutation = self.permutation;
        let mut source_bounds = Bounds::all(self.source.shape());
        for axis in 0..bounds.0.len() {
            source_bounds.0[permutation[axis]] = bounds.0[axis].clone();
        }

        let source_order = order.into_iter().map(|x| permutation[x]).collect();

        let source_blocks = self.source.blocks(source_bounds, source_order).await?;

        let blocks = source_blocks.map(move |result| {
            let (source_coords, values) = result?;
            let coords = source_coords.transpose(Some(permutation.to_vec()))?;
            Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(self, bounds: Bounds, order: Axes) -> Result<Elements<Self::DType>, Error> {
        let ndim = self.ndim();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let blocks = self.blocks(bounds, order).await?;

        let elements = blocks
            .map(move |result| {
                let (coords, values) = result?;
                let coords = coords.read(&queue)?.to_slice()?;
                let values = values.read(&queue)?.to_slice()?;
                let tuples = coords
                    .as_ref()
                    .into_par_iter()
                    .copied()
                    .chunks(ndim)
                    .zip(values.as_ref().into_par_iter().copied())
                    .map(Ok)
                    .collect::<Vec<_>>();

                Result::<_, Error>::Ok(futures::stream::iter(tuples))
            })
            .try_flatten();

        Ok(Box::pin(elements))
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
                range.insert(x, b_table::ColumnRange::Eq(Number::from(*i)));
            }
            AxisBound::In(start, stop, 1) => {
                let start = Bound::Included(Number::from(*start));
                let stop = Bound::Excluded(Number::from(*stop));
                range.insert(x, b_table::ColumnRange::In((start, stop)));
            }
            bound => {
                return Err(Error::Bounds(format!(
                    "sparse tensor does not support axis bound {:?}",
                    bound
                )));
            }
        }
    }

    Ok(range.into())
}
