use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Bound;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use b_table::TableLock;
use freqfs::DirLock;
use futures::future::TryFutureExt;
use futures::stream::{Stream, StreamExt, TryStreamExt};
use ha_ndarray::*;
use itertools::Itertools;
use number_general::{DType, Number, NumberCollator, NumberType};
use rayon::prelude::*;
use safecast::{AsType, CastInto};

use crate::{
    strides_for, validate_order, validate_transpose, Axes, AxisRange, Coord, Error, Range, Shape,
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

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, Error>;

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, Error>;

    async fn filled_at(
        self,
        range: Range,
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

        self.elements(range, order)
            .map_ok(|elements| stream::FilledAt::new(elements, axes, ndim))
            .await
    }
}

pub enum SparseAccess<FE, T> {
    Table(SparseTable<FE, T>),
    Broadcast(Box<SparseBroadcast<FE, T>>),
    BroadcastAxis(Box<SparseBroadcastAxis<Self>>),
    Expand(Box<SparseExpand<Self>>),
    Reshape(Box<SparseReshape<Self>>),
    Slice(Box<SparseSlice<Self>>),
    Transpose(Box<SparseTranspose<Self>>),
}

impl<FE, T> Clone for SparseAccess<FE, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Table(table) => Self::Table(table.clone()),
            Self::Broadcast(broadcast) => Self::Broadcast(broadcast.clone()),
            Self::BroadcastAxis(broadcast) => Self::BroadcastAxis(broadcast.clone()),
            Self::Expand(expand) => Self::Expand(expand.clone()),
            Self::Reshape(reshape) => Self::Reshape(reshape.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
            Self::Transpose(transpose) => Self::Transpose(transpose.clone()),
        }
    }
}

macro_rules! array_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::Table($var) => $call,
            Self::Broadcast($var) => $call,
            Self::BroadcastAxis($var) => $call,
            Self::Expand($var) => $call,
            Self::Reshape($var) => $call,
            Self::Slice($var) => $call,
            Self::Transpose($var) => $call,
        }
    };
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for SparseAccess<FE, T> {
    fn dtype(&self) -> NumberType {
        array_dispatch!(self, this, this.dtype())
    }

    fn shape(&self) -> &Shape {
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

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, Error> {
        match self {
            Self::Table(table) => {
                let blocks = table.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Broadcast(broadcast) => {
                let blocks = broadcast.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::BroadcastAxis(broadcast) => {
                let blocks = broadcast.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Expand(expand) => {
                let blocks = expand.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Reshape(reshape) => {
                let blocks = reshape.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Slice(slice) => {
                let blocks = slice.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Transpose(transpose) => {
                let blocks = transpose.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
        }
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, Error> {
        array_dispatch!(self, this, this.elements(range, order).await)
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

    fn shape(&self) -> &Shape {
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

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, Error> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, Error> {
        debug_assert!(self.shape().validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let range = table_range(&range)?;
        let rows = self.table.rows(range, &order, false).await?;
        let elements = rows.map_ok(|row| unwrap_row(row)).map_err(Error::from);
        Ok(Box::pin(elements))
    }
}

impl<FE, T> From<SparseTable<FE, T>> for SparseAccess<FE, T> {
    fn from(table: SparseTable<FE, T>) -> Self {
        Self::Table(table)
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

pub struct SparseBroadcast<FE, T> {
    shape: Shape,
    inner: SparseAccess<FE, T>,
}

impl<FE, T> Clone for SparseBroadcast<FE, T> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            inner: self.inner.clone(),
        }
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> SparseBroadcast<FE, T> {
    pub fn new<S>(source: S, shape: Shape) -> Result<Self, Error>
    where
        S: TensorInstance + Into<SparseAccess<FE, T>>,
    {
        let source_shape = source.shape().to_vec();
        let mut inner = source.into();

        let axes = (0..source_shape.len()).into_iter().rev();
        let dims = source_shape
            .into_iter()
            .rev()
            .zip(shape.iter().rev().copied());

        for (x, (dim, bdim)) in axes.zip(dims) {
            if dim == bdim {
                // no-op
            } else if dim == 1 {
                let broadcast_axis = SparseBroadcastAxis::new(inner, x, bdim)?;
                inner = SparseAccess::BroadcastAxis(Box::new(broadcast_axis));
            } else {
                return Err(Error::Bounds(format!(
                    "cannot broadcast {} into {} at axis {}",
                    dim, bdim, x
                )));
            }
        }

        Ok(Self { shape, inner })
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for SparseBroadcast<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseBroadcast<FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType,
    Number: CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, Error> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, mut range: Range, order: Axes) -> Result<Elements<Self::DType>, Error> {
        let ndim = self.ndim();
        let offset = ndim - self.inner.ndim();

        if offset == 0 {
            return self.inner.elements(range, order).await;
        }

        debug_assert!(self.shape.validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, ndim));

        let mut inner_range = Vec::with_capacity(self.inner.ndim());
        while range.len() > offset {
            inner_range.push(range.pop().expect("bound"));
        }

        let inner_order = if order
            .iter()
            .take(offset)
            .copied()
            .enumerate()
            .all(|(o, x)| x == o)
        {
            Ok(order.iter().skip(offset).cloned().collect::<Axes>())
        } else {
            Err(Error::Bounds(format!(
                "an outer broadcast of a sparse tensor does not support permutation"
            )))
        }?;

        let outer = range.into_iter().multi_cartesian_product();

        let inner = self.inner;
        let elements = futures::stream::iter(outer)
            .then(move |outer_coord| {
                let inner = inner.clone();
                let inner_range = inner_range.to_vec();
                let inner_order = inner_order.to_vec();

                async move {
                    let inner_elements = inner.elements(inner_range.into(), inner_order).await?;

                    let elements = inner_elements.map_ok(move |(inner_coord, value)| {
                        let mut coord = Vec::with_capacity(ndim);
                        coord.extend_from_slice(&outer_coord);
                        coord.extend(inner_coord);
                        (coord, value)
                    });

                    Result::<_, Error>::Ok(elements)
                }
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }
}

impl<FE, T> From<SparseBroadcast<FE, T>> for SparseAccess<FE, T> {
    fn from(accessor: SparseBroadcast<FE, T>) -> Self {
        Self::Broadcast(Box::new(accessor))
    }
}

impl<FE, T> fmt::Debug for SparseBroadcast<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "broadcasted sparse tensor with shape {:?}", self.shape)
    }
}

#[derive(Clone)]
pub struct SparseBroadcastAxis<S> {
    source: S,
    axis: usize,
    dim: u64,
    shape: Shape,
}

impl<S: TensorInstance + fmt::Debug> SparseBroadcastAxis<S> {
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
            shape: shape.into(),
        })
    }
}

impl<S: TensorInstance> TensorInstance for SparseBroadcastAxis<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance + Clone> SparseInstance for SparseBroadcastAxis<S> {
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = S::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, Error> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, range: Range, mut order: Axes) -> Result<Elements<Self::DType>, Error> {
        debug_assert!(self.shape.validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let axis = self.axis;
        let ndim = self.shape.len();

        let (source_range, dim) = if range.len() > axis {
            let bdim = match &range[axis] {
                AxisRange::At(i) if *i < self.dim => Ok(1),
                AxisRange::In(axis_range, 1) if axis_range.end > axis_range.start => {
                    Ok(axis_range.end - axis_range.start)
                }
                bound => Err(Error::Bounds(format!(
                    "invalid bound for axis {}: {:?}",
                    self.axis, bound
                ))),
            }?;

            let mut source_range = range;
            source_range[axis] = AxisRange::At(0);
            (source_range, bdim)
        } else {
            (range, self.dim)
        };

        let (source_order, inner_order) = if order
            .iter()
            .take(axis)
            .copied()
            .enumerate()
            .all(|(o, x)| x == o)
        {
            let mut inner_order = Axes::with_capacity(ndim - axis);

            while order.len() > axis {
                inner_order.push(order.pop().expect("axis"));
            }

            Ok((order, inner_order))
        } else {
            Err(Error::Bounds(format!(
                "an outer broadcast of a sparse tensor does not support permutation"
            )))
        }?;

        if self.axis == self.ndim() - 1 {
            let source_elements = self.source.elements(source_range, source_order).await?;

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
            let axes = (0..axis).into_iter().collect();
            let inner_range = source_range.iter().skip(axis).cloned().collect::<Vec<_>>();

            let source = self.source;
            let filled = source.clone().filled_at(source_range, axes).await?;

            let elements = filled
                .map(move |result| {
                    let outer_coord = result?;
                    debug_assert_eq!(outer_coord.len(), axis);
                    debug_assert_eq!(outer_coord.last(), Some(&0));

                    let inner_range = inner_range.to_vec();
                    let inner_order = inner_order.to_vec();

                    let prefix = outer_coord
                        .iter()
                        .copied()
                        .map(|i| AxisRange::At(i))
                        .collect();

                    let slice = SparseSlice::new(source.clone(), prefix)?;

                    let elements = futures::stream::iter(0..dim)
                        .then(move |i| {
                            let outer_coord = outer_coord[..axis - 1].to_vec();
                            let inner_range = inner_range.to_vec().into();
                            let inner_order = inner_order.to_vec();
                            let slice = slice.clone();

                            async move {
                                let inner_elements =
                                    slice.elements(inner_range, inner_order).await?;

                                let elements =
                                    inner_elements.map_ok(move |(inner_coord, value)| {
                                        let mut coord = Coord::with_capacity(ndim);
                                        coord.copy_from_slice(&outer_coord);
                                        coord.push(i);
                                        coord.extend(inner_coord);

                                        (coord, value)
                                    });

                                Result::<_, Error>::Ok(elements)
                            }
                        })
                        .try_flatten();

                    Result::<_, Error>::Ok(elements)
                })
                .try_flatten();

            Ok(Box::pin(elements))
        }
    }
}

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseBroadcastAxis<S>> for SparseAccess<FE, T> {
    fn from(broadcast: SparseBroadcastAxis<S>) -> Self {
        Self::BroadcastAxis(Box::new(SparseBroadcastAxis {
            source: broadcast.source.into(),
            axis: broadcast.axis,
            dim: broadcast.dim,
            shape: broadcast.shape,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseBroadcastAxis<S> {
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
    pub fn new(source: S, mut axes: Axes) -> Result<Self, Error> {
        axes.sort();

        let mut shape = source.shape().to_vec();
        for x in axes.iter().rev().copied() {
            shape.insert(x, 1);
        }

        if Some(source.ndim()) > axes.last().copied() {
            Ok(Self {
                source,
                shape: shape.into(),
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

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseExpand<S> {
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = S::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, Error> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, Error> {
        debug_assert!(self.shape.validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let mut source_range = range;
        for x in self.axes.iter().rev().copied() {
            if x < source_range.len() {
                source_range.remove(x);
            }
        }

        let mut source_order = order;
        for x in self.axes.iter().rev().copied() {
            source_order.remove(x);
        }

        let ndim = self.ndim();
        let axes = self.axes;
        debug_assert_eq!(self.source.ndim() + 1, ndim);

        let source_elements = self.source.elements(source_range, source_order).await?;

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

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseExpand<S>> for SparseAccess<FE, T> {
    fn from(expand: SparseExpand<S>) -> Self {
        Self::Expand(Box::new(SparseExpand {
            source: expand.source.into(),
            shape: expand.shape,
            axes: expand.axes,
        }))
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

impl<S: SparseInstance> SparseReshape<S> {
    pub fn new(source: S, shape: Shape) -> Result<Self, Error> {
        if source.shape().iter().product::<u64>() != shape.iter().product::<u64>() {
            return Err(Error::Bounds(format!(
                "cannot reshape {:?} into {:?}",
                source, shape
            )));
        }

        let source_strides = strides_for(source.shape(), source.ndim());
        let strides = strides_for(&shape, shape.len());

        Ok(Self {
            source,
            source_strides,
            shape,
            strides,
        })
    }
}

impl<S: TensorInstance> TensorInstance for SparseReshape<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseReshape<S> {
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = S::ValueBlock;
    type Blocks = Pin<Box<dyn Stream<Item = Result<(Self::CoordBlock, Self::ValueBlock), Error>>>>;
    type DType = S::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, Error> {
        debug_assert!(self.shape.validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let source_range = if range.is_empty() {
            Ok(range)
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
        let source_blocks = self.source.blocks(source_range, source_order).await?;
        let source_strides =
            ArrayBase::<Arc<Vec<_>>>::new(vec![source_ndim], Arc::new(self.source_strides))?;

        let ndim = self.shape.len();
        let strides = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], Arc::new(self.strides))?;
        let shape = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], Arc::new(self.shape.into()))?;

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

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, Error> {
        let ndim = self.shape.len();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let blocks = self.blocks(range, order).await?;

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

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseReshape<S>> for SparseAccess<FE, T> {
    fn from(reshape: SparseReshape<S>) -> Self {
        Self::Reshape(Box::new(SparseReshape {
            source: reshape.source.into(),
            source_strides: reshape.source_strides,
            shape: reshape.shape,
            strides: reshape.strides,
        }))
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
    range: Range,
    shape: Shape,
}

impl<S> SparseSlice<S>
where
    S: TensorInstance + fmt::Debug,
{
    pub fn new(source: S, range: Range) -> Result<Self, Error> {
        source.shape().validate_range(&range)?;

        let mut shape = Vec::with_capacity(source.ndim());
        for (x, bound) in range.iter().enumerate() {
            match bound {
                AxisRange::At(_) => {} // no-op
                AxisRange::In(axis_range, 1) => {
                    shape.push(axis_range.end - axis_range.start);
                }
                axis_bound => {
                    return Err(Error::Bounds(format!(
                        "invalid bound for sparse tensor axis {}: {:?}",
                        x, axis_bound
                    )));
                }
            }
        }

        shape.extend_from_slice(&source.shape()[range.len()..]);

        Ok(Self {
            source,
            range,
            shape: shape.into(),
        })
    }

    fn source_range(&self, range: Range) -> Result<Range, Error> {
        let mut source_range = Vec::with_capacity(self.source.ndim());
        let mut axis = 0;

        for axis_range in self.range.iter() {
            let axis_range = match axis_range {
                AxisRange::At(i) => AxisRange::At(*i),
                AxisRange::In(source_range, source_step) => match &range[axis] {
                    AxisRange::At(i) => {
                        debug_assert!(source_range.start + (i * source_step) < source_range.end);
                        AxisRange::At(source_range.start + (i * source_step))
                    }
                    AxisRange::In(axis_range, step) => {
                        debug_assert!(source_range.start + axis_range.start <= source_range.end);
                        debug_assert!(source_range.start + axis_range.end <= source_range.end);

                        let (source_start, source_end, source_step) = (
                            axis_range.start + source_range.start,
                            axis_range.end + source_range.start,
                            step * source_step,
                        );

                        AxisRange::In(source_start..source_end, source_step)
                    }
                    AxisRange::Of(indices) => {
                        let indices = indices
                            .iter()
                            .copied()
                            .map(|i| source_range.start + i)
                            .collect::<Vec<u64>>();
                        debug_assert!(indices.iter().copied().all(|i| i < source_range.end));
                        AxisRange::Of(indices)
                    }
                },
                AxisRange::Of(source_indices) => match &range[axis] {
                    AxisRange::At(i) => AxisRange::At(source_indices[*i as usize]),
                    AxisRange::In(axis_range, step) => {
                        debug_assert!(axis_range.start as usize <= source_indices.len());
                        debug_assert!(axis_range.end as usize <= source_indices.len());

                        let indices = source_indices
                            [(axis_range.start as usize)..(axis_range.end as usize)]
                            .iter()
                            .step_by(*step as usize)
                            .copied()
                            .collect();

                        AxisRange::Of(indices)
                    }
                    AxisRange::Of(indices) => {
                        let indices = indices
                            .iter()
                            .copied()
                            .map(|i| source_indices[i as usize])
                            .collect();

                        AxisRange::Of(indices)
                    }
                },
            };

            if !axis_range.is_index() {
                axis += 1;
            }

            source_range.push(axis_range);
        }

        source_range.extend(self.range.iter().skip(range.len()).cloned());

        Ok(source_range.into())
    }

    fn source_order(&self, order: Axes) -> Result<Axes, Error> {
        debug_assert!(validate_order(&order, self.ndim()));

        let mut source_axes = Vec::with_capacity(self.ndim());
        for (x, bound) in self.range.iter().enumerate() {
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

    fn shape(&self) -> &Shape {
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

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, Error> {
        debug_assert!(self.shape.validate_range(&range).is_ok());

        let source_order = self.source_order(order)?;

        let source_range = if range.is_empty() {
            self.range
        } else {
            self.source_range(range)?
        };

        self.source.blocks(source_range, source_order).await
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, Error> {
        debug_assert!(self.shape.validate_range(&range).is_ok());

        let source_order = self.source_order(order)?;

        let source_range = if range.is_empty() {
            self.range
        } else {
            self.source_range(range)?
        };

        self.source.elements(source_range, source_order).await
    }
}

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseSlice<S>> for SparseAccess<FE, T> {
    fn from(slice: SparseSlice<S>) -> Self {
        Self::Slice(Box::new(SparseSlice {
            source: slice.source.into(),
            range: slice.range,
            shape: slice.shape,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseSlice<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "slice of {:?} with range {:?}", self.source, self.range)
    }
}

#[derive(Clone)]
pub struct SparseTranspose<S> {
    source: S,
    permutation: Axes,
    shape: Shape,
}

impl<S: SparseInstance> SparseTranspose<S> {
    pub fn new(source: S, permutation: Option<Axes>) -> Result<Self, Error> {
        let permutation = validate_transpose(permutation, source.shape())?;

        let shape = permutation
            .iter()
            .copied()
            .map(|x| source.shape()[x])
            .collect();

        Ok(Self {
            source,
            permutation,
            shape,
        })
    }
}

impl<S> TensorInstance for SparseTranspose<S>
where
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
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

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, Error> {
        debug_assert!(self.shape.validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let range = range.normalize(self.shape());
        debug_assert_eq!(range.len(), self.ndim());

        let permutation = self.permutation;
        let mut source_range = Range::all(self.source.shape());
        for axis in 0..range.len() {
            source_range[permutation[axis]] = range[axis].clone();
        }

        let source_order = order.into_iter().map(|x| permutation[x]).collect();

        let source_blocks = self.source.blocks(source_range, source_order).await?;

        let blocks = source_blocks.map(move |result| {
            let (source_coords, values) = result?;
            let coords = source_coords.transpose(Some(permutation.to_vec()))?;
            Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, Error> {
        let ndim = self.ndim();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let blocks = self.blocks(range, order).await?;

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

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseTranspose<S>> for SparseAccess<FE, T> {
    fn from(transpose: SparseTranspose<S>) -> Self {
        Self::Transpose(Box::new(SparseTranspose {
            source: transpose.source.into(),
            permutation: transpose.permutation,
            shape: transpose.shape,
        }))
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
fn table_range(range: &Range) -> Result<b_table::Range<usize, Number>, Error> {
    if range == &Range::default() {
        return Ok(b_table::Range::default());
    }

    let mut table_range = HashMap::new();

    for (x, bound) in range.iter().enumerate() {
        match bound {
            AxisRange::At(i) => {
                table_range.insert(x, b_table::ColumnRange::Eq(Number::from(*i)));
            }
            AxisRange::In(axis_range, 1) => {
                let start = Bound::Included(Number::from(axis_range.start));
                let stop = Bound::Excluded(Number::from(axis_range.end));
                table_range.insert(x, b_table::ColumnRange::In((start, stop)));
            }
            bound => {
                return Err(Error::Bounds(format!(
                    "sparse tensor does not support axis bound {:?}",
                    bound
                )));
            }
        }
    }

    Ok(table_range.into())
}
