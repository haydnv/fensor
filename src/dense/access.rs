use std::marker::PhantomData;
use std::pin::Pin;
use std::{fmt, io};

use async_trait::async_trait;
use destream::de;
use freqfs::{DirLock, FileLoad, FileReadGuardOwned};
use futures::{stream, Stream, StreamExt, TryFutureExt, TryStreamExt};
use ha_ndarray::*;
use number_general::{DType, NumberClass, NumberInstance, NumberType};
use safecast::AsType;

use crate::{
    validate_transpose, Axes, AxisRange, Error, Range, Shape, TensorInstance, IDEAL_BLOCK_SIZE,
};

type BlockShape = ha_ndarray::Shape;
type BlockStream<Block> = Pin<Box<dyn Stream<Item = Result<Block, Error>>>>;

#[async_trait]
pub trait DenseInstance: TensorInstance + fmt::Debug + Send + Sync + 'static {
    type Block: NDArrayRead<DType = Self::DType> + NDArrayTransform + Into<Array<Self::DType>>;
    type DType: CDatatype + DType;

    fn block_size(&self) -> usize;

    async fn read_block(&self, block_id: u64) -> Result<Self::Block, Error>;

    async fn into_blocks(self) -> Result<BlockStream<Self::Block>, Error>;
}

#[async_trait]
impl<T: DenseInstance> DenseInstance for Box<T> {
    type Block = T::Block;
    type DType = T::DType;

    fn block_size(&self) -> usize {
        (&**self).block_size()
    }

    async fn read_block(&self, block_id: u64) -> Result<Self::Block, Error> {
        (**self).read_block(block_id).await
    }

    async fn into_blocks(self) -> Result<BlockStream<Self::Block>, Error> {
        (*self).into_blocks().await
    }
}

pub enum DenseAccess<FE, T> {
    File(DenseFile<FE, T>),
    Broadcast(Box<DenseBroadcast<Self>>),
    Reshape(Box<DenseReshape<Self>>),
    Slice(Box<DenseSlice<Self>>),
    Transpose(Box<DenseTranspose<Self>>),
}

impl<FE, T> Clone for DenseAccess<FE, T> {
    fn clone(&self) -> Self {
        match self {
            Self::File(file) => Self::File(file.clone()),
            Self::Broadcast(broadcast) => Self::Broadcast(broadcast.clone()),
            Self::Reshape(reshape) => Self::Reshape(reshape.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
            Self::Transpose(transpose) => Self::Transpose(transpose.clone()),
        }
    }
}

macro_rules! array_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::File($var) => $call,
            Self::Broadcast($var) => $call,
            Self::Reshape($var) => $call,
            Self::Slice($var) => $call,
            Self::Transpose($var) => $call,
        }
    };
}

impl<FE, T> TensorInstance for DenseAccess<FE, T>
where
    FE: Send + Sync + 'static,
    T: DType + Send + Sync + 'static,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        array_dispatch!(self, this, this.shape())
    }
}

#[async_trait]
impl<FE, T> DenseInstance for DenseAccess<FE, T>
where
    FE: FileLoad + AsType<Buffer<T>> + Send + Sync,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        array_dispatch!(self, this, this.block_size())
    }

    async fn read_block(&self, block_id: u64) -> Result<Self::Block, Error> {
        array_dispatch!(
            self,
            this,
            this.read_block(block_id).map_ok(Array::from).await
        )
    }

    async fn into_blocks(self) -> Result<BlockStream<Self::Block>, Error> {
        match self {
            Self::File(file) => Ok(Box::pin(file.into_blocks().await?.map_ok(Array::from))),
            Self::Broadcast(broadcast) => {
                Ok(Box::pin(broadcast.into_blocks().await?.map_ok(Array::from)))
            }
            Self::Reshape(reshape) => {
                Ok(Box::pin(reshape.into_blocks().await?.map_ok(Array::from)))
            }
            Self::Slice(slice) => Ok(Box::pin(slice.into_blocks().await?.map_ok(Array::from))),
            Self::Transpose(transpose) => {
                Ok(Box::pin(transpose.into_blocks().await?.map_ok(Array::from)))
            }
        }
    }
}

impl<FE, T> fmt::Debug for DenseAccess<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        array_dispatch!(self, this, this.fmt(f))
    }
}

pub struct DenseFile<FE, T> {
    dir: DirLock<FE>,
    block_map: ArrayBase<Vec<u64>>,
    block_size: usize,
    shape: Shape,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for DenseFile<FE, T> {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            block_map: self.block_map.clone(),
            block_size: self.block_size,
            shape: self.shape.clone(),
            dtype: PhantomData,
        }
    }
}

impl<FE, T> DenseFile<FE, T>
where
    FE: FileLoad + AsType<Buffer<T>> + Send + Sync,
    T: CDatatype + DType + NumberInstance,
{
    pub async fn constant(dir: DirLock<FE>, shape: Shape, value: T) -> Result<Self, Error> {
        shape.validate()?;

        let ndim = shape.len();
        let size = shape.iter().product();

        let ideal_block_size = IDEAL_BLOCK_SIZE as u64;
        let (block_size, num_blocks) = if size < (2 * ideal_block_size) {
            (size as usize, 1)
        } else if ndim == 1 && size % ideal_block_size == 0 {
            (IDEAL_BLOCK_SIZE, (size / ideal_block_size) as usize)
        } else if ndim == 1
            || (shape.iter().rev().take(2).product::<u64>() > (2 * ideal_block_size))
        {
            let num_blocks = div_ceil(size, ideal_block_size) as usize;
            (IDEAL_BLOCK_SIZE, num_blocks as usize)
        } else {
            let matrix_size = shape.iter().rev().take(2).product::<u64>();
            let block_size = ideal_block_size + (matrix_size - (ideal_block_size % matrix_size));
            let num_blocks = div_ceil(size, ideal_block_size);
            (block_size as usize, num_blocks as usize)
        };

        debug_assert!(block_size > 0);

        {
            let dtype_size = T::dtype().size();

            let mut dir = dir.write().await;

            for block_id in 0..num_blocks {
                dir.create_file(
                    block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )?;
            }

            let last_block_id = num_blocks - 1;
            if size % block_size as u64 == 0 {
                dir.create_file(
                    last_block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )
            } else {
                dir.create_file(
                    last_block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )
            }?;
        };

        let block_axis = block_axis_for(&shape, block_size);
        let map_shape = shape
            .iter()
            .take(block_axis)
            .copied()
            .map(|dim| dim as usize)
            .collect();

        let block_map =
            ArrayBase::<Vec<_>>::new(map_shape, (0u64..num_blocks as u64).into_iter().collect())?;

        Ok(Self {
            dir,
            block_map,
            block_size,
            shape,
            dtype: PhantomData,
        })
    }
}

impl<FE, T> TensorInstance for DenseFile<FE, T>
where
    FE: Send + Sync + 'static,
    T: DType + Send + Sync + 'static,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<FE, T> DenseInstance for DenseFile<FE, T>
where
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType + 'static,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Block = ArrayBase<FileReadGuardOwned<FE, Buffer<T>>>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, block_id: u64) -> Result<Self::Block, Error> {
        let dir = self.dir.read().await;
        let file = dir.get_file(&block_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("dense tensor block {}", block_id),
            )
        })?;

        let buffer = file.read_owned().await?;
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, buffer.len());
        ArrayBase::<FileReadGuardOwned<FE, Buffer<T>>>::new(block_shape, buffer)
            .map_err(Error::from)
    }

    async fn into_blocks(self) -> Result<BlockStream<Self::Block>, Error> {
        let shape = self.shape;
        let block_axis = block_axis_for(&shape, self.block_size);
        let dir = self.dir.into_read().await;

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                dir.get_file(&block_id).cloned().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("dense tensor block {}", block_id),
                    )
                    .into()
                })
            })
            .map_ok(|block| block.into_read())
            .try_buffered(num_cpus::get())
            .map(move |result| {
                let buffer = result?;
                let block_shape = block_shape_for(block_axis, &shape, buffer.len());
                ArrayBase::<FileReadGuardOwned<FE, Buffer<T>>>::new(block_shape, buffer)
                    .map_err(Error::from)
            });

        Ok(Box::pin(blocks))
    }
}

impl<FE, T> From<DenseFile<FE, T>> for DenseAccess<FE, T> {
    fn from(file: DenseFile<FE, T>) -> Self {
        Self::File(file)
    }
}

impl<FE, T> fmt::Debug for DenseFile<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "dense tensor with shape {:?}", self.shape)
    }
}

#[derive(Clone)]
pub struct DenseBroadcast<S> {
    source: S,
    shape: Shape,
    block_map: ArrayBase<Vec<u64>>,
    block_size: usize,
}

impl<S: DenseInstance> DenseBroadcast<S> {
    pub fn new(source: S, shape: Shape) -> Result<Self, Error> {
        let num_blocks = div_ceil(source.size(), source.block_size() as u64);
        let block_axis = block_axis_for(source.shape(), source.block_size());
        let source_block_shape = block_shape_for(block_axis, source.shape(), source.block_size());

        let mut block_shape = BlockShape::with_capacity(source_block_shape.len());
        block_shape.push(source_block_shape[0]);
        block_shape.extend(
            shape
                .iter()
                .rev()
                .take(source_block_shape.len() - 1)
                .rev()
                .copied()
                .map(|dim| dim as usize),
        );

        let block_size = block_shape.iter().product();

        let mut block_map_shape = BlockShape::with_capacity(source.ndim());
        block_map_shape.extend(
            shape
                .iter()
                .take(block_axis)
                .copied()
                .map(|dim| dim as usize),
        );
        block_map_shape.push(shape[block_axis] as usize / source_block_shape[0]);

        let block_map =
            ArrayBase::<Vec<_>>::new(block_map_shape, (0..num_blocks).into_iter().collect())?;

        Ok(Self {
            source,
            shape,
            block_map,
            block_size,
        })
    }
}

impl<S: TensorInstance> TensorInstance for DenseBroadcast<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseBroadcast<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Broadcast:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Broadcast;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, block_id: u64) -> Result<Self::Block, Error> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;
        let block_axis = block_axis_for(&self.shape, self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, self.block_size);
        let source_block = self.source.read_block(source_block_id).await?;
        source_block.broadcast(block_shape).map_err(Error::from)
    }

    async fn into_blocks(self) -> Result<BlockStream<Self::Block>, Error> {
        let block_axis = block_axis_for(&self.shape, self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, self.block_size);

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                let source = self.source.clone();
                async move { source.read_block(block_id).await }
            })
            .buffered(num_cpus::get())
            .map(move |result| {
                let source_block = result?;
                source_block
                    .broadcast(block_shape.to_vec())
                    .map_err(Error::from)
            });

        Ok(Box::pin(blocks))
    }
}

impl<FE, T, S: Into<DenseAccess<FE, T>>> From<DenseBroadcast<S>> for DenseAccess<FE, T> {
    fn from(broadcast: DenseBroadcast<S>) -> Self {
        Self::Broadcast(Box::new(DenseBroadcast {
            source: broadcast.source.into(),
            shape: broadcast.shape,
            block_map: broadcast.block_map,
            block_size: broadcast.block_size,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseBroadcast<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "broadcast of {:?} into {:?}", self.source, self.shape)
    }
}

#[derive(Clone)]
pub struct DenseReshape<S> {
    source: S,
    shape: Shape,
}

impl<S: DenseInstance> DenseReshape<S> {
    pub fn new(source: S, shape: Shape) -> Result<Self, Error> {
        if shape.iter().product::<u64>() == source.size() {
            Ok(Self { source, shape })
        } else {
            Err(Error::Bounds(format!(
                "cannot reshape {:?} into {:?}",
                source, shape
            )))
        }
    }
}

impl<S: TensorInstance> TensorInstance for DenseReshape<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }
}

#[async_trait]
impl<S: DenseInstance> DenseInstance for DenseReshape<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Reshape:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Reshape;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, block_id: u64) -> Result<Self::Block, Error> {
        let block_axis = block_axis_for(self.shape(), self.block_size());
        let mut block_shape = block_shape_for(block_axis, self.shape(), self.block_size());

        let block = self.source.read_block(block_id).await?;

        if block.size() < self.block_size() {
            // this must be the trailing block
            let axis_dim = self.block_size() / block_shape.iter().skip(1).product::<usize>();
            block_shape[0] = axis_dim;
        }

        block.reshape(block_shape).map_err(Error::from)
    }

    async fn into_blocks(self) -> Result<BlockStream<Self::Block>, Error> {
        let block_size = self.block_size();
        let block_axis = block_axis_for(self.shape(), block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), block_size);

        let source_blocks = self.source.into_blocks().await?;
        let blocks = source_blocks.map(move |result| {
            let block = result?;
            let mut block_shape = block_shape.to_vec();

            if block.size() < block_size {
                // this must be the trailing block
                let axis_dim = block_size / block_shape.iter().skip(1).product::<usize>();
                block_shape[0] = axis_dim;
            }

            block.reshape(block_shape).map_err(Error::from)
        });

        Ok(Box::pin(blocks))
    }
}

impl<FE, T, S: Into<DenseAccess<FE, T>>> From<DenseReshape<S>> for DenseAccess<FE, T> {
    fn from(reshape: DenseReshape<S>) -> Self {
        Self::Reshape(Box::new(DenseReshape {
            source: reshape.source.into(),
            shape: reshape.shape,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseReshape<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "reshape {:?} into {:?}", self.source, self.shape)
    }
}

#[derive(Clone)]
pub struct DenseSlice<S> {
    source: S,
    range: Range,
    shape: Shape,
    block_map: ArrayBase<Vec<u64>>,
    block_size: usize,
}

impl<S: DenseInstance> DenseSlice<S> {
    pub fn new(source: S, range: Range) -> Result<Self, Error> {
        source.shape().validate_range(&range)?;

        let block_axis = block_axis_for(source.shape(), source.block_size());
        let block_shape = block_shape_for(block_axis, source.shape(), source.block_size());
        let num_blocks = div_ceil(
            source.size() as u64,
            block_shape.iter().product::<usize>() as u64,
        ) as usize;

        let block_map_shape = source
            .shape()
            .iter()
            .take(block_axis)
            .copied()
            .map(|dim| dim.try_into().map_err(Error::Index))
            .collect::<Result<_, _>>()?;

        let block_map = ArrayBase::<Vec<_>>::new(
            block_map_shape,
            (0..num_blocks as u64).into_iter().collect(),
        )?;

        let mut block_map_bounds = Vec::with_capacity(block_axis + 1);
        for axis_range in range.iter().take(block_axis).cloned() {
            let bound = axis_range.try_into()?;
            block_map_bounds.push(bound);
        }

        if range.len() > block_axis {
            let bound = match &range[block_axis] {
                AxisRange::At(i) => {
                    let stride = block_map.shape().last().expect("stride");
                    let i = usize::try_from(*i).map_err(Error::Index)? / stride;
                    ha_ndarray::AxisBound::At(i)
                }
                AxisRange::In(axis_range, _step) => {
                    let stride = block_shape[0];
                    let start = usize::try_from(axis_range.start).map_err(Error::Index)? / stride;
                    let stop = usize::try_from(axis_range.end).map_err(Error::Index)? / stride;
                    ha_ndarray::AxisBound::In(start, stop, 1)
                }
                AxisRange::Of(indices) => {
                    let stride = block_map.shape().last().expect("stride");
                    let indices = indices
                        .iter()
                        .copied()
                        .map(|i| usize::try_from(i).map(|i| i / stride).map_err(Error::Index))
                        .collect::<Result<Vec<usize>, Error>>()?;

                    ha_ndarray::AxisBound::Of(indices)
                }
            };

            block_map_bounds.push(bound);
        }

        let block_map = block_map.slice(block_map_bounds)?;
        let block_map = ArrayBase::<Vec<u64>>::copy(&block_map)?;

        let mut shape = Vec::with_capacity(source.ndim());
        for (bound, dim) in range.iter().zip(source.shape().iter()) {
            match bound {
                AxisRange::At(i) => {
                    if i > dim {
                        return Err(Error::Bounds(format!(
                            "index {} is out of bounds for dimension {}",
                            i, dim
                        )));
                    }
                }
                AxisRange::In(axis_range, step) => {
                    if axis_range.start < axis_range.end {
                        shape.push((axis_range.end - axis_range.start) / step);
                    }
                }
                AxisRange::Of(indices) => {
                    if indices.iter().all(|i| i < dim) {
                        shape.push(indices.len() as u64);
                    } else {
                        return Err(Error::Bounds(format!(
                            "indices {:?} are out of bounds for dimension {}",
                            indices, dim
                        )));
                    }
                }
            }
        }

        let block_size = shape.iter().product::<u64>() as usize / num_blocks;

        Ok(Self {
            source,
            range,
            shape: shape.into(),
            block_map,
            block_size,
        })
    }
}

impl<S: TensorInstance> TensorInstance for DenseSlice<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseSlice<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Slice;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, block_id: u64) -> Result<Self::Block, Error> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;

        let block_axis = block_axis_for(&self.shape, self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, self.block_size);

        let local_bound = match ha_ndarray::AxisBound::try_from(self.range[block_axis].clone())? {
            ha_ndarray::AxisBound::At(i) => ha_ndarray::AxisBound::At(i),
            ha_ndarray::AxisBound::In(start, stop, step) => {
                let stride = block_shape[0];

                if source_block_id == 0 {
                    ha_ndarray::AxisBound::In(start, stride, step)
                } else if source_block_id == self.block_map.size() as u64 - 1 {
                    ha_ndarray::AxisBound::In(stop - (stop % stride), stop, step)
                } else {
                    let start = source_block_id as usize * stride;
                    ha_ndarray::AxisBound::In(start, start + stride, step)
                }
            }
            ha_ndarray::AxisBound::Of(indices) => {
                if source_block_id < indices.len() as u64 {
                    let i = indices[source_block_id as usize] as usize;
                    ha_ndarray::AxisBound::At(i)
                } else {
                    return Err(Error::Bounds(format!(
                        "block id {} is out of range",
                        block_id
                    )));
                }
            }
        };

        let mut block_bounds = Vec::with_capacity(self.ndim());
        for bound in self.range.iter().take(block_axis).cloned() {
            block_bounds.push(bound.try_into()?);
        }

        if block_bounds.is_empty() {
            block_bounds.push(local_bound);
        } else {
            block_bounds[0] = local_bound;
        }

        let source_block = self.source.read_block(source_block_id).await?;
        source_block.slice(block_bounds).map_err(Error::from)
    }

    async fn into_blocks(self) -> Result<BlockStream<Self::Block>, Error> {
        let block_map = self.block_map;
        let range = self.range;
        let ndim = self.shape.len();
        let source = self.source;

        let block_axis = block_axis_for(&self.shape, self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, self.block_size);

        let local_bounds = match ha_ndarray::AxisBound::try_from(range[block_axis].clone())? {
            ha_ndarray::AxisBound::At(i) => {
                debug_assert_eq!(block_map.size(), 1);
                vec![ha_ndarray::AxisBound::At(i)]
            }
            ha_ndarray::AxisBound::In(start, stop, step) => {
                let stride = block_shape[0];

                if block_map.size() == 1 {
                    vec![ha_ndarray::AxisBound::In(start, stop, step)]
                } else {
                    let mut local_bounds = Vec::with_capacity(block_map.size());
                    local_bounds.push(ha_ndarray::AxisBound::In(start, stride, step));

                    for i in 0..(block_map.size() - 2) {
                        let start = stride * i;
                        local_bounds.push(ha_ndarray::AxisBound::In(start, start + stride, step));
                    }

                    local_bounds.push(ha_ndarray::AxisBound::In(
                        stop - (stop % stride),
                        stop,
                        step,
                    ));

                    local_bounds
                }
            }
            ha_ndarray::AxisBound::Of(indices) => {
                indices.into_iter().map(ha_ndarray::AxisBound::At).collect()
            }
        };

        let mut block_bounds = Vec::<ha_ndarray::AxisBound>::with_capacity(ndim);
        for bound in range.iter().skip(block_axis).cloned() {
            block_bounds.push(bound.try_into()?);
        }

        debug_assert_eq!(block_map.size(), local_bounds.len());
        let blocks = stream::iter(block_map.into_inner().into_iter().zip(local_bounds))
            .map(move |(block_id, local_bound)| {
                let mut block_bounds = block_bounds.to_vec();
                let source = source.clone();

                async move {
                    let block = source.read_block(block_id).await?;

                    if block_bounds.is_empty() {
                        block_bounds.push(local_bound);
                    } else {
                        block_bounds[0] = local_bound;
                    }

                    block.slice(block_bounds).map_err(Error::from)
                }
            })
            .buffered(num_cpus::get());

        Ok(Box::pin(blocks))
    }
}

impl<FE, T, S: Into<DenseAccess<FE, T>>> From<DenseSlice<S>> for DenseAccess<FE, T> {
    fn from(slice: DenseSlice<S>) -> Self {
        Self::Slice(Box::new(DenseSlice {
            source: slice.source.into(),
            range: slice.range,
            shape: slice.shape,
            block_map: slice.block_map,
            block_size: slice.block_size,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseSlice<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "slice {:?} from {:?}", self.range, self.source)
    }
}

#[derive(Clone)]
pub struct DenseTranspose<S> {
    source: S,
    shape: Shape,
    permutation: Axes,
    block_map: ArrayBase<Vec<u64>>,
}

impl<S: DenseInstance> DenseTranspose<S> {
    pub fn new(source: S, permutation: Option<Axes>) -> Result<Self, Error> {
        let permutation = validate_transpose(permutation, source.shape())?;

        let shape = permutation
            .iter()
            .copied()
            .map(|x| source.shape()[x])
            .collect();

        let num_blocks = div_ceil(source.size(), source.block_size() as u64);
        let block_axis = block_axis_for(source.shape(), source.block_size());

        let map_shape = source
            .shape()
            .iter()
            .take(block_axis)
            .copied()
            .map(|dim| dim as usize)
            .collect();

        let (map_axes, permutation) = permutation.split_at(block_axis);

        if map_axes.iter().copied().any(|x| x >= block_axis)
            || permutation.iter().copied().any(|x| x <= block_axis)
        {
            return Err(Error::Bounds(format!(
                "cannot transpose axes {:?} of {:?} without copying",
                permutation, source
            )));
        }

        let block_map = ArrayBase::<Vec<_>>::new(map_shape, (0..num_blocks).into_iter().collect())?;
        let block_map = block_map.transpose(Some(map_axes.to_vec()))?;
        let block_map = ArrayBase::<Vec<_>>::copy(&block_map)?;

        Ok(Self {
            source,
            shape,
            permutation: permutation.to_vec(),
            block_map,
        })
    }
}

impl<S: TensorInstance> TensorInstance for DenseTranspose<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseTranspose<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Transpose:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Transpose;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, block_id: u64) -> Result<Self::Block, Error> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;
        let block = self.source.read_block(source_block_id).await?;
        block
            .transpose(Some(self.permutation.to_vec()))
            .map_err(Error::from)
    }

    async fn into_blocks(self) -> Result<BlockStream<Self::Block>, Error> {
        let permutation = self.permutation;

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                let source = self.source.clone();
                async move { source.read_block(block_id).await }
            })
            .buffered(num_cpus::get())
            .map(move |result| {
                let block = result?;

                block
                    .transpose(Some(permutation.to_vec()))
                    .map_err(Error::from)
            });

        Ok(Box::pin(blocks))
    }
}

impl<FE, T, S: Into<DenseAccess<FE, T>>> From<DenseTranspose<S>> for DenseAccess<FE, T> {
    fn from(transpose: DenseTranspose<S>) -> Self {
        Self::Transpose(Box::new(DenseTranspose {
            source: transpose.source.into(),
            shape: transpose.shape,
            permutation: transpose.permutation.into(),
            block_map: transpose.block_map,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseTranspose<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "transpose axes {:?} of {:?}",
            self.permutation, self.source
        )
    }
}

#[inline]
fn block_axis_for(shape: &[u64], block_size: usize) -> usize {
    debug_assert!(!shape.is_empty());
    debug_assert!(shape.iter().copied().all(|dim| dim > 0));
    debug_assert!(shape.iter().product::<u64>() >= block_size as u64);

    let mut block_ndim = 1;
    let mut size = 1;
    for dim in shape.iter().rev() {
        size *= dim;

        if size > block_size as u64 {
            break;
        } else {
            block_ndim += 1;
        }
    }

    shape.len() - block_ndim
}

#[inline]
fn block_shape_for(axis: usize, shape: &[u64], block_size: usize) -> BlockShape {
    if axis == shape.len() - 1 {
        vec![block_size]
    } else {
        let axis_dim = (shape.iter().skip(axis).product::<u64>() / block_size as u64) as usize;
        debug_assert_eq!(block_size % axis_dim, 0);

        let mut block_shape = BlockShape::with_capacity(shape.len() - axis + 1);
        block_shape.push(axis_dim);
        block_shape.extend(shape.iter().skip(axis).copied().map(|dim| dim as usize));

        debug_assert!(!block_shape.is_empty());

        block_shape
    }
}

#[inline]
fn div_ceil(num: u64, denom: u64) -> u64 {
    if num % denom == 0 {
        num / denom
    } else {
        (num / denom) + 1
    }
}

#[inline]
fn source_block_id_for(block_map: &ArrayBase<Vec<u64>>, block_id: u64) -> Result<u64, Error> {
    block_map
        .as_slice()
        .get(block_id as usize)
        .copied()
        .ok_or_else(|| Error::Bounds(format!("block id {} is out of range", block_id)))
}
