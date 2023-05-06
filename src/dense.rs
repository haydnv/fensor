use std::marker::PhantomData;
use std::pin::Pin;
use std::{fmt, io};

use async_trait::async_trait;
use destream::de;
use freqfs::{DirLock, FileLoad};
use futures::{stream, Stream, StreamExt, TryStreamExt};
use ha_ndarray::*;
use number_general::{DType, NumberClass, NumberInstance, NumberType};
use safecast::AsType;

use super::{Axes, Error, Shape, TensorInstance, IDEAL_BLOCK_SIZE};

type BlockShape = ha_ndarray::Shape;
type BlockStream<Block> = Pin<Box<dyn Stream<Item = Result<Block, Error>>>>;

pub struct Array<T> {
    data: Vec<T>,
}

impl<T> From<Vec<T>> for Array<T> {
    fn from(data: Vec<T>) -> Self {
        Self { data }
    }
}

struct ArrayVisitor<T> {
    data: Vec<T>,
}

impl<T> ArrayVisitor<T> {
    fn new() -> Self {
        Self {
            data: Vec::with_capacity(IDEAL_BLOCK_SIZE * 2),
        }
    }
}

macro_rules! decode_array {
    ($t:ty, $name:expr, $decode:ident, $visit:ident) => {
        #[async_trait]
        impl de::Visitor for ArrayVisitor<$t> {
            type Value = Array<$t>;

            fn expecting() -> &'static str {
                $name
            }

            async fn $visit<A: de::ArrayAccess<$t>>(
                self,
                mut array: A,
            ) -> Result<Self::Value, A::Error> {
                const BUF_SIZE: usize = 4_096;
                let mut data = self.data;

                let mut buf = [<$t>::zero(); BUF_SIZE];
                loop {
                    let len = array.buffer(&mut buf).await?;
                    if len == 0 {
                        break;
                    } else {
                        data.extend_from_slice(&buf[..len]);
                    }
                }

                Ok(Array { data })
            }
        }

        #[async_trait]
        impl de::FromStream for Array<$t> {
            type Context = ();

            async fn from_stream<D: de::Decoder>(
                _cxt: (),
                decoder: &mut D,
            ) -> Result<Self, D::Error> {
                decoder.$decode(ArrayVisitor::<$t>::new()).await
            }
        }
    };
}

decode_array!(u8, "byte array", decode_array_u8, visit_array_u8);
decode_array!(
    u16,
    "16-bit unsigned int array",
    decode_array_u16,
    visit_array_u16
);
decode_array!(
    u32,
    "32-bit unsigned int array",
    decode_array_u32,
    visit_array_u32
);
decode_array!(
    u64,
    "64-bit unsigned int array",
    decode_array_u64,
    visit_array_u64
);

decode_array!(i16, "16-bit int array", decode_array_i16, visit_array_i16);
decode_array!(i32, "32-bit int array", decode_array_i32, visit_array_i32);
decode_array!(i64, "64-bit int array", decode_array_i64, visit_array_i64);

decode_array!(f32, "32-bit int array", decode_array_f32, visit_array_f32);
decode_array!(f64, "64-bit int array", decode_array_f64, visit_array_f64);

#[async_trait]
pub trait DenseInstance: TensorInstance + fmt::Debug + Send + Sync + 'static {
    type Block: NDArrayRead<DType = Self::DType> + NDArrayTransform;
    type DType: CDatatype + DType;

    fn block_size(&self) -> usize;

    async fn read_block(&self, block_id: u64) -> Result<Self::Block, Error>;

    async fn blocks(self) -> Result<BlockStream<Self::Block>, Error>;
}

#[derive(Clone)]
pub struct DenseFile<FE, T> {
    dir: DirLock<FE>,
    block_map: ArrayBase<u64>,
    block_size: usize,
    shape: Shape,
    dtype: PhantomData<T>,
}

impl<FE, T> DenseFile<FE, T>
where
    FE: FileLoad + AsType<Array<T>> + Send + Sync,
    T: CDatatype + DType + NumberInstance,
{
    pub async fn constant(dir: DirLock<FE>, shape: Shape, value: T) -> Result<Self, Error> {
        validate_shape(&shape)?;

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

        let block_map = ArrayBase::new(map_shape, (0u64..num_blocks as u64).into_iter().collect())?;

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

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<FE, T> DenseInstance for DenseFile<FE, T>
where
    FE: FileLoad + AsType<Array<T>>,
    T: CDatatype + DType + 'static,
    Array<T>: de::FromStream<Context = ()>,
{
    type Block = ArrayBase<T>;
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

        let array = file.read().await?;
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, array.data.len());
        ArrayBase::new(block_shape, array.data.to_vec()).map_err(Error::from)
    }

    async fn blocks(self) -> Result<BlockStream<Self::Block>, Error> {
        let shape = self.shape;
        let block_axis = block_axis_for(&shape, self.block_size);
        let dir = self.dir.into_read().await;

        let blocks = stream::iter(self.block_map.into_data())
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
                let array = result?;
                let block_shape = block_shape_for(block_axis, &shape, array.data.len());
                ArrayBase::new(block_shape, array.data.to_vec()).map_err(Error::from)
            });

        Ok(Box::pin(blocks))
    }
}

impl<FE, T> fmt::Debug for DenseFile<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "dense tensor with shape {:?}", self.shape)
    }
}

pub struct DenseBroadcast<S> {
    source: S,
    shape: Shape,
    block_map: ArrayBase<u64>,
    block_size: usize,
}

impl<S: TensorInstance> TensorInstance for DenseBroadcast<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseBroadcast<S>
where
    <S::Block as NDArrayTransform>::Broadcast: NDArrayRead<DType = S::DType> + NDArrayTransform,
{
    type Block = <S::Block as NDArrayTransform>::Broadcast;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, block_id: u64) -> Result<Self::Block, Error> {
        let source_block_id = self
            .block_map
            .as_slice()
            .get(block_id as usize)
            .copied()
            .ok_or_else(|| {
                Error::Bounds(format!(
                    "block {} is out of bounds for {:?}",
                    block_id, self
                ))
            })?;

        let block_axis = block_axis_for(&self.shape, self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, self.block_size);
        let source_block = self.source.read_block(source_block_id).await?;
        source_block.broadcast(block_shape).map_err(Error::from)
    }

    async fn blocks(self) -> Result<BlockStream<Self::Block>, Error> {
        let block_axis = block_axis_for(&self.shape, self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, self.block_size);

        let blocks = stream::iter(self.block_map.into_data())
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

impl<S: fmt::Debug> fmt::Debug for DenseBroadcast<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "broadcast of {:?} into {:?}", self.source, self.shape)
    }
}

pub struct DenseTranspose<S> {
    source: S,
    shape: Shape,
    permutation: Axes,
    block_map: ArrayBase<u64>,
}

impl<S: DenseInstance> DenseTranspose<S> {
    fn new(source: S, permutation: Option<Axes>) -> Result<Self, Error> {
        let permutation = if let Some(axes) = permutation {
            if axes.len() == source.ndim()
                && (0..source.ndim()).into_iter().all(|x| axes.contains(&x))
            {
                Ok(axes)
            } else {
                Err(Error::Bounds(format!(
                    "invalid permutation for {:?}: {:?}",
                    source, axes
                )))
            }
        } else {
            Ok((0..source.ndim()).into_iter().rev().collect())
        }?;

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

        let block_map = ArrayBase::new(map_shape, (0..num_blocks).into_iter().collect())?;
        let block_map = block_map.transpose(Some(map_axes.to_vec()))?;
        let block_map = ArrayBase::copy(&block_map)?;

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

    fn shape(&self) -> &[u64] {
        self.source.shape()
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseTranspose<S>
where
    <S::Block as NDArrayTransform>::Transpose: NDArrayRead<DType = S::DType> + NDArrayTransform,
{
    type Block = <S::Block as NDArrayTransform>::Transpose;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, block_id: u64) -> Result<Self::Block, Error> {
        let source_block_id = self
            .block_map
            .as_slice()
            .get(block_id as usize)
            .copied()
            .ok_or_else(|| {
                Error::Bounds(format!(
                    "block {} is out of bounds for {:?}",
                    block_id, self
                ))
            })?;

        let block = self.source.read_block(source_block_id).await?;

        block
            .transpose(Some(self.permutation.to_vec()))
            .map_err(Error::from)
    }

    async fn blocks(self) -> Result<BlockStream<Self::Block>, Error> {
        let permutation = self.permutation;

        let blocks = stream::iter(self.block_map.into_data())
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
fn validate_shape(shape: &[u64]) -> Result<(), Error> {
    if shape.is_empty()
        || shape
            .iter()
            .copied()
            .any(|dim| dim == 0 || dim > u32::MAX as u64)
    {
        Err(Error::Bounds(format!(
            "invalid shape for dense tensor: {:?}",
            shape
        )))
    } else {
        Ok(())
    }
}
