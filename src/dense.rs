use std::marker::PhantomData;

use async_trait::async_trait;
use destream::de;
use freqfs::{DirLock, FileLoad, FileLock};
use futures::future::{FutureExt, TryFutureExt};
use ha_ndarray::*;
use number_general::{DType, NumberClass, NumberInstance, NumberType};
use safecast::AsType;

use super::{Error, Shape, TensorInstance, IDEAL_BLOCK_SIZE};

type BlockShape = ha_ndarray::Shape;

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
pub trait Block: Send + Sync + 'static {
    type Array: NDArrayRead;

    async fn into_read(self) -> Result<Self::Array, Error>;
}

pub struct DenseBlock<FE, T> {
    shape: BlockShape,
    file: FileLock<FE>,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for DenseBlock<FE, T> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.to_vec(),
            file: self.file.clone(),
            dtype: PhantomData,
        }
    }
}

#[async_trait]
impl<FE, T> Block for DenseBlock<FE, T>
where
    FE: FileLoad + AsType<Array<T>>,
    T: CDatatype + DType,
    Array<T>: de::FromStream<Context = ()>,
{
    type Array = ArrayBase<T>;

    async fn into_read(self) -> Result<Self::Array, Error> {
        self.file
            .into_read()
            .map(|result| {
                result.map_err(Error::from).and_then(|array| {
                    ArrayBase::new(self.shape, array.data.to_vec()).map_err(Error::from)
                })
            })
            .map_err(Error::from)
            .await
    }
}

pub trait DenseInstance {
    type Block: Block;
    type DType: CDatatype + DType;

    fn into_blocks(self) -> Vec<Self::Block>;
}

#[derive(Clone)]
pub struct DenseFile<FE, T> {
    dir: DirLock<FE>,
    block_map: ArrayBase<u64>,
    blocks: Vec<DenseBlock<FE, T>>,
    shape: Shape,
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
        let block_axis = block_axis_for(&shape, block_size);

        let blocks = {
            let dtype_size = T::dtype().size();
            let mut blocks = Vec::with_capacity(num_blocks);
            let mut dir = dir.write().await;
            for block_id in 0..(num_blocks - 1) {
                let file = dir.create_file(
                    block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )?;

                let block_shape = block_shape_for(block_axis, &shape, block_size);

                blocks.push(DenseBlock {
                    file,
                    shape: block_shape,
                    dtype: PhantomData,
                });
            }

            let last_block_id = num_blocks - 1;
            let (last_file, last_block_size) = if size % block_size as u64 == 0 {
                let file = dir.create_file(
                    last_block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )?;

                (file, block_size)
            } else {
                let block_size = (size % block_size as u64) as usize;

                let file = dir.create_file(
                    last_block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )?;

                (file, block_size)
            };

            blocks.push(DenseBlock {
                file: last_file,
                shape: block_shape_for(block_axis, &shape, last_block_size),
                dtype: PhantomData,
            });

            blocks
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
            blocks,
            shape,
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

impl<FE, T> DenseInstance for DenseFile<FE, T>
where
    FE: FileLoad + AsType<Array<T>>,
    T: CDatatype + DType + 'static,
    Array<T>: de::FromStream<Context = ()>,
{
    type Block = DenseBlock<FE, T>;
    type DType = T;

    fn into_blocks(self) -> Vec<Self::Block> {
        self.blocks
    }
}

#[derive(Clone)]
pub struct DenseView<B> {
    blocks: Vec<B>,
    block_map: ArrayBase<u64>,
    shape: Shape,
}

impl<B> DenseView<B> {
    fn new(blocks: Vec<B>, block_map: ArrayBase<u64>, shape: Shape) -> Self {
        debug_assert!(block_map.ndim() <= shape.len());
        debug_assert!(block_map
            .as_slice()
            .iter()
            .copied()
            .all(|block_id| block_id < blocks.len() as u64));

        Self {
            blocks,
            block_map,
            shape,
        }
    }
}

impl<B: Block> TensorInstance for DenseView<B>
where
    <B::Array as NDArray>::DType: DType,
{
    fn dtype(&self) -> NumberType {
        <B::Array as NDArray>::DType::dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

impl<B: Block> DenseInstance for DenseView<B>
where
    <B::Array as NDArray>::DType: DType,
{
    type Block = B;
    type DType = <B::Array as NDArray>::DType;

    fn into_blocks(self) -> Vec<Self::Block> {
        self.blocks
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
