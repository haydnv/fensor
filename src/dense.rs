use std::io;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::de;
use freqfs::{DirLock, FileLoad};
use futures::stream::{self, Stream, StreamExt};
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
trait DenseInstance {
    type Block: NDArrayRead;

    async fn blocks(self) -> Result<Box<dyn Stream<Item = Result<Self::Block, Error>>>, Error>;
}

pub struct DenseFile<FE, T> {
    file: DirLock<FE>,
    block_size: usize,
    block_map: ArrayBase<u64>,
    shape: Shape,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for DenseFile<FE, T> {
    fn clone(&self) -> Self {
        // TODO: can DenseFile::clone be zero-alloc?
        Self {
            file: self.file.clone(),
            block_size: self.block_size,
            block_map: self.block_map.clone(),
            shape: self.shape.to_vec(),
            dtype: PhantomData,
        }
    }
}

impl<FE, T> DenseFile<FE, T>
where
    FE: AsType<Array<T>> + Send + Sync,
    T: CDatatype + DType + NumberInstance,
{
    pub async fn create(file: DirLock<FE>, shape: Shape) -> Result<Self, Error> {
        let size = shape.iter().product();

        let (block_size, num_blocks) = if size < (2 * IDEAL_BLOCK_SIZE) as u64 {
            (size as usize, 1)
        } else if shape.len() == 1 && size % IDEAL_BLOCK_SIZE as u64 == 0 {
            (IDEAL_BLOCK_SIZE, (size / IDEAL_BLOCK_SIZE as u64) as usize)
        } else if shape.len() == 1
            || (shape.iter().rev().take(2).product::<u64>() > (2 * IDEAL_BLOCK_SIZE as u64))
        {
            let num_blocks = div_ceil(size, IDEAL_BLOCK_SIZE as u64) as usize;
            (IDEAL_BLOCK_SIZE, num_blocks as usize)
        } else {
            let matrix_size = shape.iter().rev().take(2).product::<u64>();
            let block_size =
                IDEAL_BLOCK_SIZE as u64 + (matrix_size - (IDEAL_BLOCK_SIZE as u64 % matrix_size));
            let num_blocks = div_ceil(size, IDEAL_BLOCK_SIZE as u64);
            (block_size as usize, num_blocks as usize)
        };

        debug_assert!(block_size > 0);

        {
            let zero = T::zero();
            let dtype_size = T::dtype().size();
            let mut blocks = file.write().await;
            for block_id in 0..(num_blocks - 1) {
                blocks.create_file(
                    block_id.to_string(),
                    vec![zero; block_size].into(),
                    block_size * dtype_size,
                )?;
            }

            if size % block_size as u64 == 0 {
                blocks.create_file(
                    (num_blocks - 1).to_string(),
                    vec![zero; block_size].into(),
                    block_size * dtype_size,
                )?;
            } else {
                blocks.create_file(
                    (num_blocks - 1).to_string(),
                    vec![zero; block_size].into(),
                    block_size * dtype_size,
                )?;
            }
        }

        let block_axis = block_axis_for(&shape, block_size);
        let map_shape = shape
            .iter()
            .take(block_axis)
            .copied()
            .map(|dim| dim as usize)
            .collect();

        let block_map = ArrayBase::new(map_shape, (0u64..num_blocks as u64).into_iter().collect())?;

        Ok(Self {
            file,
            block_size,
            block_map,
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
    T: CDatatype + 'static,
    Array<T>: de::FromStream<Context = ()>,
{
    type Block = ArrayBase<T>;

    async fn blocks(self) -> Result<Box<dyn Stream<Item = Result<Self::Block, Error>>>, Error> {
        let file = self.file.read().await;

        let block_reads = self
            .block_map
            .as_slice()
            .iter()
            .map(|block_id| {
                file.get_file(block_id)
                    .cloned()
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::NotFound,
                            format!("dense tensor block {}", block_id),
                        )
                    })
                    .map(|file| file.into_read())
                    .map_err(Error::from)
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let block_axis = block_axis_for(&self.shape, self.block_size);

        let blocks = stream::iter(block_reads)
            .buffered(num_cpus::get())
            .map(move |result| {
                let array = result?;
                let block_size = self.shape.iter().rev().take(block_axis).product::<u64>() as usize;

                let block_shape = if block_axis == self.shape.len() - 1 {
                    vec![array.data.len()]
                } else {
                    let axis_dim = array.data.len() / block_size;
                    debug_assert_eq!(array.data.len() % axis_dim, 0);

                    let mut shape = Vec::with_capacity(self.shape.len() - block_axis + 1);
                    shape.push(axis_dim);
                    shape.extend(
                        self.shape
                            .iter()
                            .skip(block_axis)
                            .copied()
                            .map(|dim| dim as usize),
                    );

                    shape
                };

                ArrayBase::new(block_shape, array.data.to_vec()).map_err(Error::from)
            });

        Ok(Box::new(blocks))
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
fn div_ceil(num: u64, denom: u64) -> u64 {
    if num % denom == 0 {
        num / denom
    } else {
        (num / denom) + 1
    }
}
