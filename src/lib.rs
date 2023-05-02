use std::marker::PhantomData;

use async_trait::async_trait;
use derive_more::Display;
use destream::de;
use freqfs::{DirLock, FileLoad};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use ha_ndarray::*;
use safecast::AsType;

const IDEAL_BLOCK_SIZE: usize = 65_536;

pub struct Array<T> {
    data: Vec<T>,
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

#[derive(Debug, Display)]
pub enum Error {
    IO(std::io::Error),
    Math(ha_ndarray::Error),
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(cause: std::io::Error) -> Self {
        Self::IO(cause)
    }
}

impl From<ha_ndarray::Error> for Error {
    fn from(cause: ha_ndarray::Error) -> Self {
        Self::Math(cause)
    }
}

#[async_trait]
trait DenseAccessor {
    type Block: NDArrayRead;

    async fn blocks(self) -> Result<Box<dyn Stream<Item = Result<Self::Block, Error>>>, Error>;
}

pub struct DenseFile<FE, T> {
    axis: usize,
    shape: Vec<usize>,
    blocks: DirLock<FE>,
    dtype: PhantomData<T>,
}

#[async_trait]
impl<FE, T> DenseAccessor for DenseFile<FE, T>
where
    FE: FileLoad + AsType<Array<T>>,
    T: CDatatype + 'static,
    Array<T>: de::FromStream<Context = ()>,
{
    type Block = ArrayBase<T>;

    async fn blocks(self) -> Result<Box<dyn Stream<Item = Result<Self::Block, Error>>>, Error> {
        let dir = self.blocks.read().await;

        let block_reads = dir
            .files()
            .cloned()
            .map(|file| file.into_read())
            .collect::<Vec<_>>();

        let blocks = stream::iter(block_reads)
            .buffered(num_cpus::get())
            .map_err(Error::from)
            .map(move |array| {
                let array = array?;
                let block_size = self.shape.iter().rev().take(self.axis).product::<usize>();

                let block_shape = if self.axis == self.shape.len() - 1 {
                    vec![array.data.len()]
                } else {
                    let axis_dim = array.data.len() / block_size;
                    debug_assert_eq!(array.data.len() % axis_dim, 0);

                    let mut shape = Vec::<usize>::with_capacity(self.shape.len() - self.axis + 1);
                    shape.push(axis_dim);
                    shape.extend(self.shape.iter().skip(self.axis).copied());

                    shape
                };

                ArrayBase::new(block_shape, array.data.to_vec()).map_err(Error::from)
            });

        Ok(Box::new(blocks))
    }
}
