use async_trait::async_trait;
use destream::de;
use ha_ndarray::CDatatype;

use crate::IDEAL_BLOCK_SIZE;

pub struct Cached<T> {
    pub data: Vec<T>,
}

impl<T> From<Vec<T>> for Cached<T> {
    fn from(data: Vec<T>) -> Self {
        Self { data }
    }
}

struct CacheVisitor<T> {
    data: Vec<T>,
}

impl<T> CacheVisitor<T> {
    fn new() -> Self {
        Self {
            data: Vec::with_capacity(IDEAL_BLOCK_SIZE * 2),
        }
    }
}

macro_rules! decode_cached {
    ($t:ty, $name:expr, $decode:ident, $visit:ident) => {
        #[async_trait]
        impl de::Visitor for CacheVisitor<$t> {
            type Value = Cached<$t>;

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

                Ok(Cached { data })
            }
        }

        #[async_trait]
        impl de::FromStream for Cached<$t> {
            type Context = ();

            async fn from_stream<D: de::Decoder>(
                _cxt: (),
                decoder: &mut D,
            ) -> Result<Self, D::Error> {
                decoder.$decode(CacheVisitor::<$t>::new()).await
            }
        }
    };
}

decode_cached!(u8, "byte array", decode_array_u8, visit_array_u8);
decode_cached!(
    u16,
    "16-bit unsigned int array",
    decode_array_u16,
    visit_array_u16
);
decode_cached!(
    u32,
    "32-bit unsigned int array",
    decode_array_u32,
    visit_array_u32
);
decode_cached!(
    u64,
    "64-bit unsigned int array",
    decode_array_u64,
    visit_array_u64
);

decode_cached!(i16, "16-bit int array", decode_array_i16, visit_array_i16);
decode_cached!(i32, "32-bit int array", decode_array_i32, visit_array_i32);
decode_cached!(i64, "64-bit int array", decode_array_i64, visit_array_i64);

decode_cached!(f32, "32-bit int array", decode_array_f32, visit_array_f32);
decode_cached!(f64, "64-bit int array", decode_array_f64, visit_array_f64);
