use std::marker::PhantomData;
use std::{fmt, io};

use async_trait::async_trait;
use b_table::{Range, TableLock};
use futures::{Stream, TryStreamExt};
use ha_ndarray::CDatatype;
use number_general::{DType, Number, NumberCollator, NumberType};
use safecast::{AsType, CastInto};

use super::{Coord, Error, Shape, TensorInstance};

mod stream;

const BLOCK_SIZE: usize = 4_096;

pub type Node = b_table::b_tree::Node<Vec<Vec<Number>>>;

#[derive(Clone, Eq, PartialEq)]
pub struct IndexSchema {
    columns: Vec<usize>,
}

impl b_table::b_tree::Schema for IndexSchema {
    type Error = Error;
    type Value = Number;

    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn len(&self) -> usize {
        self.columns.len()
    }

    fn order(&self) -> usize {
        12
    }

    fn validate(&self, key: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if key.len() == self.len() {
            Ok(key)
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidData, "wrong number of values").into())
        }
    }
}

impl b_table::IndexSchema for IndexSchema {
    type Id = usize;

    fn columns(&self) -> &[Self::Id] {
        &self.columns
    }

    fn extract_key(&self, key: &[Self::Value], other: &Self) -> Vec<Self::Value> {
        debug_assert_eq!(key.len(), self.columns.len());
        other.columns.iter().copied().map(|x| key[x]).collect()
    }
}

impl fmt::Debug for IndexSchema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("sparse tensor index schema")
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Schema {
    primary: IndexSchema,
    auxiliary: Vec<(String, IndexSchema)>,
    shape: Shape,
}

impl b_table::Schema for Schema {
    type Id = usize;
    type Error = Error;
    type Value = Number;
    type Index = IndexSchema;

    fn key(&self) -> &[Self::Id] {
        &self.primary.columns[..self.shape.len()]
    }

    fn values(&self) -> &[Self::Id] {
        &self.primary.columns[self.shape.len()..]
    }

    fn primary(&self) -> &Self::Index {
        &self.primary
    }

    fn auxiliary(&self) -> &[(String, IndexSchema)] {
        &self.auxiliary
    }

    fn validate_key(&self, key: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if key.len() == self.shape.len() {
            Ok(key)
        } else {
            let cause = io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid key: {:?}", key),
            );

            Err(cause.into())
        }
    }

    fn validate_values(&self, values: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if values.len() == 1 {
            Ok(values)
        } else {
            let cause = io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid values: {:?}", values),
            );

            Err(cause.into())
        }
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("sparse tensor schema")
    }
}

#[async_trait]
pub trait SparseInstance {
    type DType;

    async fn elements(
        self,
    ) -> Result<Box<dyn Stream<Item = Result<(Coord, Self::DType), Error>>>, Error>;
}

pub struct SparseTable<FE, T> {
    table: TableLock<Schema, IndexSchema, NumberCollator, FE>,
    dtype: PhantomData<T>,
}

impl<FE, T: DType> TensorInstance for SparseTable<FE, T> {
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
    type DType = T;

    async fn elements(
        self,
    ) -> Result<Box<dyn Stream<Item = Result<(Coord, Self::DType), Error>>>, Error> {
        let rows = self.table.rows(Range::default(), &[], false).await?;
        let elements = rows.map_ok(|row| unwrap_row(row)).map_err(Error::from);
        Ok(Box::new(elements))
    }
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
