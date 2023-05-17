use std::marker::PhantomData;
use std::pin::Pin;

use futures::Stream;
use ha_ndarray::CDatatype;
use number_general::{DType, Number, NumberType};
use safecast::AsType;

use super::{Axes, Coord, Error, Range, Shape, TensorInstance, TensorTransform};

pub use access::*;
pub use schema::*;

mod access;
mod schema;
mod stream;

const BLOCK_SIZE: usize = 4_096;

pub type Blocks<C, V> = Pin<Box<dyn Stream<Item = Result<(C, V), Error>> + Send>>;
pub type Elements<T> = Pin<Box<dyn Stream<Item = Result<(Coord, T), Error>> + Send>>;
pub type Node = b_table::b_tree::Node<Vec<Vec<Number>>>;

pub struct SparseTensor<FE, T, A> {
    accessor: A,
    phantom: PhantomData<(FE, T)>,
}

impl<FE, T, A> SparseTensor<FE, T, A> {
    pub fn into_inner(self) -> A {
        self.accessor
    }
}

impl<FE, T, A: Clone> Clone for SparseTensor<FE, T, A> {
    fn clone(&self) -> Self {
        Self {
            accessor: self.accessor.clone(),
            phantom: PhantomData,
        }
    }
}

impl<FE, T, A> TensorInstance for SparseTensor<FE, T, A>
where
    FE: Send + Sync + 'static,
    T: CDatatype + DType,
    A: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }
}

impl<FE, T, A> TensorTransform for SparseTensor<FE, T, A>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType,
    A: SparseInstance + Into<SparseAccess<FE, T>>,
{
    type Broadcast = SparseTensor<FE, T, SparseBroadcast<FE, T>>;
    type Expand = SparseTensor<FE, T, SparseExpand<A>>;
    type Reshape = SparseTensor<FE, T, SparseReshape<A>>;
    type Slice = SparseTensor<FE, T, SparseSlice<A>>;
    type Transpose = SparseTensor<FE, T, SparseTranspose<A>>;

    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error> {
        let accessor = SparseBroadcast::new(self.accessor, shape)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn expand(self, axes: Axes) -> Result<Self::Expand, Error> {
        let accessor = SparseExpand::new(self.accessor, axes)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn reshape(self, shape: Shape) -> Result<Self::Reshape, Error> {
        let accessor = SparseReshape::new(self.accessor, shape)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn slice(self, range: Range) -> Result<Self::Slice, Error> {
        let accessor = SparseSlice::new(self.accessor, range)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self::Transpose, Error> {
        let accessor = SparseTranspose::new(self.accessor, permutation)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }
}

impl<FE, T, A: Into<SparseAccess<FE, T>>> From<A> for SparseTensor<FE, T, SparseAccess<FE, T>> {
    fn from(accessor: A) -> Self {
        Self {
            accessor: accessor.into(),
            phantom: PhantomData,
        }
    }
}
