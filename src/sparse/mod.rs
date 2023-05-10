use std::pin::Pin;

use futures::Stream;
use ha_ndarray::CDatatype;
use number_general::{DType, Number, NumberType};

use super::{Coord, Error, TensorInstance};

pub use access::{SparseAccess, SparseInstance, SparseSlice, SparseTable};
pub use schema::*;

mod access;
mod schema;
mod stream;

const BLOCK_SIZE: usize = 4_096;

pub type Elements<T> = Pin<Box<dyn Stream<Item = Result<(Coord, T), Error>>>>;
pub type Node = b_table::b_tree::Node<Vec<Vec<Number>>>;

pub enum SparseTensor<FE, T> {
    Table(SparseTable<FE, T>),
    Slice(SparseSlice<SparseTable<FE, T>>),
    View(SparseAccess<FE, T>),
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for SparseTensor<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &[u64] {
        match self {
            Self::Table(table) => table.shape(),
            Self::Slice(slice) => slice.shape(),
            Self::View(view) => view.shape(),
        }
    }
}
