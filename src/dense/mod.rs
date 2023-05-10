use ha_ndarray::CDatatype;
use number_general::{DType, NumberType};

use super::TensorInstance;

pub use access::{DenseAccess, DenseFile, DenseInstance, DenseSlice};

mod access;

pub enum DenseTensor<FE, T> {
    Base(DenseFile<FE, T>),
    Slice(DenseSlice<DenseFile<FE, T>>),
    View(DenseAccess<FE, T>),
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for DenseTensor<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &[u64] {
        match self {
            Self::Base(base) => base.shape(),
            Self::Slice(slice) => slice.shape(),
            Self::View(view) => view.shape(),
        }
    }
}
