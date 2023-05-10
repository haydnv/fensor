use ha_ndarray::{Buffer, CDatatype};
use number_general::{DType, NumberType};
use safecast::AsType;

use super::{TensorInstance, TensorTransform};

use crate::{Axes, Bounds, Error, Shape};
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

impl<FE, T> TensorTransform for DenseTensor<FE, T>
where
    FE: AsType<Buffer<T>> + Send + Sync + 'static,
    T: CDatatype + DType,
{
    fn broadcast(self, shape: Shape) -> Result<Self, Error> {
        todo!()
    }

    fn expand(self, axes: Axes) -> Result<Self, Error> {
        todo!()
    }

    fn reshape(self, shape: Shape) -> Result<Self, Error> {
        todo!()
    }

    fn slice(self, bounds: Bounds) -> Result<Self, Error> {
        todo!()
    }

    fn transpose(self, axes: Axes) -> Result<Self, Error> {
        todo!()
    }
}
