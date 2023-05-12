use number_general::NumberType;

use super::{Axes, Bounds, Error, Shape, TensorInstance, TensorTransform};

pub use access::{DenseAccess, DenseBroadcast, DenseFile, DenseInstance, DenseSlice};

mod access;

#[derive(Clone)]
pub struct DenseTensor<A> {
    accessor: A,
}

impl<A> DenseTensor<A> {
    pub fn into_inner(self) -> A {
        self.accessor
    }
}

impl<A: TensorInstance> TensorInstance for DenseTensor<A> {
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn shape(&self) -> &[u64] {
        self.accessor.shape()
    }
}

impl<A> TensorTransform for DenseTensor<A>
where
    A: DenseInstance,
{
    type Broadcast = DenseTensor<DenseBroadcast<A>>;

    fn broadcast(self, shape: Shape) -> Result<DenseTensor<DenseBroadcast<A>>, Error> {
        let accessor = DenseBroadcast::new(self.accessor, shape)?;
        Ok(DenseTensor { accessor })
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

impl<A> From<A> for DenseTensor<A> {
    fn from(accessor: A) -> Self {
        Self { accessor }
    }
}
