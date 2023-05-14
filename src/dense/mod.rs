use number_general::NumberType;

use super::{Axes, Error, Range, Shape, TensorInstance, TensorTransform};

pub use access::*;

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

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }
}

impl<A> TensorTransform for DenseTensor<A>
where
    A: DenseInstance,
{
    type Broadcast = DenseTensor<DenseBroadcast<A>>;
    type Expand = DenseTensor<DenseReshape<A>>;
    type Reshape = DenseTensor<DenseReshape<A>>;
    type Slice = DenseTensor<DenseSlice<A>>;
    type Transpose = DenseTensor<DenseTranspose<A>>;

    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error> {
        DenseBroadcast::new(self.accessor, shape).map(DenseTensor::from)
    }

    fn expand(self, mut axes: Axes) -> Result<Self::Expand, Error> {
        let mut shape = self.shape().to_vec();

        axes.sort();

        for x in axes.into_iter().rev() {
            shape.insert(x, 1);
        }

        DenseReshape::new(self.accessor, shape.into()).map(DenseTensor::from)
    }

    fn reshape(self, shape: Shape) -> Result<Self::Reshape, Error> {
        DenseReshape::new(self.accessor, shape).map(DenseTensor::from)
    }

    fn slice(self, range: Range) -> Result<Self::Slice, Error> {
        DenseSlice::new(self.accessor, range).map(DenseTensor::from)
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self::Transpose, Error> {
        DenseTranspose::new(self.accessor, permutation).map(DenseTensor::from)
    }
}

impl<A> From<A> for DenseTensor<A> {
    fn from(accessor: A) -> Self {
        Self { accessor }
    }
}
