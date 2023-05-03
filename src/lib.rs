use derive_more::Display;
use ha_ndarray::AxisBound;
use number_general::NumberType;

pub mod dense;
pub mod sparse;

pub type Coord = Vec<u64>;
pub type Shape = Vec<u64>;

#[derive(Clone)]
pub struct Bounds(Vec<AxisBound>);

#[derive(Debug, Display)]
pub enum Error {
    Bounds(String),
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

pub trait TensorInstance: Clone + Send + Sync + 'static {
    fn dtype(&self) -> NumberType;

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn shape(&self) -> &[u64];

    fn size(&self) -> u64 {
        self.shape().iter().product()
    }
}

pub trait TensorTransform: TensorInstance {
    type Broadcast: TensorInstance;
    type Expand: TensorInstance;
    type Reshape: TensorInstance;
    type Slice: TensorInstance;
    type Transpose: TensorInstance;

    fn broadcast(&self, shape: Shape) -> Result<Self::Broadcast, Error>;

    fn expand(&self, axes: Vec<usize>) -> Result<Self::Expand, Error>;

    fn reshape(&self, shape: Shape) -> Result<Self::Reshape, Error>;

    fn slice(&self, bounds: Bounds) -> Result<Self::Slice, Error>;

    fn transpose(&self, axes: Vec<usize>) -> Result<Self::Transpose, Error>;
}
