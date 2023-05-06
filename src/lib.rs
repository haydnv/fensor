use std::fmt;

use number_general::NumberType;

pub mod dense;
pub mod sparse;

pub type Axes = Vec<usize>;
pub type Coord = Vec<u64>;
pub type Shape = Vec<u64>;
pub type Strides = Vec<u64>;

#[cfg(debug_assertions)]
const IDEAL_BLOCK_SIZE: usize = 24;

#[cfg(not(debug_assertions))]
const IDEAL_BLOCK_SIZE: usize = 65_536;

#[derive(Clone, Debug)]
pub enum AxisBound {
    At(u64),
    In(u64, u64, u64),
    Of(Vec<u64>),
}

#[derive(Clone, Debug, Default)]
pub struct Bounds(Vec<AxisBound>);

impl FromIterator<AxisBound> for Bounds {
    fn from_iter<I: IntoIterator<Item = AxisBound>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

#[derive(Debug)]
pub enum Error {
    Bounds(String),
    IO(std::io::Error),
    Math(ha_ndarray::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => cause.fmt(f),
            Self::IO(cause) => cause.fmt(f),
            Self::Math(cause) => cause.fmt(f),
        }
    }
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

pub trait TensorInstance: Send + Sync + 'static {
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

    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error>;

    fn expand(self, axes: Axes) -> Result<Self::Expand, Error>;

    fn reshape(self, shape: Shape) -> Result<Self::Reshape, Error>;

    fn slice(self, bounds: Bounds) -> Result<Self::Slice, Error>;

    fn transpose(self, axes: Axes) -> Result<Self::Transpose, Error>;
}

#[inline]
fn strides_for(shape: &[u64], ndim: usize) -> Strides {
    debug_assert!(ndim >= shape.len());

    let zeros = std::iter::repeat(0).take(ndim - shape.len());

    let strides = shape.iter().enumerate().map(|(x, dim)| {
        if *dim == 1 {
            0
        } else {
            shape.iter().rev().take(shape.len() - 1 - x).product()
        }
    });

    zeros.chain(strides).collect()
}
