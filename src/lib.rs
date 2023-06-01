pub extern crate ha_ndarray;

use std::fmt;

use destream::de;
use freqfs::FileLoad;
use number_general::{DType, Number, NumberInstance, NumberType};
use safecast::{AsType, CastInto};

pub use dense::{
    DenseAccess, DenseCow, DenseFile, DenseInstance, DenseSlice, DenseTensor, DenseWrite,
    DenseWriteGuard, DenseWriteLock,
};
pub use ha_ndarray::{Buffer, CDatatype};
pub use shape::{AxisRange, Range, Shape};
pub use sparse::{
    Node, SparseAccess, SparseCow, SparseInstance, SparseSlice, SparseTable, SparseTensor,
    SparseWrite, SparseWriteGuard,
};

pub mod dense;
pub mod sparse;

mod shape;

pub type Axes = Vec<usize>;
pub type Coord = Vec<u64>;
pub type Strides = Vec<u64>;

#[cfg(debug_assertions)]
const IDEAL_BLOCK_SIZE: usize = 24;

#[cfg(not(debug_assertions))]
const IDEAL_BLOCK_SIZE: usize = 65_536;

#[derive(Debug)]
pub enum Error {
    Bounds(String),
    IO(std::io::Error),
    Math(ha_ndarray::Error),
    Index(<u64 as TryInto<usize>>::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => cause.fmt(f),
            Self::IO(cause) => cause.fmt(f),
            Self::Math(cause) => cause.fmt(f),
            Self::Index(cause) => cause.fmt(f),
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

    fn shape(&self) -> &Shape;

    fn size(&self) -> u64 {
        self.shape().iter().product()
    }
}

impl<T: TensorInstance> TensorInstance for Box<T> {
    fn dtype(&self) -> NumberType {
        (**self).dtype()
    }

    fn shape(&self) -> &Shape {
        (**self).shape()
    }
}

pub trait TensorTransform: TensorInstance + Sized {
    type Broadcast: TensorInstance;
    type Expand: TensorInstance;
    type Reshape: TensorInstance;
    type Slice: TensorInstance;
    type Transpose: TensorInstance;

    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error>;

    fn expand(self, axes: Axes) -> Result<Self::Expand, Error>;

    fn reshape(self, shape: Shape) -> Result<Self::Reshape, Error>;

    fn slice(self, range: Range) -> Result<Self::Slice, Error>;

    fn transpose(self, permutation: Option<Axes>) -> Result<Self::Transpose, Error>;
}

pub enum Dense<FE, T> {
    File(DenseTensor<DenseFile<FE, T>>),
    Slice(DenseTensor<DenseSlice<DenseFile<FE, T>>>),
    View(DenseTensor<DenseAccess<FE, T>>),
}

macro_rules! dense_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::File($var) => $call,
            Self::Slice($var) => $call,
            Self::View($var) => $call,
        }
    };
}

impl<FE, T> Clone for Dense<FE, T> {
    fn clone(&self) -> Self {
        match self {
            Self::File(file) => Self::File(file.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
            Self::View(view) => Self::View(view.clone()),
        }
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for Dense<FE, T> {
    fn dtype(&self) -> NumberType {
        dense_dispatch!(self, this, this.dtype())
    }

    fn shape(&self) -> &Shape {
        dense_dispatch!(self, this, this.shape())
    }
}

impl<FE, T> TensorTransform for Dense<FE, T>
where
    FE: AsType<Buffer<T>> + FileLoad,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Broadcast = Self;
    type Expand = Self;
    type Reshape = Self;
    type Slice = Self;
    type Transpose = Self;

    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error> {
        if &shape == self.shape() {
            Ok(self)
        } else {
            dense_dispatch!(self, this, this.broadcast(shape).map(Self::from))
        }
    }

    fn expand(self, axes: Axes) -> Result<Self, Error> {
        if axes.is_empty() {
            Ok(self)
        } else {
            dense_dispatch!(self, this, this.expand(axes).map(Self::from))
        }
    }

    fn reshape(self, shape: Shape) -> Result<Self, Error> {
        if &shape == self.shape() {
            Ok(self)
        } else {
            dense_dispatch!(self, this, this.reshape(shape).map(Self::from))
        }
    }

    fn slice(self, range: Range) -> Result<Self, Error> {
        if range == Range::default()
            || range
                .iter()
                .zip(self.shape().iter().copied())
                .all(|(bound, dim)| bound.dim() == dim)
        {
            Ok(self)
        } else {
            dense_dispatch!(self, this, this.slice(range).map(Self::from))
        }
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self, Error> {
        if let Some(axes) = &permutation {
            if axes
                .iter()
                .copied()
                .zip(0..self.ndim())
                .all(|(o, x)| x == o)
            {
                return Ok(self);
            }
        }

        dense_dispatch!(self, this, this.transpose(permutation).map(Self::from))
    }
}

impl<FE, T, A: Into<DenseAccess<FE, T>>> From<DenseTensor<A>> for Dense<FE, T> {
    fn from(dense: DenseTensor<A>) -> Self {
        Self::View(DenseTensor::from(dense.into_inner().into()))
    }
}

pub enum Sparse<FE, T> {
    Table(SparseTensor<FE, T, SparseTable<FE, T>>),
    Slice(SparseTensor<FE, T, SparseSlice<SparseTable<FE, T>>>),
    View(SparseTensor<FE, T, SparseAccess<FE, T>>),
}

macro_rules! sparse_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::Table($var) => $call,
            Self::Slice($var) => $call,
            Self::View($var) => $call,
        }
    };
}

impl<FE, T> Clone for Sparse<FE, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Table(table) => Self::Table(table.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
            Self::View(view) => Self::View(view.clone()),
        }
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for Sparse<FE, T> {
    fn dtype(&self) -> NumberType {
        sparse_dispatch!(self, this, this.dtype())
    }

    fn shape(&self) -> &Shape {
        sparse_dispatch!(self, this, this.shape())
    }
}

impl<FE, T> TensorTransform for Sparse<FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType + NumberInstance,
    Number: CastInto<T>,
{
    type Broadcast = Self;
    type Expand = Self;
    type Reshape = Self;
    type Slice = Self;
    type Transpose = Self;

    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error> {
        if &shape == self.shape() {
            Ok(self)
        } else {
            sparse_dispatch!(self, this, this.broadcast(shape).map(Self::from))
        }
    }

    fn expand(self, axes: Axes) -> Result<Self, Error> {
        if axes.is_empty() {
            Ok(self)
        } else {
            sparse_dispatch!(self, this, this.expand(axes).map(Self::from))
        }
    }

    fn reshape(self, shape: Shape) -> Result<Self, Error> {
        if &shape == self.shape() {
            Ok(self)
        } else {
            sparse_dispatch!(self, this, this.reshape(shape).map(Self::from))
        }
    }

    fn slice(self, range: Range) -> Result<Self, Error> {
        if range == Range::default() {
            Ok(self)
        } else {
            sparse_dispatch!(self, this, this.slice(range).map(Self::from))
        }
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self::Transpose, Error> {
        if let Some(axes) = &permutation {
            if axes
                .iter()
                .copied()
                .zip(0..self.ndim())
                .all(|(o, x)| x == o)
            {
                return Ok(self);
            }
        }

        sparse_dispatch!(self, this, this.transpose(permutation).map(Self::from))
    }
}

impl<FE, T, A: Into<SparseAccess<FE, T>>> From<SparseTensor<FE, T, A>> for Sparse<FE, T> {
    fn from(sparse: SparseTensor<FE, T, A>) -> Self {
        Self::View(SparseTensor::from(sparse.into_inner().into()))
    }
}

pub enum Tensor<FE, T> {
    Dense(Dense<FE, T>),
    Sparse(Sparse<FE, T>),
}

impl<FE, T> TensorInstance for Tensor<FE, T>
where
    FE: Send + Sync + 'static,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::Sparse(sparse) => sparse.shape(),
        }
    }
}

impl<FE, T> TensorTransform for Tensor<FE, T>
where
    FE: AsType<Node> + AsType<Buffer<T>> + FileLoad,
    T: CDatatype + DType + NumberInstance,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: CastInto<T>,
{
    type Broadcast = Self;
    type Expand = Self;
    type Reshape = Self;
    type Slice = Self;
    type Transpose = Self;

    fn broadcast(self, shape: Shape) -> Result<Self, Error> {
        match self {
            Self::Dense(dense) => dense.broadcast(shape).map(Self::Dense),
            Self::Sparse(sparse) => sparse.broadcast(shape).map(Self::Sparse),
        }
    }

    fn expand(self, axes: Axes) -> Result<Self, Error> {
        match self {
            Self::Dense(dense) => dense.expand(axes).map(Self::Dense),
            Self::Sparse(sparse) => sparse.expand(axes).map(Self::Sparse),
        }
    }

    fn reshape(self, shape: Shape) -> Result<Self, Error> {
        match self {
            Self::Dense(dense) => dense.reshape(shape).map(Self::Dense),
            Self::Sparse(sparse) => sparse.reshape(shape).map(Self::Sparse),
        }
    }

    fn slice(self, range: Range) -> Result<Self, Error> {
        match self {
            Self::Dense(dense) => dense.slice(range).map(Self::Dense),
            Self::Sparse(sparse) => sparse.slice(range).map(Self::Sparse),
        }
    }

    fn transpose(self, permutation: Option<Axes>) -> Result<Self, Error> {
        match self {
            Self::Dense(dense) => dense.transpose(permutation).map(Self::Dense),
            Self::Sparse(sparse) => sparse.transpose(permutation).map(Self::Sparse),
        }
    }
}

#[inline]
fn offset_of(coord: Coord, shape: &[u64]) -> u64 {
    let strides = shape.iter().enumerate().map(|(x, dim)| {
        if *dim == 1 {
            0
        } else {
            shape.iter().rev().take(shape.len() - 1 - x).product()
        }
    });

    coord.into_iter().zip(strides).map(|(i, dim)| i * dim).sum()
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

#[inline]
fn validate_order(order: &[usize], ndim: usize) -> bool {
    order.len() == ndim && order.iter().all(|x| x < &ndim)
}

#[inline]
fn validate_transpose(permutation: Option<Axes>, shape: &[u64]) -> Result<Axes, Error> {
    if let Some(axes) = permutation {
        if axes.len() == shape.len() && (0..shape.len()).into_iter().all(|x| axes.contains(&x)) {
            Ok(axes)
        } else {
            Err(Error::Bounds(format!(
                "invalid permutation for shape {:?}: {:?}",
                shape, axes
            )))
        }
    } else {
        Ok((0..shape.len()).into_iter().rev().collect())
    }
}
