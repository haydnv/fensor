use std::fmt;

use b_table::collate::{Collate, Collator, Overlap, OverlapsRange, OverlapsValue};
use ha_ndarray::{Buffer, CDatatype};
use number_general::{DType, NumberType};
use safecast::AsType;

pub use dense::{DenseFile, DenseTensor};
pub use sparse::{Node, SparseTable, SparseTensor};

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AxisBound {
    At(u64),
    In(u64, u64, u64),
    Of(Vec<u64>),
}

impl AxisBound {
    pub fn is_index(&self) -> bool {
        if let Self::At(_) = self {
            true
        } else {
            false
        }
    }
}

impl OverlapsRange<AxisBound, Collator<u64>> for AxisBound {
    fn overlaps(&self, other: &AxisBound, collator: &Collator<u64>) -> Overlap {
        #[inline]
        fn invert(overlap: Overlap) -> Overlap {
            match overlap {
                Overlap::Less => Overlap::Greater,
                Overlap::Greater => Overlap::Less,

                Overlap::WideLess => Overlap::Narrow,
                Overlap::Wide => Overlap::Narrow,
                Overlap::WideGreater => Overlap::Narrow,

                Overlap::Equal => Overlap::Equal,

                overlap => unreachable!("range overlaps index: {:?}", overlap),
            }
        }

        if self == other {
            return Overlap::Equal;
        }

        match self {
            Self::At(this) => match other {
                Self::At(that) => this.cmp(that).into(),
                Self::In(start, stop, _step) => {
                    let that = *start..*stop;
                    invert(that.overlaps_value(this, collator))
                }
                Self::Of(that) if that.is_empty() => Overlap::Wide,
                Self::Of(that) => invert(to_range(that).overlaps_value(this, collator)),
            },
            Self::In(start, stop, _step) => {
                let this = *start..*stop;

                match other {
                    Self::At(that) => this.overlaps_value(that, collator),
                    Self::In(start, stop, _step) => {
                        let that = *start..*stop;
                        this.overlaps(&that, collator)
                    }
                    Self::Of(that) if that.is_empty() => Overlap::Wide,
                    Self::Of(that) => this.overlaps(&to_range(that), collator),
                }
            }
            Self::Of(this) if this.is_empty() => Overlap::Narrow,
            Self::Of(this) => {
                let this = to_range(this);

                match other {
                    Self::At(that) => this.overlaps_value(that, collator),
                    Self::In(start, stop, _step) => {
                        let that = *start..*stop;
                        this.overlaps(&that, collator)
                    }
                    Self::Of(that) if that.is_empty() => Overlap::Wide,
                    Self::Of(that) => this.overlaps(&to_range(that), collator),
                }
            }
        }
    }
}

impl OverlapsValue<u64, Collator<u64>> for AxisBound {
    fn overlaps_value(&self, value: &u64, collator: &Collator<u64>) -> Overlap {
        match self {
            Self::At(this) => collator.cmp(this, value).into(),
            Self::In(start, stop, _step) => {
                let this = *start..*stop;
                this.overlaps_value(value, collator)
            }
            Self::Of(this) if this.is_empty() => Overlap::Narrow,
            Self::Of(this) => to_range(this).overlaps_value(value, collator),
        }
    }
}

#[inline]
fn to_range(indices: &[u64]) -> std::ops::Range<u64> {
    debug_assert!(!indices.is_empty());
    let start = *indices.iter().fold(&u64::MAX, Ord::min);
    let stop = *indices.iter().fold(&0, Ord::max);
    start..stop
}

impl TryFrom<AxisBound> for ha_ndarray::AxisBound {
    type Error = Error;

    fn try_from(bound: AxisBound) -> Result<Self, Self::Error> {
        match bound {
            AxisBound::At(i) => i
                .try_into()
                .map(ha_ndarray::AxisBound::At)
                .map_err(Error::Index),

            AxisBound::In(start, stop, step) => {
                let start = start.try_into().map_err(Error::Index)?;
                let stop = stop.try_into().map_err(Error::Index)?;
                let step = step.try_into().map_err(Error::Index)?;
                Ok(ha_ndarray::AxisBound::In(start, stop, step))
            }

            AxisBound::Of(indices) => {
                let indices = indices
                    .into_iter()
                    .map(|i| i.try_into().map_err(Error::Index))
                    .collect::<Result<Vec<usize>, Error>>()?;

                Ok(ha_ndarray::AxisBound::Of(indices))
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Bounds(Vec<AxisBound>);

impl Bounds {
    fn all(shape: &[u64]) -> Self {
        let bounds = shape
            .iter()
            .copied()
            .map(|dim| AxisBound::In(0, dim, 1))
            .collect();
        Self(bounds)
    }

    fn normalize(mut self, shape: &[u64]) -> Self {
        for dim in shape.iter().skip(self.0.len()).copied() {
            self.0.push(AxisBound::In(0, dim, 1))
        }

        self
    }
}

impl OverlapsRange<Bounds, Collator<u64>> for Bounds {
    fn overlaps(&self, other: &Bounds, collator: &Collator<u64>) -> Overlap {
        match (self.0.is_empty(), other.0.is_empty()) {
            (true, true) => return Overlap::Equal,
            (true, false) => return Overlap::Greater,
            (false, true) => return Overlap::Narrow,
            (false, false) => {}
        }

        let mut overlap = Overlap::Equal;
        for (this, that) in self.0.iter().zip(&other.0) {
            match this.overlaps(that, collator) {
                Overlap::Less => return Overlap::Less,
                Overlap::Greater => return Overlap::Greater,
                axis_overlap => overlap = overlap.then(axis_overlap),
            }
        }

        overlap
    }
}

impl OverlapsValue<Coord, Collator<u64>> for Bounds {
    fn overlaps_value(&self, value: &Coord, collator: &Collator<u64>) -> Overlap {
        let mut overlap = if self.0.len() == value.len() {
            Overlap::Equal
        } else {
            Overlap::Wide
        };

        for (axis_bound, i) in self.0.iter().zip(value) {
            match axis_bound.overlaps_value(i, collator) {
                Overlap::Less => return Overlap::Less,
                Overlap::Greater => return Overlap::Greater,
                axis_overlap => overlap = overlap.then(axis_overlap),
            }
        }

        overlap
    }
}

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

    fn shape(&self) -> &[u64];

    fn size(&self) -> u64 {
        self.shape().iter().product()
    }
}

impl<T: TensorInstance> TensorInstance for Box<T> {
    fn dtype(&self) -> NumberType {
        (**self).dtype()
    }

    fn shape(&self) -> &[u64] {
        (**self).shape()
    }
}

pub trait TensorTransform: TensorInstance + Sized {
    fn broadcast(self, shape: Shape) -> Result<Self, Error>;

    fn expand(self, axes: Axes) -> Result<Self, Error>;

    fn reshape(self, shape: Shape) -> Result<Self, Error>;

    fn slice(self, bounds: Bounds) -> Result<Self, Error>;

    fn transpose(self, axes: Axes) -> Result<Self, Error>;
}

pub enum Tensor<FE, T> {
    Dense(DenseTensor<FE, T>),
    Sparse(SparseTensor<FE, T>),
}

impl<FE, T> TensorInstance for Tensor<FE, T>
where
    FE: Send + Sync + 'static,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &[u64] {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::Sparse(sparse) => sparse.shape(),
        }
    }
}

impl<FE, T> TensorTransform for Tensor<FE, T>
where
    FE: AsType<Node> + AsType<Buffer<T>> + Send + Sync + 'static,
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
fn validate_bound(bound: &AxisBound, dim: &u64) -> bool {
    match bound {
        AxisBound::At(i) => i < dim,
        AxisBound::In(start, stop, _step) => {
            if start > dim || stop > dim {
                false
            } else {
                start < stop
            }
        }
        AxisBound::Of(indices) => indices.iter().all(|i| i < dim),
    }
}

#[inline]
fn validate_bounds(bounds: &Bounds, shape: &[u64]) -> Result<(), Error> {
    for (x, (bound, dim)) in bounds.0.iter().zip(shape).enumerate() {
        if !validate_bound(bound, dim) {
            return Err(Error::Bounds(format!(
                "invalid bound for axis {} with dimension {}: {:?}",
                x, dim, bound
            )));
        }
    }

    Ok(())
}

#[inline]
fn validate_shape(shape: &[u64]) -> Result<(), Error> {
    if shape.is_empty()
        || shape
            .iter()
            .copied()
            .any(|dim| dim == 0 || dim > u32::MAX as u64)
    {
        Err(Error::Bounds(format!(
            "invalid shape for dense tensor: {:?}",
            shape
        )))
    } else {
        Ok(())
    }
}
