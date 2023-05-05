use std::cmp::Ordering;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Bound;
use std::pin::Pin;
use std::{fmt, io};

use async_trait::async_trait;
use b_table::{Range, TableLock};
use freqfs::DirLock;
use futures::stream::{Stream, StreamExt, TryStreamExt};
use ha_ndarray::*;
use number_general::{DType, Number, NumberCollator, NumberType};
use rayon::prelude::*;
use safecast::{AsType, CastInto};

use super::{AxisBound, Bounds, Coord, Error, Shape, TensorInstance};

mod stream;

const BLOCK_SIZE: usize = 4_096;

pub type Node = b_table::b_tree::Node<Vec<Vec<Number>>>;

#[derive(Clone, Eq, PartialEq)]
pub struct IndexSchema {
    columns: Vec<usize>,
}

impl IndexSchema {
    pub fn new(columns: Vec<usize>) -> Self {
        Self { columns }
    }
}

impl b_table::b_tree::Schema for IndexSchema {
    type Error = Error;
    type Value = Number;

    fn block_size(&self) -> usize {
        BLOCK_SIZE
    }

    fn len(&self) -> usize {
        self.columns.len()
    }

    fn order(&self) -> usize {
        12
    }

    fn validate(&self, key: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if key.len() == self.len() {
            Ok(key)
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidData, "wrong number of values").into())
        }
    }
}

impl b_table::IndexSchema for IndexSchema {
    type Id = usize;

    fn columns(&self) -> &[Self::Id] {
        &self.columns
    }

    fn extract_key(&self, key: &[Self::Value], other: &Self) -> Vec<Self::Value> {
        debug_assert_eq!(key.len(), self.columns.len());
        other.columns.iter().copied().map(|x| key[x]).collect()
    }
}

impl fmt::Debug for IndexSchema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("sparse tensor index schema")
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Schema {
    primary: IndexSchema,
    auxiliary: Vec<(String, IndexSchema)>,
    shape: Shape,
}

impl Schema {
    pub fn new(shape: Shape) -> Self {
        let primary = IndexSchema::new((0..shape.len() + 1).into_iter().collect());
        let mut auxiliary = Vec::with_capacity(shape.len());
        for x in 0..shape.len() {
            let mut columns = Vec::with_capacity(shape.len());
            columns.push(x);

            for xo in 0..shape.len() {
                if xo != x {
                    columns.push(xo);
                }
            }

            let index_schema = IndexSchema::new(columns);
            auxiliary.push((x.to_string(), index_schema));
        }

        Self {
            primary,
            auxiliary,
            shape,
        }
    }
}

impl b_table::Schema for Schema {
    type Id = usize;
    type Error = Error;
    type Value = Number;
    type Index = IndexSchema;

    fn key(&self) -> &[Self::Id] {
        &self.primary.columns[..self.shape.len()]
    }

    fn values(&self) -> &[Self::Id] {
        &self.primary.columns[self.shape.len()..]
    }

    fn primary(&self) -> &Self::Index {
        &self.primary
    }

    fn auxiliary(&self) -> &[(String, IndexSchema)] {
        &self.auxiliary
    }

    fn validate_key(&self, key: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if key.len() == self.shape.len() {
            Ok(key)
        } else {
            let cause = io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid key: {:?}", key),
            );

            Err(cause.into())
        }
    }

    fn validate_values(&self, values: Vec<Self::Value>) -> Result<Vec<Self::Value>, Self::Error> {
        if values.len() == 1 {
            Ok(values)
        } else {
            let cause = io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid values: {:?}", values),
            );

            Err(cause.into())
        }
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("sparse tensor schema")
    }
}

#[async_trait]
pub trait SparseInstance: TensorInstance {
    type DType: CDatatype + DType;

    async fn elements(
        self,
        bounds: Bounds,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Coord, Self::DType), Error>>>>, Error>;
}

pub struct SparseTable<FE, T> {
    table: TableLock<Schema, IndexSchema, NumberCollator, FE>,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for SparseTable<FE, T> {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            dtype: PhantomData,
        }
    }
}

impl<FE: AsType<Node> + Send + Sync, T> SparseTable<FE, T> {
    pub async fn create(dir: DirLock<FE>, shape: Shape) -> Result<Self, Error> {
        let schema = Schema::new(shape);
        let collator = NumberCollator::default();
        let table = TableLock::create(schema, collator, dir)?;

        Ok(Self {
            table,
            dtype: PhantomData,
        })
    }
}

impl<FE, T> TensorInstance for SparseTable<FE, T>
where
    FE: Send + Sync + 'static,
    T: DType + Send + Sync + 'static,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.table.schema().shape
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseTable<FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType,
    Number: CastInto<T>,
{
    type DType = T;

    async fn elements(
        self,
        bounds: Bounds,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Coord, Self::DType), Error>>>>, Error> {
        let range = range_from_bounds(&bounds, self.shape())?;
        let rows = self.table.rows(range, &[], false).await?;
        let elements = rows.map_ok(|row| unwrap_row(row)).map_err(Error::from);
        Ok(Box::pin(elements))
    }
}

impl<FE, T> fmt::Debug for SparseTable<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "sparse table with shape {:?}", self.table.schema().shape)
    }
}

#[derive(Clone)]
pub struct SparseBroadcast<S> {
    source: S,
    shape: Shape,
}

impl<S: TensorInstance + fmt::Debug> SparseBroadcast<S> {
    fn new(source: S, shape: Shape) -> Result<Self, Error> {
        if shape.len() < source.ndim() {
            return Err(Error::Bounds(format!(
                "cannot broadcast {:?} into {:?}",
                source, shape
            )));
        }

        for (dim, bdim) in source
            .shape()
            .iter()
            .zip(shape.iter().skip(shape.len() - source.ndim()))
        {
            if dim == bdim || *dim == 1 {
                // pass
            } else {
                return Err(Error::Bounds(format!(
                    "cannot broadcast dimension {} into {}",
                    dim, bdim
                )));
            }
        }

        Ok(Self { source, shape })
    }
}

impl<S: TensorInstance> TensorInstance for SparseBroadcast<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseBroadcast<S> {
    type DType = S::DType;

    async fn elements(
        self,
        _bounds: Bounds,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Coord, Self::DType), Error>>>>, Error> {
        todo!()
    }
}

#[derive(Clone)]
pub struct SparseExpand<S> {
    source: S,
    shape: Shape,
    axis: usize,
}

impl<S: TensorInstance> TensorInstance for SparseExpand<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseExpand<S> {
    type DType = S::DType;

    async fn elements(
        self,
        mut bounds: Bounds,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Coord, Self::DType), Error>>>>, Error> {
        let source_bounds = if bounds.0.len() > self.axis {
            bounds.0.remove(self.axis);
            bounds
        } else {
            bounds
        };

        let axis = self.axis;
        let ndim = self.ndim();
        debug_assert_eq!(self.source.ndim() + 1, ndim);

        let source_elements = self.source.elements(source_bounds).await?;

        let elements = source_elements.map_ok(move |(mut coord, value)| {
            coord.insert(axis, 0);
            debug_assert_eq!(coord.len(), ndim);
            (coord, value)
        });

        Ok(Box::pin(elements))
    }
}

#[derive(Clone)]
pub struct SparseReshape<S> {
    source: S,
    source_strides: Vec<u64>,
    shape: Shape,
    strides: Vec<u64>,
}

impl<S: TensorInstance> SparseReshape<S> {
    fn new(source: S, shape: Shape) -> Self {
        let source_strides = strides_for(source.shape(), source.ndim());
        let strides = strides_for(&shape, shape.len());

        Self {
            source,
            source_strides,
            shape,
            strides,
        }
    }
}

impl<S: TensorInstance> TensorInstance for SparseReshape<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseReshape<S> {
    type DType = S::DType;

    async fn elements(
        self,
        bounds: Bounds,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Coord, Self::DType), Error>>>>, Error> {
        let source_bounds = if bounds.0.is_empty() {
            Ok(bounds)
        } else {
            Err(Error::Bounds(format!(
                "cannot slice a reshaped sparse tensor (consider making a copy first)"
            )))
        }?;

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let source_ndim = self.source.ndim();
        let source_elements = self.source.elements(source_bounds).await?;
        let source_strides = ArrayBase::new(vec![source_ndim], self.source_strides)?;

        let ndim = self.shape.len();
        let strides = ArrayBase::new(vec![ndim], self.strides)?;
        let shape = ArrayBase::new(vec![ndim], self.shape)?;

        let elements = stream::BlockCoords::new(source_elements, source_ndim)
            .map(move |result| {
                let (source_coords, values) = result?;

                debug_assert_eq!(source_coords.size() % source_ndim, 0);
                debug_assert_eq!(source_coords.size() / source_ndim, values.len());

                let source_strides = source_strides.broadcast(vec![values.len(), source_ndim])?;

                let offsets = source_coords.mul(&source_strides)?;
                let offsets = offsets.sum_axis(1)?;

                let broadcast = vec![offsets.size(), ndim];
                let strides = strides.broadcast(broadcast.to_vec())?;
                let offsets = offsets
                    .expand_dims(vec![1])?
                    .broadcast(broadcast.to_vec())?;

                let dims = shape.expand_dims(vec![0])?.broadcast(broadcast)?;
                let coords = (offsets / strides) % dims;

                let coords = coords.to_vec(&queue)?;

                Result::<_, Error>::Ok((coords, values))
            })
            .map_ok(move |(coords, values)| {
                let coords = coords.into_par_iter().chunks(ndim).collect::<Vec<Coord>>();
                futures::stream::iter(coords.into_iter().zip(values).map(Ok))
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }
}

#[derive(Clone)]
pub struct SparseSlice<S> {
    source: S,
    bounds: Bounds,
    shape: Shape,
}

impl<S> SparseSlice<S>
where
    S: TensorInstance + fmt::Debug,
{
    fn new(source: S, bounds: Bounds) -> Result<Self, Error> {
        if bounds.0.len() > source.ndim() {
            return Err(Error::Bounds(format!(
                "invalid slice bounds for {:?}: {:?}",
                source, bounds.0
            )));
        }

        let mut shape = Vec::with_capacity(source.ndim());
        for (x, bound) in bounds.0.iter().enumerate() {
            match bound {
                AxisBound::At(_) => {} // no-op
                AxisBound::In(start, stop, 1) => {
                    shape.push(stop - start);
                }
                axis_bound => {
                    return Err(Error::Bounds(format!(
                        "invalid bound for sparse tensor axis {}: {:?}",
                        x, axis_bound
                    )));
                }
            }
        }

        shape.extend_from_slice(&source.shape()[bounds.0.len()..]);

        Ok(Self {
            source,
            bounds,
            shape,
        })
    }
}

impl<S> TensorInstance for SparseSlice<S>
where
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<S> SparseInstance for SparseSlice<S>
where
    S: SparseInstance + fmt::Debug,
{
    type DType = S::DType;

    async fn elements(
        self,
        bounds: Bounds,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Coord, Self::DType), Error>>>>, Error> {
        let source_bounds = if bounds.0.is_empty() {
            self.bounds
        } else if bounds.0.len() > self.ndim() {
            return Err(Error::Bounds(format!(
                "invalid bounds for {:?}: {:?}",
                self, bounds.0
            )));
        } else {
            let mut source_bounds = Vec::with_capacity(self.source.ndim());

            let mut axis = 0;
            for bound in self.bounds.0.into_iter().take(bounds.0.len()) {
                match bound {
                    AxisBound::At(i) => {
                        source_bounds.push(AxisBound::At(i));
                    }
                    AxisBound::In(start, stop, 1) => {
                        let source_bound = match &bounds.0[axis] {
                            AxisBound::At(i) => {
                                let i = start + i;
                                if i < stop {
                                    Ok(AxisBound::At(i))
                                } else {
                                    Err(Error::Bounds(format!(
                                        "index {} is out of bounds for axis {}",
                                        i, axis
                                    )))
                                }
                            }
                            AxisBound::In(start, stop, 1) => {
                                let (source_start, source_stop) = (start + start, stop + start);
                                if source_stop <= *stop {
                                    Ok(AxisBound::In(source_start, source_stop, 1))
                                } else {
                                    Err(Error::Bounds(format!(
                                        "range [{}, {}) is out of bounds for axis {}",
                                        source_start, source_stop, axis
                                    )))
                                }
                            }
                            bound => Err(Error::Bounds(format!(
                                "invalid bound for axis {}: {:?}",
                                axis, bound
                            ))),
                        }?;

                        source_bounds.push(source_bound);
                        axis += 1;
                    }
                    bound => {
                        return Err(Error::Bounds(format!(
                            "invalid bound for sparse tensor: {:?}",
                            bound
                        )))
                    }
                }
            }

            Bounds(source_bounds)
        };

        self.source.elements(source_bounds).await
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseSlice<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "slice of {:?} with bounds {:?}",
            self.source, self.bounds
        )
    }
}

#[derive(Clone)]
pub struct SparseTranspose<S> {
    source: S,
    permutation: Vec<usize>,
    shape: Shape,
}

impl<S> TensorInstance for SparseTranspose<S>
where
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[async_trait]
impl<S> SparseInstance for SparseTranspose<S>
where
    S: SparseInstance + fmt::Debug,
{
    type DType = S::DType;

    async fn elements(
        self,
        mut bounds: Bounds,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Coord, Self::DType), Error>>>>, Error> {
        let bounds = match bounds.0.len().cmp(&self.ndim()) {
            Ordering::Equal => Ok(bounds),
            Ordering::Greater => Err(Error::Bounds(format!(
                "invalid bounds for {:?}: {:?}",
                self, bounds
            ))),
            Ordering::Less => {
                let dims = self.shape.iter().skip(bounds.0.len()).copied();
                bounds.0.extend(dims.map(|dim| AxisBound::In(0, dim, 1)));
                Ok(bounds)
            }
        }?;

        let ndim = self.ndim();
        let size = self.size();
        let permutation = self.permutation;

        let source_bounds = permutation
            .iter()
            .copied()
            .map(|x| bounds.0[x].clone())
            .collect();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(size))?;
        let source_elements = self.source.elements(source_bounds).await?;
        let blocks = stream::BlockCoords::new(source_elements, ndim);
        let elements = blocks
            .map(move |result| {
                let (source_coords, values) = result?;
                let coords = source_coords.transpose(Some(permutation.to_vec()))?;
                Result::<_, Error>::Ok((coords.to_vec(&queue)?, values))
            })
            .map_ok(move |(coords, values)| {
                let coords = coords.into_par_iter().chunks(ndim).collect::<Vec<Coord>>();
                (coords, values)
            })
            .map_ok(|(coords, values)| {
                futures::stream::iter(coords.into_iter().zip(values).map(Result::<_, Error>::Ok))
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseTranspose<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "transpose of {:?} with permutation {:?}",
            self.source, self.permutation
        )
    }
}

#[inline]
fn size_hint(size: u64) -> usize {
    size.try_into().ok().unwrap_or_else(|| usize::MAX)
}

#[inline]
fn strides_for(shape: &[u64], ndim: usize) -> Vec<u64> {
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
fn unwrap_row<T>(mut row: Vec<Number>) -> (Coord, T)
where
    Number: CastInto<T> + CastInto<u64>,
{
    let n = row.pop().expect("n").cast_into();
    let coord = row.into_iter().map(|i| i.cast_into()).collect();
    (coord, n)
}

#[inline]
fn range_from_bounds(bounds: &Bounds, shape: &[u64]) -> Result<Range<usize, Number>, Error> {
    if bounds.0.is_empty() {
        return Ok(Range::default());
    } else if bounds.0.len() > shape.len() {
        return Err(Error::Bounds(format!(
            "invalid bounds for shape {:?}: {:?}",
            shape, bounds.0
        )));
    }

    let mut range = HashMap::new();

    for (x, bound) in bounds.0.iter().enumerate() {
        match bound {
            AxisBound::At(i) => {
                if *i >= shape[x] {
                    return Err(Error::Bounds(format!(
                        "invalid index for axis {}: {}",
                        x, i
                    )));
                }

                range.insert(x, b_table::ColumnRange::Eq(Number::from(*i)));
            }
            AxisBound::In(start, stop, step) => {
                if stop < start {
                    return Err(Error::Bounds(format!(
                        "invalid range for axis {}: [{}, {})",
                        x, start, stop
                    )));
                } else if *step != 1 {
                    return Err(Error::Bounds(format!(
                        "sparse tensor does not support stride {}",
                        step
                    )));
                } else if *stop > shape[x] {
                    return Err(Error::Bounds(format!(
                        "index {} is out of bounds for dimension {}",
                        stop, shape[x]
                    )));
                }

                let start = Bound::Included(Number::from(*start));
                let stop = Bound::Excluded(Number::from(*stop));
                range.insert(x, b_table::ColumnRange::In((start, stop)));
            }
            AxisBound::Of(indices) => {
                return Err(Error::Bounds(format!(
                    "sparse tensor does not support axis bound {:?}",
                    indices
                )));
            }
        }
    }

    Ok(range.into())
}
