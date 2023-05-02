use derive_more::Display;

pub mod dense;
pub mod sparse;

pub type Coord = Vec<u64>;

#[derive(Debug, Display)]
pub enum Error {
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
