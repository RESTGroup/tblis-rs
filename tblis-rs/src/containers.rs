//! TBLIS data containers.

use crate::prelude::*;

/* #region TblisScalar */

#[derive(Clone, Copy)]
pub struct TblisScalar<T>
where
    T: TblisFloatAPI,
{
    pub scalar: tblis_ffi::tblis::tblis_scalar,
    pub _phantom: std::marker::PhantomData<T>,
}

impl<T> From<T> for TblisScalar<T>
where
    T: TblisFloatAPI,
{
    fn from(val: T) -> Self {
        TblisFloatAPI::from_scalar(val)
    }
}

impl<T> std::fmt::Debug for TblisScalar<T>
where
    T: TblisFloatAPI + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TblisScalar({:?})", TblisFloatAPI::to_scalar(self))
    }
}

/* #endregion */
