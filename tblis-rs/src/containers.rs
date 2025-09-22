//! TBLIS data containers.

use crate::prelude::*;
use core::ffi::c_int;

/* #region TblisTensor */

#[derive(Debug, Clone)]
pub struct TblisTensor<T>
where
    T: TblisFloatAPI,
{
    pub data: *mut T,
    pub shape: Vec<isize>,
    pub stride: Vec<isize>,
    pub conj: bool,
    pub scalar: T,
}

impl<T> TblisTensor<T>
where
    T: TblisFloatAPI,
{
    pub fn new(data: *mut T, shape: Vec<isize>, stride: Vec<isize>) -> Self {
        assert!(shape.len() == stride.len());
        Self { data, shape, stride, conj: false, scalar: T::one() }
    }

    pub fn to_ffi_tensor(&self) -> tblis_ffi::tblis::tblis_tensor {
        assert!(self.shape.len() == self.stride.len());
        tblis_ffi::tblis::tblis_tensor {
            type_: T::TYPE,
            conj: if self.conj { 1 } else { 0 },
            scalar: self.scalar.to_ffi_scalar(),
            data: self.data as *mut std::ffi::c_void,
            ndim: self.shape.len() as c_int,
            len: self.shape.as_ptr() as *mut isize,
            stride: self.stride.as_ptr() as *mut isize,
        }
    }
}

/* #endregion */

/* #region conversion */

pub trait ToTblisTensor<T>
where
    T: TblisFloatAPI,
{
    /// # Safety
    ///
    /// This function does not check mutability of the data pointer. The TBLIS tensor can
    /// always be mutable.
    unsafe fn to_tblis_tensor(&self) -> TblisTensor<T>;
}

/* #endregion */
