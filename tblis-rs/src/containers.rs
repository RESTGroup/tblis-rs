//! TBLIS data containers.

use crate::prelude::*;
use core::ffi::c_int;

/* #region TblisTensor */

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
