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
    pub fn new(data: *mut T, shape: &[isize], stride: &[isize]) -> Self {
        assert!(shape.len() == stride.len());
        Self { data, shape: shape.to_vec(), stride: stride.to_vec(), conj: false, scalar: T::one() }
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

    pub fn to_scalar(&self) -> Result<T, String> {
        // only size=1 tensors can be converted to scalars
        if self.shape.iter().product::<isize>() == 1 {
            let val = unsafe { *self.data * self.scalar };
            let val = if self.conj { val.conj() } else { val };
            Ok(val)
        } else {
            Err("Tensor is not a scalar".into())
        }
    }

    pub fn set_scalar(&mut self, scalar: T) -> &mut Self {
        self.scalar = scalar;
        self
    }

    pub fn set_conj(&mut self, conj: bool) -> &mut Self {
        self.conj = conj;
        self
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
    /// This function does not check mutability and lifetime of the data pointer. The TBLIS tensor
    /// can always be mutable.
    unsafe fn to_tblis_tensor(&self) -> TblisTensor<T>;
}

/* #endregion */
