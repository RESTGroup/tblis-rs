//! TBLIS data containers.

use crate::prelude::*;
use core::ffi::c_int;

/* #region TblisTensor */

/// Tensor container for TBLIS operations.
///
/// This struct is a (somehow) safe wrapper of `tblis_ffi::tblis::tblis_tensor`.
///
/// - `data` is a raw pointer to the data. It is defined as mutable for simplicity. It is not safe
///   to access directly.
/// - `shape` is a vector of dimensions. Though we do not allow zero/negative dimensions, the type
///   [`isize`] is used for compatibility with TBLIS.
/// - `stride` is a vector of strides. Though by definition strides can be negative, we have not
///   tested TBLIS with negative strides.
/// - `conj` indicates whether the tensor is to be conjugated during operation. By default it is
///   false.
/// - `scalar` is a scalar multiplier applied to the tensor during operation. By default it is one.
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
    /// Create a new tensor from raw parts.
    ///
    /// This will create a tensor with `conj = false` and `scalar = 1`.
    ///
    /// # Safety
    ///
    /// Though this function is safe to call, the user must ensure that the data pointer is valid
    /// for the given shape and stride, and that the data lives long enough.
    pub fn new(data: *mut T, shape: &[isize], stride: &[isize]) -> Self {
        assert!(shape.len() == stride.len());
        Self { data, shape: shape.to_vec(), stride: stride.to_vec(), conj: false, scalar: T::one() }
    }

    /// (dev-only) Convert to a FFI object `tblis_ffi::tblis::tblis_tensor`.
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

    /// Try to convert to a scalar value.
    ///
    /// Please note that this function not only dereferences the raw pointer, but also applies the
    /// scalar multiplier and conjugation if applicable.
    ///
    /// # Errors
    ///
    /// If the tensor is not a scalar (i.e., its total size is not 1), an error string is returned.
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

    /// Set the scalar multiplier during operation.
    pub fn set_scalar(&mut self, scalar: T) -> &mut Self {
        self.scalar = scalar;
        self
    }

    /// Set whether to conjugate the tensor during operation.
    pub fn set_conj(&mut self, conj: bool) -> &mut Self {
        self.conj = conj;
        self
    }
}

/* #endregion */

/* #region conversion */

/// Trait to convert a tensor view to a TBLIS tensor.
pub trait ToTblisTensor<T>
where
    T: TblisFloatAPI,
{
    /// Convert to a TBLIS tensor.
    ///
    /// # Safety
    ///
    /// Though this function is safe to call, the user must ensure that the data pointer is valid
    /// for the given shape and stride, and that the data lives long enough, and the mutability of
    /// the data pointer is properly handled.
    fn to_tblis_tensor(&self) -> TblisTensor<T>;
}

/* #endregion */
