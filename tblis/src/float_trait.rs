//! TBLIS trait for floating point types.

// use crate::prelude::*;
use duplicate::duplicate_item;
use num::Complex;
use num::complex::ComplexFloat;

pub trait TblisFloatAPI: ComplexFloat {
    const TYPE: tblis_ffi::tblis::type_t;
    fn to_ffi_scalar(&self) -> tblis_ffi::tblis::tblis_scalar;
}

#[duplicate_item(
     T              FLOAT_TYPE                        FLOAT_FIELD ;
    [f32         ] [tblis_ffi::tblis::TYPE_FLOAT   ] [s          ];
    [f64         ] [tblis_ffi::tblis::TYPE_DOUBLE  ] [d          ];
    [Complex<f32>] [tblis_ffi::tblis::TYPE_SCOMPLEX] [c          ];
    [Complex<f64>] [tblis_ffi::tblis::TYPE_DCOMPLEX] [z          ];
)]
impl TblisFloatAPI for T {
    const TYPE: tblis_ffi::tblis::type_t = FLOAT_TYPE;

    fn to_ffi_scalar(&self) -> tblis_ffi::tblis::tblis_scalar {
        tblis_ffi::tblis::tblis_scalar {
            data: tblis_ffi::tblis::tblis_scalar_scalar { FLOAT_FIELD: *self },
            type_: FLOAT_TYPE,
        }
    }
}
