//! TBLIS trait for floating point types.

use crate::prelude::*;
use duplicate::duplicate_item;
use num::Complex;

pub trait TblisFloatAPI: Sized {
    const TYPE: tblis_ffi::tblis::type_t;
    fn from_scalar(s: Self) -> TblisScalar<Self>;
    fn to_scalar(s: &TblisScalar<Self>) -> Self;
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

    fn from_scalar(val: Self) -> TblisScalar<Self> {
        TblisScalar {
            scalar: tblis_ffi::tblis::tblis_scalar {
                data: tblis_ffi::tblis::tblis_scalar_scalar { FLOAT_FIELD: val },
                type_: FLOAT_TYPE,
            },
            _phantom: std::marker::PhantomData,
        }
    }

    fn to_scalar(scalar: &TblisScalar<Self>) -> Self {
        unsafe { scalar.scalar.data.FLOAT_FIELD }
    }
}
