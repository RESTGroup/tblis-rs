//! TBLIS wrapper and several minimal implementations.
//!
//! # API Documentation Summary
//!
//! ## About function [`tblis_einsum`]
//!
//! This is the most important function of this crate.
//!
//! For the parameters of this function, also refer to crate [opt-einsum-path](https://github.com/RESTGroup/opt-einsum-path), document of function [`contract_path`](https://docs.rs/opt-einsum-path/latest/opt_einsum_path/contract/fn.contract_path.html).
//!
//! ## Most important functions, structs and traits
//!
//! | Item | Description |
//! |--|--|
//! | fn [`tblis_einsum`] | Einstein summation |
//! | fn [`tblis_einsum_f`] | Einstein summation (failable) |
//! | struct [`TblisTensor`] | Tensor struct of this crate |
//! | trait [`TblisFloatAPI`] | Float trait for TBLIS operations (f32, f64, c32, c64) |
//!
//! ## Functions
//!
//! | Item | Description |
//! |--|--|
//! | [`tblis_einsum`] | (high-level) Einstein summation |
//! | [`tblis_einsum_f`] | (high-level) Einstein summation (failable) |
//! | [`tblis_tensor_add`] | (1t-level) $B = \alpha A + \beta B$ |
//! | [`tblis_tensor_scale`] | (1t-level) $A = \alpha A$ |
//! | [`tblis_tensor_set`] | (1t-level) $A = \alpha$ |
//! | [`tblis_tensor_shift`] | (1t-level) $A = \alpha + \beta A$ |
//! | [`tblis_tensor_reduce`] | (1t-level) $\gamma = op(A)$ |
//! | [`tblis_tensor_dot`] | (1t-level) $\gamma = A B$ |
//! | [`tblis_tensor_mult`] | (3t-level) $C = \alpha A B + \beta C$ |
//!
//! ## Traits
//!
//! | Item | Description |
//! |--|--|
//! | [`TblisFloatAPI`] | Float trait for TBLIS operations (f32, f64, c32, c64) |
//! | [`ToTblisTensor`] | Tensor view convert to TBLIS (mutable) tensor<br>Method function [`ToTblisTensor::to_tblis_tensor`] |
//! | [`ArrayFromTblisTensor`] | Convert from TBLIS tensor to ndarray object<br>Method function [`ArrayFromTblisTensor::into_array`]<br>Only available for crate feature `ndarray` |
//!
//! ## Configurations
//!
//! | Item | Description |
//! |--|--|
//! | [`TblisReduceOp`] | Reduction operations for [`tblis_tensor_reduce`]<br>- Supported operations: sum, sumabs/norm1, max, maxabs/norminf, min, minabs, norm2 |
//! | [`TblisZeroCfg`]<br>[`TblisZeroCfgBuilder`] | Zero parameter configuration for TBLIS operations<br>- Used in [`tblis_tensor_set`]<br>- Aliased as [`TblisSetCfg`] |
//! | [`TblisUniCfg`]<br>[`TblisUniCfgBuilder`] | One parameter configuration for TBLIS operations<br>- By default $\alpha = 1$, no conjugate<br>- Used in [`tblis_tensor_scale`], [`tblis_tensor_shift`], [`tblis_tensor_reduce`]<br>- Aliased as [`TblisScaleCfg`], [`TblisShiftCfg`], [`TblisReduceCfg`] |
//! | [`TblisBiCfg`]<br>[`TblisBiCfgBuilder`] | Two parameter configuration for TBLIS operations<br>- By default $\alpha = 1, \beta = 1$, no conjugate<br>- Used in [`tblis_tensor_add`], [`tblis_tensor_dot`]<br>- Aliased as [`TblisAddCfg`], [`TblisDotCfg`] |
//! | [`TblisTriCfg`]<br>[`TblisTriCfgBuilder`] | Three parameter configuration for TBLIS operations<br>- By default $\alpha = 1, \beta = 0$, no conjugate<br>- Used in [`tblis_tensor_mult`]<br>- Aliased as [`TblisMultCfg`] |
#![doc = include_str!("../readme.md")]

pub mod alloc_vec;
pub mod char_parse;
pub mod containers;
pub mod einsum_impl;
pub mod float_trait;
pub mod tensor_ops;

pub mod prelude {
    pub use crate::containers::*;
    pub use crate::einsum_impl::*;
    pub use crate::float_trait::*;
    pub use crate::tensor_ops::*;
}

#[allow(unused_imports)]
use prelude::*;
