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
