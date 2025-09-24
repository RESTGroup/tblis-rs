//! Operations for tensors in TBLIS.

#![allow(clippy::too_many_arguments)]

use crate::char_parse::char_parse;
use crate::prelude::*;
use core::ptr::null;
use derive_builder::Builder;
use tblis_ffi::tblis::{tblis_comm, tblis_config};

/* #region cfg builder */

#[non_exhaustive]
#[derive(Builder, Debug, Clone)]
pub struct TblisTriCfg<T>
where
    T: TblisFloatAPI,
{
    #[builder(default = "null()")]
    pub comm: *const tblis_comm,
    #[builder(default = "null()")]
    pub cntx: *const tblis_config,
    #[builder(default = "T::one()")]
    pub alpha: T,
    #[builder(default = "T::zero()")]
    pub beta: T,
    #[builder(default = "false")]
    pub conja: bool,
    #[builder(default = "false")]
    pub conjb: bool,
}

impl<T> Default for TblisTriCfg<T>
where
    T: TblisFloatAPI,
{
    fn default() -> Self {
        TblisTriCfgBuilder::default().build().unwrap()
    }
}

#[non_exhaustive]
#[derive(Builder, Debug, Clone)]
pub struct TblisBiCfg<T>
where
    T: TblisFloatAPI,
{
    #[builder(default = "null()")]
    pub comm: *const tblis_comm,
    #[builder(default = "null()")]
    pub cntx: *const tblis_config,
    #[builder(default = "T::one()")]
    pub alpha: T,
    #[builder(default = "T::one()")]
    pub beta: T,
    #[builder(default = "false")]
    pub conja: bool,
    #[builder(default = "false")]
    pub conjb: bool,
}

impl<T> Default for TblisBiCfg<T>
where
    T: TblisFloatAPI,
{
    fn default() -> Self {
        TblisBiCfgBuilder::default().build().unwrap()
    }
}

#[non_exhaustive]
#[derive(Builder, Debug, Clone)]
pub struct TblisUniCfg<T>
where
    T: TblisFloatAPI,
{
    #[builder(default = "null()")]
    pub comm: *const tblis_comm,
    #[builder(default = "null()")]
    pub cntx: *const tblis_config,
    #[builder(default = "T::one()")]
    pub alpha: T,
    #[builder(default = "false")]
    pub conj: bool,
}

impl<T> Default for TblisUniCfg<T>
where
    T: TblisFloatAPI,
{
    fn default() -> Self {
        TblisUniCfgBuilder::default().build().unwrap()
    }
}

#[non_exhaustive]
#[derive(Builder, Debug, Clone)]
pub struct TblisZeroCfg {
    #[builder(default = "null()")]
    pub comm: *const tblis_comm,
    #[builder(default = "null()")]
    pub cntx: *const tblis_config,
}

impl Default for TblisZeroCfg {
    fn default() -> Self {
        TblisZeroCfgBuilder::default().build().unwrap()
    }
}

/* #endregion */

/* #region add */

pub use TblisBiCfg as TblisAddCfg;
pub use TblisBiCfgBuilder as TblisAddCfgBuilder;

/// # Safety
///
/// - This function does not check tensor data validity and mutability.
pub unsafe fn tblis_tensor_add<T>(
    a: &TblisTensor<T>,
    idx_a: &str,
    b: &mut TblisTensor<T>,
    idx_b: &str,
    cfg: Option<TblisAddCfg<T>>,
) where
    T: TblisFloatAPI,
{
    let indices = char_parse(&[idx_a, idx_b]);
    let (a_idx, b_idx) = (indices[0].as_ptr(), indices[1].as_ptr());
    let TblisAddCfg { comm, cntx, alpha, beta, conja, conjb } = cfg.unwrap_or_default();

    let mut a = a.clone();

    a.scalar = alpha;
    b.scalar = beta;
    b.conj = conjb;
    a.conj = conja;

    unsafe {
        tblis_ffi::tblis::tblis_tensor_add(comm, cntx, &a.to_ffi_tensor(), a_idx, &mut b.to_ffi_tensor(), b_idx);
    }
}

/* #endregion */

/* #region dot */

pub use TblisBiCfg as TblisDotCfg;
pub use TblisBiCfgBuilder as TblisDotCfgBuilder;

/// # Safety
///
/// - This function does not check tensor data validity and mutability.
pub unsafe fn tblis_tensor_dot<T>(
    a: &TblisTensor<T>,
    idx_a: &str,
    b: &TblisTensor<T>,
    idx_b: &str,
    cfg: Option<TblisDotCfg<T>>,
) -> T
where
    T: TblisFloatAPI,
{
    let indices = char_parse(&[idx_a, idx_b]);
    let (a_idx, b_idx) = (indices[0].as_ptr(), indices[1].as_ptr());
    let TblisDotCfg { comm, cntx, alpha, beta, conja, conjb } = cfg.unwrap_or_default();

    let mut a = a.clone();
    let mut b = b.clone();

    a.scalar = alpha;
    b.scalar = beta;
    a.conj = conja;
    b.conj = conjb;

    let result = T::zero();
    unsafe {
        tblis_ffi::tblis::tblis_tensor_dot(
            comm,
            cntx,
            &a.to_ffi_tensor(),
            a_idx,
            &b.to_ffi_tensor(),
            b_idx,
            &mut result.to_ffi_scalar(),
        );
    }
    result
}

/* #endregion */

/* #region mult */

pub use TblisTriCfg as TblisMultCfg;
pub use TblisTriCfgBuilder as TblisMultCfgBuilder;

/// # Safety
///
/// - This function does not check tensor data validity and mutability.
pub unsafe fn tblis_tensor_mult<T>(
    a: &TblisTensor<T>,
    idx_a: &str,
    b: &TblisTensor<T>,
    idx_b: &str,
    c: &mut TblisTensor<T>,
    idx_c: &str,
    cfg: Option<TblisMultCfg<T>>,
) where
    T: TblisFloatAPI,
{
    let indices = char_parse(&[idx_a, idx_b, idx_c]);
    let (a_idx, b_idx, c_idx) = (indices[0].as_ptr(), indices[1].as_ptr(), indices[2].as_ptr());
    let TblisMultCfg { comm, cntx, alpha, beta, conja, conjb } = cfg.unwrap_or_default();

    let mut a = a.clone();
    let mut b = b.clone();

    a.scalar = alpha;
    b.scalar = T::one();
    c.scalar = beta;
    b.conj = conjb;
    a.conj = conja;
    c.conj = false;

    unsafe {
        tblis_ffi::tblis::tblis_tensor_mult(
            comm,
            cntx,
            &a.to_ffi_tensor(),
            a_idx,
            &b.to_ffi_tensor(),
            b_idx,
            &mut c.to_ffi_tensor(),
            c_idx,
        );
    }
}

/* #endregion */

/* #region reduce */

#[repr(u32)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TblisReduceOp {
    Sum = tblis_ffi::tblis::reduce_t::REDUCE_SUM as _,
    SumAbs = tblis_ffi::tblis::reduce_t::REDUCE_SUM_ABS as _,
    Max = tblis_ffi::tblis::reduce_t::REDUCE_MAX as _,
    MaxAbs = tblis_ffi::tblis::reduce_t::REDUCE_MAX_ABS as _,
    Min = tblis_ffi::tblis::reduce_t::REDUCE_MIN as _,
    MinAbs = tblis_ffi::tblis::reduce_t::REDUCE_MIN_ABS as _,
    Norm2 = tblis_ffi::tblis::reduce_t::REDUCE_NORM_2 as _,
}

#[allow(non_upper_case_globals)]
impl TblisReduceOp {
    pub const Norm1: Self = Self::SumAbs;
    pub const NormInf: Self = Self::MaxAbs;
}

impl From<&str> for TblisReduceOp {
    fn from(s: &str) -> Self {
        let st = s.to_lowercase().replace(['-', ' ', '_'], "");
        // simplify the previous code
        match st.as_str() {
            "sum" => Self::Sum,
            "sumabs" => Self::SumAbs,
            "max" => Self::Max,
            "maxabs" => Self::MaxAbs,
            "min" => Self::Min,
            "minabs" => Self::MinAbs,
            "norm2" => Self::Norm2,
            "norm1" => Self::Norm1,
            "norminf" => Self::NormInf,
            _ => panic!("Invalid reduction operation string: {s}"),
        }
    }
}

impl From<TblisReduceOp> for tblis_ffi::tblis::reduce_t {
    fn from(op: TblisReduceOp) -> Self {
        match op {
            TblisReduceOp::Sum => tblis_ffi::tblis::reduce_t::REDUCE_SUM,
            TblisReduceOp::SumAbs => tblis_ffi::tblis::reduce_t::REDUCE_SUM_ABS,
            TblisReduceOp::Max => tblis_ffi::tblis::reduce_t::REDUCE_MAX,
            TblisReduceOp::MaxAbs => tblis_ffi::tblis::reduce_t::REDUCE_MAX_ABS,
            TblisReduceOp::Min => tblis_ffi::tblis::reduce_t::REDUCE_MIN,
            TblisReduceOp::MinAbs => tblis_ffi::tblis::reduce_t::REDUCE_MIN_ABS,
            TblisReduceOp::Norm2 => tblis_ffi::tblis::reduce_t::REDUCE_NORM_2,
        }
    }
}

pub use TblisUniCfg as TblisReduceCfg;
pub use TblisUniCfgBuilder as TblisReduceCfgBuilder;

/// # Safety
///
/// - This function does not check tensor data validity and mutability.
pub unsafe fn tblis_tensor_reduce<T>(
    a: &TblisTensor<T>,
    idx_a: &str,
    op: TblisReduceOp,
    cfg: Option<TblisReduceCfg<T>>,
) -> T
where
    T: TblisFloatAPI,
{
    let indices = char_parse(&[idx_a]);
    let a_idx = indices[0].as_ptr();
    let TblisReduceCfg { comm, cntx, alpha, conj } = cfg.unwrap_or_default();

    let mut a = a.clone();
    a.scalar = alpha;
    a.conj = conj;
    let op = op.into();

    let result = T::zero();
    let mut idx = 0_isize;
    unsafe {
        tblis_ffi::tblis::tblis_tensor_reduce(
            comm,
            cntx,
            op,
            &a.to_ffi_tensor(),
            a_idx,
            &mut result.to_ffi_scalar(),
            &mut idx,
        );
    }
    result
}

/* #endregion */

/* #region scale */

pub use TblisUniCfg as TblisScaleCfg;
pub use TblisUniCfgBuilder as TblisScaleCfgBuilder;

/// # Safety
///
/// - This function does not check tensor data validity and mutability.
pub unsafe fn tblis_tensor_scale<T>(a: &mut TblisTensor<T>, idx_a: &str, cfg: Option<TblisScaleCfg<T>>)
where
    T: TblisFloatAPI,
{
    let indices = char_parse(&[idx_a]);
    let a_idx = indices[0].as_ptr();
    let TblisScaleCfg { comm, cntx, alpha, conj } = cfg.unwrap_or_default();

    a.scalar = alpha;
    a.conj = conj;

    unsafe {
        tblis_ffi::tblis::tblis_tensor_scale(comm, cntx, &mut a.to_ffi_tensor(), a_idx);
    }
}

/* #endregion */

/* #region set */

pub use TblisZeroCfg as TblisSetCfg;
pub use TblisZeroCfgBuilder as TblisSetCfgBuilder;

/// # Safety
///
/// - This function does not check tensor data validity and mutability.
pub unsafe fn tblis_tensor_set<T>(a: &mut TblisTensor<T>, idx_a: &str, alpha: T, cfg: Option<TblisSetCfg>)
where
    T: TblisFloatAPI,
{
    let indices = char_parse(&[idx_a]);
    let a_idx = indices[0].as_ptr();
    let TblisSetCfg { comm, cntx } = cfg.unwrap_or_default();

    unsafe {
        tblis_ffi::tblis::tblis_tensor_set(comm, cntx, &alpha.to_ffi_scalar(), &mut a.to_ffi_tensor(), a_idx);
    }
}

/* #endregion */

/* #region shift */

pub use TblisUniCfg as TblisShiftCfg;
pub use TblisUniCfgBuilder as TblisShiftCfgBuilder;

/// # Safety
///
/// - This function does not check tensor data validity and mutability.
pub unsafe fn tblis_tensor_shift<T>(a: &mut TblisTensor<T>, idx_a: &str, alpha: T, cfg: Option<TblisShiftCfg<T>>)
where
    T: TblisFloatAPI,
{
    let indices = char_parse(&[idx_a]);
    let a_idx = indices[0].as_ptr();
    let TblisShiftCfg { comm, cntx, alpha: alpha_a, conj } = cfg.unwrap_or_default();

    a.scalar = alpha_a;
    a.conj = conj;

    unsafe {
        tblis_ffi::tblis::tblis_tensor_shift(comm, cntx, &alpha.to_ffi_scalar(), &mut a.to_ffi_tensor(), a_idx);
    }
}

/* #endregion */
