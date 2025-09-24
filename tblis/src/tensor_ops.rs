//! Operations for tensors in TBLIS.

#![allow(clippy::too_many_arguments)]

use crate::char_parse::char_parse;
use crate::prelude::*;
use core::ptr::null;
use derive_builder::Builder;
use std::collections::{BTreeMap, BTreeSet};
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

/* #region validity check */

/// Check the validity of subscripts and shapes, and return a size dictionary.
fn check_size_dict(subscripts: &[&str], shapes: &[&[isize]]) -> Result<BTreeMap<char, isize>, String> {
    let mut size_dict = BTreeMap::new();
    if subscripts.len() != shapes.len() {
        return Err(format!("Number of subscripts and shapes do not match: {} vs {}", subscripts.len(), shapes.len()));
    }
    for (subscript, shape) in subscripts.iter().zip(shapes.iter()) {
        let subscript = subscript.chars().collect::<Vec<char>>();
        if subscript.len() != shape.len() {
            return Err(format!("Subscript length and shape length do not match: {subscript:?} vs {shape:?}"));
        }
        for (c, &s) in subscript.iter().zip(shape.iter()) {
            if s < 0 {
                return Err(format!("Invalid dimension size {s} for index {c} in subscript {subscript:?}"));
            }
            if let Some(&existing) = size_dict.get(c) {
                if existing != s {
                    return Err(format!(
                        "Inconsistent dimension size for index {c}: {existing} vs {s} in subscript {subscript:?}"
                    ));
                }
            } else {
                size_dict.insert(*c, s);
            }
        }
    }
    Ok(size_dict)
}

/* #endregion */

/* #region add */

pub use TblisBiCfg as TblisAddCfg;
pub use TblisBiCfgBuilder as TblisAddCfgBuilder;

/// TBLIS tensor addition $B = \alpha A + \beta B$.
///
/// Note that this function can also perform many other operations, such as:
/// - Copy: $B = A$ (set `alpha=1`, `beta=0`).
/// - Scale and add: $B = A + \beta B$ (set `alpha=1`).
/// - Partial sum (trace): $B_i = \sum_j A_{ij}$ (set `alpha=1`, `beta=0`).
/// - Replication: $B_{ij} = A_i$ (set `alpha=1`, `beta=0`).
///
/// # Parameters
///
/// - `a`: The tensor $A$.
/// - `idx_a`: The indices of tensor $A$.
/// - `b`: The tensor $B$ (will be modified in place).
/// - `idx_b`: The indices of tensor $B$.
/// - `cfg`: Optional configuration for the operation.
///   - `comm`: The communicator for parallel execution (default: `null`).
///   - `cntx`: The TBLIS context (default: `null`).
///   - `alpha`: The scalar multiplier $\alpha$ for tensor $A$ (default: `1`).
///   - `beta`: The scalar multiplier $\beta$ for tensor $B$ (default: `1`).
///   - `conja`: Whether to conjugate tensor $A$ (default: `false`).
///   - `conjb`: Whether to conjugate tensor $B$ (default: `false`).
///
/// # Panics
///
/// - If the indices and shapes of the tensors are incompatible.
/// - If both tensors have unique indices (i.e., indices that do not appear in the other tensor).
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_add_f`] for fallible version.
pub unsafe fn tblis_tensor_add<T>(
    a: &TblisTensor<T>,
    idx_a: &str,
    b: &mut TblisTensor<T>,
    idx_b: &str,
    cfg: Option<TblisAddCfg<T>>,
) where
    T: TblisFloatAPI,
{
    unsafe { tblis_tensor_add_f(a, idx_a, b, idx_b, cfg).unwrap() }
}

/// TBLIS tensor addition $B = \alpha A + \beta B$.
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_add`] for infallible version.
pub unsafe fn tblis_tensor_add_f<T>(
    a: &TblisTensor<T>,
    idx_a: &str,
    b: &mut TblisTensor<T>,
    idx_b: &str,
    cfg: Option<TblisAddCfg<T>>,
) -> Result<(), String>
where
    T: TblisFloatAPI,
{
    check_size_dict(&[idx_a, idx_b], &[&a.shape, &b.shape])?;
    let chk_a = idx_a.chars().collect::<BTreeSet<char>>();
    let chk_b = idx_b.chars().collect::<BTreeSet<char>>();
    let chk_ab = &chk_a & &chk_b;
    let chk_a_only = &chk_a - &chk_ab;
    let chk_b_only = &chk_b - &chk_ab;
    if !(chk_a_only.is_empty() || chk_b_only.is_empty()) {
        return Err(format!(
            "tblis_tensor_add: Only one of two tensors can have unique indices. Unique to a ({idx_a}): {chk_a_only:?}, unique to b ({idx_b}): {chk_b_only:?}"
        ));
    }

    let indices = char_parse(&[idx_a, idx_b])?;
    let (a_idx, b_idx) = (indices[0].as_ptr(), indices[1].as_ptr());
    let TblisAddCfg { comm, cntx, alpha, beta, conja, conjb } = cfg.unwrap_or_default();

    let mut a = a.clone();

    a.scalar = alpha;
    b.scalar = beta;
    b.conj = conjb;
    a.conj = conja;

    unsafe { tblis_ffi::tblis::tblis_tensor_add(comm, cntx, &a.to_ffi_tensor(), a_idx, &mut b.to_ffi_tensor(), b_idx) };
    Ok(())
}

/* #endregion */

/* #region dot */

pub use TblisBiCfg as TblisDotCfg;
pub use TblisBiCfgBuilder as TblisDotCfgBuilder;

/// TBLIS tensor dot product $\gamma = A B$.
///
/// # Parameters
///
/// - `a`: The tensor $A$.
/// - `idx_a`: The indices of tensor $A$.
/// - `b`: The tensor $B$.
/// - `idx_b`: The indices of tensor $B$.
/// - `cfg`: Optional configuration for the operation.
///   - `comm`: The communicator for parallel execution (default: `null`).
///   - `cntx`: The TBLIS context (default: `null`).
///   - `alpha`: The scalar multiplier $\alpha$ for tensor $A$ (default: `1`).
///   - `beta`: The scalar multiplier $\beta$ for tensor $B$ (default: `1`).
///   - `conja`: Whether to conjugate tensor $A$ (default: `false`).
///   - `conjb`: Whether to conjugate tensor $B$ (default: `false`).
///
/// # Panics
///
/// - If the indices and shapes of the tensors are incompatible.
/// - If either tensors have unique indices (i.e., indices that do not appear in the other tensor).
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_dot_f`] for fallible version.
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
    unsafe { tblis_tensor_dot_f(a, idx_a, b, idx_b, cfg).unwrap() }
}

/// TBLIS tensor dot product $\gamma = A B$.
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_dot`] for infallible version.
pub unsafe fn tblis_tensor_dot_f<T>(
    a: &TblisTensor<T>,
    idx_a: &str,
    b: &TblisTensor<T>,
    idx_b: &str,
    cfg: Option<TblisDotCfg<T>>,
) -> Result<T, String>
where
    T: TblisFloatAPI,
{
    check_size_dict(&[idx_a, idx_b], &[&a.shape, &b.shape])?;
    let chk_a = idx_a.chars().collect::<BTreeSet<char>>();
    let chk_b = idx_b.chars().collect::<BTreeSet<char>>();
    let chk_ab = &chk_a & &chk_b;
    let chk_a_only = &chk_a - &chk_ab;
    let chk_b_only = &chk_b - &chk_ab;
    if !chk_a_only.is_empty() {
        return Err(format!(
            "tblis_tensor_dot: Unique indices is not allowed. Unique to a ({idx_a}) of b ({idx_b}): {chk_a_only:?}"
        ));
    }
    if !chk_b_only.is_empty() {
        return Err(format!(
            "tblis_tensor_dot: Unique indices is not allowed. Unique to b ({idx_b}) of a ({idx_a}): {chk_b_only:?}"
        ));
    }

    let indices = char_parse(&[idx_a, idx_b])?;
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
    Ok(result)
}

/* #endregion */

/* #region mult */

pub use TblisTriCfg as TblisMultCfg;
pub use TblisTriCfgBuilder as TblisMultCfgBuilder;

/// TBLIS tensor multiplication $C = \alpha A B + \beta C$.
///
/// Note that this function may not only perform tensor contraction, but also hadamard product.
///
/// # Parameters
///
/// - `a`: The tensor $A$.
/// - `idx_a`: The indices of tensor $A$.
/// - `b`: The tensor $B$.
/// - `idx_b`: The indices of tensor $B$.
/// - `c`: The tensor $C$ (will be modified in place).
/// - `idx_c`: The indices of tensor $C$.
/// - `cfg`: Optional configuration for the operation.
///   - `comm`: The communicator for parallel execution (default: `null`).
///   - `cntx`: The TBLIS context (default: `null`).
///   - `alpha`: The scalar multiplier $\alpha$ for tensor $A B$ (default: `1`).
///   - `beta`: The scalar multiplier $\beta$ for tensor $C$ (default: `0`).
///   - `conja`: Whether to conjugate tensor $A$ (default: `false`).
///   - `conjb`: Whether to conjugate tensor $B$ (default: `false`).
///   - `conjc`: Whether to conjugate tensor $C$ (default: `false`).
///
/// # Panics
///
/// - If the indices and shapes of the tensors are incompatible.
/// - If either tensors have unique indices (i.e., indices that do not appear in the other two
///   tensors).
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_mult_f`] for fallible version.
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
    unsafe { tblis_tensor_mult_f(a, idx_a, b, idx_b, c, idx_c, cfg).unwrap() }
}

/// TBLIS tensor multiplication $C = \alpha A B + \beta C$.
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_mult`] for infallible version.
pub unsafe fn tblis_tensor_mult_f<T>(
    a: &TblisTensor<T>,
    idx_a: &str,
    b: &TblisTensor<T>,
    idx_b: &str,
    c: &mut TblisTensor<T>,
    idx_c: &str,
    cfg: Option<TblisMultCfg<T>>,
) -> Result<(), String>
where
    T: TblisFloatAPI,
{
    check_size_dict(&[idx_a, idx_b, idx_c], &[&a.shape, &b.shape, &c.shape])?;
    let chk_a = idx_a.chars().collect::<BTreeSet<char>>();
    let chk_b = idx_b.chars().collect::<BTreeSet<char>>();
    let chk_c = idx_c.chars().collect::<BTreeSet<char>>();
    let chk_ab = &chk_a & &chk_b;
    let chk_ac = &chk_a & &chk_c;
    let chk_bc = &chk_b & &chk_c;
    let chk_a_only = &chk_a - &(&chk_ab | &chk_ac);
    let chk_b_only = &chk_b - &(&chk_ab | &chk_bc);
    let chk_c_only = &chk_c - &(&chk_ac | &chk_bc);
    if !chk_a_only.is_empty() || !chk_b_only.is_empty() || !chk_c_only.is_empty() {
        return Err(format!(
            "tblis_tensor_mult: Unique indices is not allowed. Input and unique indices: a ({idx_a}): {chk_a_only:?}, b ({idx_b}): {chk_b_only:?}, c ({idx_c}): {chk_c_only:?}"
        ));
    }

    let indices = char_parse(&[idx_a, idx_b, idx_c])?;
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
    };
    Ok(())
}

/* #endregion */

/* #region reduce */

/// Reduction operations supported by TBLIS.
///
/// The enum variants correspond to [`tblis_ffi::tblis::reduce_t`].
///
/// This enum can be constructed from strings (case-insensitive, and ignoring `-`, `_`, and spaces),
/// e.g., `"sum"`, `"SumAbs"`, `"NORM-2", etc.
///
/// Supported operations include:
/// - `Sum`: Sum of elements.
/// - `SumAbs`: Sum of absolute values (also known as `Norm1` L1 norm).
/// - `Max`: Maximum value.
/// - `MaxAbs`: Maximum absolute value (also known as `NormInf` Infinity norm).
/// - `Min`: Minimum value.
/// - `MinAbs`: Minimum absolute value.
/// - `Norm2`: Euclidean norm (L2 norm).
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

/// TBLIS tensor reduction $\gamma = \alpha \mathrm{op}(A)$.
///
/// # Parameters
///
/// - `a`: The tensor $A$.
/// - `idx_a`: The indices of tensor $A$ to be reduced. All indices in `idx_a` will be reduced.
/// - `op`: The reduction operation to be performed.
///   - Allowed values include: `Sum`, `SumAbs` (or `Norm1`), `Max` (or `NormInf`), `MaxAbs`, `Min`,
///     `MinAbs`, and `Norm2`.
/// - `cfg`: Optional configuration for the reduction operation.
///   - `comm`: The communicator for parallel execution (default: `null`).
///   - `cntx`: The TBLIS context (default: `null`).
///   - `alpha`: The scalar multiplier $\alpha$ for the result (default: `1`).
///   - `conj`: Whether to conjugate tensor $A$ (default: `false`).
///
/// # Panics
///
/// - If the indices and shapes of the tensor are incompatible.
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_reduce_f`] for fallible version.
pub unsafe fn tblis_tensor_reduce<T>(
    a: &TblisTensor<T>,
    idx_a: &str,
    op: TblisReduceOp,
    cfg: Option<TblisReduceCfg<T>>,
) -> T
where
    T: TblisFloatAPI,
{
    unsafe { tblis_tensor_reduce_f(a, idx_a, op, cfg).unwrap() }
}

/// TBLIS tensor reduction $\gamma = \alpha \mathrm{op}(A)$.
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
pub unsafe fn tblis_tensor_reduce_f<T>(
    a: &TblisTensor<T>,
    idx_a: &str,
    op: TblisReduceOp,
    cfg: Option<TblisReduceCfg<T>>,
) -> Result<T, String>
where
    T: TblisFloatAPI,
{
    check_size_dict(&[idx_a], &[&a.shape]).unwrap();

    let indices = char_parse(&[idx_a])?;
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
    Ok(result)
}

/* #endregion */

/* #region scale */

pub use TblisUniCfg as TblisScaleCfg;
pub use TblisUniCfgBuilder as TblisScaleCfgBuilder;

/// TBLIS tensor scaling $A = \alpha A$ inplace.
///
/// # Parameters
///
/// - `a`: The tensor $A$ (will be modified in place).
/// - `idx_a`: The indices of tensor $A$.
/// - `cfg`: Optional configuration for the operation.
///   - `comm`: The communicator for parallel execution (default: `null`).
///   - `cntx`: The TBLIS context (default: `null`).
///   - `alpha`: The scalar multiplier $\alpha$ for tensor $A$ (default: `1`).
///   - `conj`: Whether to conjugate tensor $A$ (default: `false`).
///
/// # Panics
///
/// - If the indices and shapes of the tensor are incompatible.
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_scale_f`] for fallible version.
pub unsafe fn tblis_tensor_scale<T>(a: &mut TblisTensor<T>, idx_a: &str, cfg: Option<TblisScaleCfg<T>>)
where
    T: TblisFloatAPI,
{
    unsafe { tblis_tensor_scale_f(a, idx_a, cfg).unwrap() }
}

/// TBLIS tensor scaling $A = \alpha A$ inplace.
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_scale`] for infallible version.
pub unsafe fn tblis_tensor_scale_f<T>(
    a: &mut TblisTensor<T>,
    idx_a: &str,
    cfg: Option<TblisScaleCfg<T>>,
) -> Result<(), String>
where
    T: TblisFloatAPI,
{
    check_size_dict(&[idx_a], &[&a.shape]).unwrap();

    let indices = char_parse(&[idx_a])?;
    let a_idx = indices[0].as_ptr();
    let TblisScaleCfg { comm, cntx, alpha, conj } = cfg.unwrap_or_default();

    a.scalar = alpha;
    a.conj = conj;

    unsafe {
        tblis_ffi::tblis::tblis_tensor_scale(comm, cntx, &mut a.to_ffi_tensor(), a_idx);
    };
    Ok(())
}

/* #endregion */

/* #region set */

pub use TblisZeroCfg as TblisSetCfg;
pub use TblisZeroCfgBuilder as TblisSetCfgBuilder;

/// TBLIS tensor set $A = \alpha$ inplace.
///
/// # Parameters
///
/// - `a`: The tensor $A$ (will be modified in place).
/// - `idx_a`: The indices of tensor $A$.
/// - `alpha`: The scalar value to set.
/// - `cfg`: Optional configuration for the operation.
///   - `comm`: The communicator for parallel execution (default: `null`).
///   - `cntx`: The TBLIS context (default: `null`).
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_set_f`] for fallible version.
pub unsafe fn tblis_tensor_set<T>(a: &mut TblisTensor<T>, idx_a: &str, alpha: T, cfg: Option<TblisSetCfg>)
where
    T: TblisFloatAPI,
{
    unsafe { tblis_tensor_set_f(a, idx_a, alpha, cfg).unwrap() }
}

/// TBLIS tensor set $A = \alpha$ inplace.
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_set`] for infallible version.
pub unsafe fn tblis_tensor_set_f<T>(
    a: &mut TblisTensor<T>,
    idx_a: &str,
    alpha: T,
    cfg: Option<TblisSetCfg>,
) -> Result<(), String>
where
    T: TblisFloatAPI,
{
    check_size_dict(&[idx_a], &[&a.shape]).unwrap();

    let indices = char_parse(&[idx_a])?;
    let a_idx = indices[0].as_ptr();
    let TblisSetCfg { comm, cntx } = cfg.unwrap_or_default();

    unsafe {
        tblis_ffi::tblis::tblis_tensor_set(comm, cntx, &alpha.to_ffi_scalar(), &mut a.to_ffi_tensor(), a_idx);
    };
    Ok(())
}

/* #endregion */

/* #region shift */

pub use TblisUniCfg as TblisShiftCfg;
pub use TblisUniCfgBuilder as TblisShiftCfgBuilder;

/// TBLIS tensor shift $A = \alpha + \beta A$ inplace.
///
/// Please note the equation convention here is different to other operations in TBLIS.
///
/// # Parameters
///
/// - `a`: The tensor $A$ (will be modified in place).
/// - `idx_a`: The indices of tensor $A$.
/// - `alpha`: The scalar value to shift.
/// - `cfg`: Optional configuration for the operation.
///   - `comm`: The communicator for parallel execution (default: `null`).
///   - `cntx`: The TBLIS context (default: `null`).
///   - `alpha`: The scalar multiplier $\beta$ for tensor $A$ (default: `1`) **NOTE that this is
///     BETA, not ALPHA in equation**.
///   - `conj`: Whether to conjugate tensor $A$ (default: `false`).
///
/// # Panics
///
/// - If the indices and shapes of the tensor are incompatible.
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_shift_f`] for fallible version.
pub unsafe fn tblis_tensor_shift<T>(a: &mut TblisTensor<T>, idx_a: &str, alpha: T, cfg: Option<TblisShiftCfg<T>>)
where
    T: TblisFloatAPI,
{
    unsafe { tblis_tensor_shift_f(a, idx_a, alpha, cfg).unwrap() }
}

/// TBLIS tensor shift $A = \alpha + \beta A$ inplace.
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_tensor_shift`] for infallible version.
pub unsafe fn tblis_tensor_shift_f<T>(
    a: &mut TblisTensor<T>,
    idx_a: &str,
    alpha: T,
    cfg: Option<TblisShiftCfg<T>>,
) -> Result<(), String>
where
    T: TblisFloatAPI,
{
    check_size_dict(&[idx_a], &[&a.shape]).unwrap();

    let indices = char_parse(&[idx_a])?;
    let a_idx = indices[0].as_ptr();
    let TblisShiftCfg { comm, cntx, alpha: alpha_a, conj } = cfg.unwrap_or_default();

    a.scalar = alpha_a;
    a.conj = conj;

    unsafe {
        tblis_ffi::tblis::tblis_tensor_shift(comm, cntx, &alpha.to_ffi_scalar(), &mut a.to_ffi_tensor(), a_idx);
    };
    Ok(())
}

/* #endregion */
