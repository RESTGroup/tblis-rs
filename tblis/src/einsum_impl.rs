//! Implementation of einsum operations preparations.
//!
//! For crate `ndarray`, also implements einsum execution.

use crate::prelude::*;
use opt_einsum_path::typing::{ContractionType, SizeLimitType, TensorShapeType};
use opt_einsum_path::{PathOptimizer, contract_path};
use std::collections::BTreeSet;

/// (dev-only) Intermediate representation of einsum contraction step.
///
/// This is used to represent each contraction step in the optimized contraction path.
/// - `indices`: indices (step of a path) of tensors involved in this contraction step.
/// - `idx_a`: einsum subscript of the first tensor.
/// - `idx_b`: einsum subscript of the second tensor (None for single tensor operations).
/// - `idx_c`: einsum subscript of the output tensor.
/// - `shape_c`: shape of the output tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TblisContractStep {
    pub indices: Vec<usize>,
    pub idx_a: String,
    pub idx_b: Option<String>,
    pub idx_c: String,
    pub shape_c: Vec<isize>,
}

/// (dev-only) Prepare einsum contraction steps for TBLIS internally from output of
/// [opt_einsum_path::contract_path].
///
/// # Panics
///
/// A single step in contraction involves more than two tensors. This is not supported in TBLIS
/// (TBLIS only supports trace/scale/set of one tensor, and add/mult for two tensors).
///
/// # See also
///
/// [`tblis_einsum_prep_f`] for fallible version.
pub fn tblis_einsum_prep(
    subscripts: &str,
    operands: &[TensorShapeType],
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
) -> Vec<TblisContractStep> {
    tblis_einsum_prep_f(subscripts, operands, optimize, memory_limit).unwrap()
}

/// (dev-only) Prepare einsum contraction steps for TBLIS internally from output of
/// [opt_einsum_path::contract_path].
///
/// # See also
///
/// [`tblis_einsum_prep`] for non-fallible version.
pub fn tblis_einsum_prep_f(
    subscripts: &str,
    operands: &[TensorShapeType],
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
) -> Result<Vec<TblisContractStep>, String> {
    let (_, path_info) = contract_path(subscripts, operands, optimize, memory_limit)?;
    let size_dict = path_info.size_dict;
    let mut steps = Vec::new();
    for contract_step in path_info.contraction_list {
        let ContractionType { indices, einsum_str, .. } = contract_step;
        // Only pairwise / single tensor contractions are supported.
        match indices.len() {
            2 => {
                let einsum_str_split: Vec<&str> = einsum_str.split("->").collect();
                let idx_c = einsum_str_split[1].to_string();
                let idx_ab_split = einsum_str_split[0].split(',').collect::<Vec<&str>>();
                let idx_a = idx_ab_split[0].to_string();
                let idx_b = idx_ab_split[1].to_string();
                let shape_c = idx_c.chars().map(|c| size_dict[&c] as isize).collect();
                steps.push(TblisContractStep { indices, idx_a, idx_b: Some(idx_b), idx_c, shape_c });
            },
            1 => {
                let einsum_str_split: Vec<&str> = einsum_str.split("->").collect();
                let idx_a = einsum_str_split[0].to_string();
                let idx_c = einsum_str_split[1].to_string();
                let shape_c = idx_c.chars().map(|c| size_dict[&c] as isize).collect();
                steps.push(TblisContractStep { indices, idx_a, idx_b: None, idx_c, shape_c });
            },
            _ => return Err("Only pairwise / single tensor contractions are supported.".to_string()),
        }
    }
    Ok(steps)
}

/// Generate strides by shape and row-major/col-major flag.
fn shape_to_stride(shape: &[isize], row_major: bool) -> Vec<isize> {
    let ndim = shape.len();
    let mut stride = vec![1isize; ndim];
    if ndim == 0 {
        return stride;
    }
    if row_major {
        stride[ndim - 1] = 1;
        for i in (0..ndim - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
    } else {
        stride[0] = 1;
        for i in 1..ndim {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
    }
    stride
}

/// Perform trace operation on a tensor. This can be used when [`tblis_tensor_mult`] could not
/// handle the case where `idx_a` or `idx_b` contains redundant indices.
fn tblis_trace_f<T>(
    subscript_prev: &str,
    subscript_traced: &str,
    tsr_prev: &TblisTensor<T>,
    row_major: bool,
) -> Result<(Vec<T>, TblisTensor<T>), String>
where
    T: TblisFloatAPI,
{
    let idx_prev: Vec<char> = subscript_prev.chars().collect();
    let idx_traced: Vec<char> = subscript_traced.chars().collect();
    let subscript_traced = idx_traced.iter().collect::<String>();
    if idx_prev.len() != tsr_prev.shape.len() {
        return Err("Subscript length does not match tensor ndim.".to_string());
    }
    // build shape/stride for traced tensor
    let shape_traced = idx_traced
        .iter()
        .map(|&c| tsr_prev.shape[idx_prev.iter().position(|&x| x == c).unwrap()])
        .collect::<Vec<isize>>();
    let stride_traced = shape_to_stride(&shape_traced, row_major);
    let size_traced = shape_traced.iter().product::<isize>() as usize;
    let vec_traced = unsafe { crate::alloc_vec::uninitialized_vec::<T>(size_traced)? };
    let mut tsr_traced = TblisTensor::new(vec_traced.as_ptr() as *mut T, &shape_traced, &stride_traced);
    let cfg = TblisAddCfgBuilder::default().beta(T::zero()).build().unwrap();
    unsafe { tblis_tensor_add(tsr_prev, subscript_prev, &mut tsr_traced, &subscript_traced, Some(cfg)) };
    Ok((vec_traced, tsr_traced))
}

/// Perform einsum operation using TBLIS.
///
/// # Parameters
///
/// - `subscripts`: einsum subscripts, e.g. `"ij,jk->ik"`.
/// - `operands`: list of input tensors (see [`TblisTensor`] for data structure and
///   [`ToTblisTensor`] for conversion trait).
/// - `optimize`: contraction path optimization strategy (see [`opt_einsum_path::contract_path`]).
/// - `memory_limit`: memory limit for contraction path optimization (see
///   [`opt_einsum_path::contract_path`]).
/// - `row_major`: whether the input/output tensors are in row-major (C-style) or col-major
///   (Fortran-style).
/// - `out_tblis_tensor`: pre-allocated output tensor. If `None`, the output tensor is allocated
///   internally.
///
/// # Returns
///
/// - `Option<(Vec<T>, TblisTensor<T>)>`: If the output tensor is allocated internally, returns the
///   allocated vector and tensor.
/// - If the output tensor is provided, returns `None`.
///
/// If you activated cargo feature `ndarray`, you can convert the output to [ndarray::ArrayD] by
///
/// ```rust
/// use tblis::prelude::*;
/// # use ndarray::prelude::*;
/// # let vec_g = vec![0.0f64; 2*2*2*2]; // pre-allocated data
/// # let tsr_g = TblisTensor::new(vec_g.as_ptr() as *mut f64, &[2,2,2,2], &[4,2,1,1]);
/// // with predefined
/// // - `vec_g`: `Vec<T>`
/// // - `tsr_g`: `TblisTensor<T>`
/// let out_g: ArrayD<_> = (vec_g, tsr_g).into_array(); // ndarray::ArrayD<T>
/// ```
///
/// # Panics
///
/// - This function will panic if failed. Use [`tblis_einsum_f`] for fallible version.
/// - TBLIS cannot handle a single contraction step involving more than two tensors.
/// - [opt_einsum_path::contract_path] may fail if the subscripts and tensor shapes that user
///   provides are invalid, or give a contraction step involving more than two tensors in strict
///   memory limit.
/// - This function allows non-ASCII characters in einsum subscripts, but TBLIS may panic if too
///   many characters are used. It is recommended to use no more than 52 characters in total. It is
///   not allowed to use more than 128 characters.
///
/// # Example
///
/// The following example is to perform contraction:
/// $$
/// G_{pqrs} = \sum_{\mu \nu \kappa \lambda} C_{\mu p} C_{\nu q} E_{\mu \nu \kappa \lambda}
/// C_{\kappa r} C_{\lambda s} $$
/// This tensor contraction is utilized in electronic structure (electronic integral in atomic
/// orbital basis $E_{\mu \nu \kappa \lambda}$ to molecular orbital basis $G_{pqrs}$).
///
/// To run the following code, you may need to
/// - activate crate feature `ndarray` to make conversion between [`ndarray::Array`],
///   [`ndarray::ArrayView`], [`ndarray::ArrayViewMut`] and [`TblisTensor`];
/// - properly link libtblis.so in your project (also see crate [tblis_ffi] and [tblis_src](https://docs.rs/tblis-src)
///   for more information).
///
/// The following code snippet performs this contraction.
///
/// ```rust
/// // Must declare crate `tblis-src` if you want link tblis dynamically.
/// // You can also call the following code in `build.rs`, instead of using crate `tblis-src`:
/// //     println!("cargo:rustc-link-lib=tblis");
/// extern crate tblis_src;
///
/// use ndarray::prelude::*;
/// use tblis::prelude::*;
///
/// // dummy setting of matrix C and tensor E
/// let (nao, nmo): (usize, usize) = (3, 2);
/// let vec_c: Vec<f64> = (0..nao * nmo).map(|x| x as f64).collect();
/// let vec_e: Vec<f64> = (0..nao * nao * nao * nao).map(|x| x as f64).collect();
///
/// let arr_c = ArrayView2::from_shape((nao, nmo), &vec_c).unwrap();
/// let arr_e = ArrayView4::from_shape((nao, nao, nao, nao), &vec_e).unwrap();
///
/// /// # Parameters
/// /// - `arr_c`: coefficient matrix $C_{\mu p}$
/// /// - `arr_s`: electronic integral $E_{\mu \nu \kappa \lambda}$ (in atomic orbital basis)
/// ///
/// /// # Returns
/// /// - `arr_g`: electronic integral $G_{pqrs}$ (in molecular orbital basis)
/// fn ao2mo(arr_c: ArrayView2<f64>, arr_e: ArrayView4<f64>) -> Array4<f64> {
///     // transform ndarray objects to tblis objects
///     let tsr_c = unsafe { arr_c.to_tblis_tensor() };
///     let tsr_e = unsafe { arr_e.to_tblis_tensor() };
///
///     // generate operands and perform contraction
///     let operands = [&tsr_c, &tsr_c, &tsr_e, &tsr_c, &tsr_c];
///     let out_g = unsafe {
///         tblis_einsum(
///             "μi,νa,μνκλ,κj,λb->iajb", // einsum subscripts
///             &operands,                // tensors to be contracted
///             "optimal",                // contraction strategy (see crate opt-einsum-path)
///             None,                     // memory limit (None means no limit, see crate opt-einsum-path)
///             true,                     // row-major (true) or col-major (false)
///             None,                     // pre-allocated output tensor (None to allocate internally)
///         )
///     };
///     let (vec_g, tsr_g) = out_g.unwrap(); // (underlying data, tensor shape/stride info)
///
///     // transform tblis object back to ndarray object
///     let arr_g = (vec_g, tsr_g).into_array().into_dimensionality().unwrap();
///     arr_g
/// }
///
/// let arr_g = ao2mo(arr_c, arr_e);
/// println!("{:?}", arr_g);
/// ```
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_einsum_f`] for fallible version.
/// - [`opt_einsum_path::contract_path`] for details of contraction path optimization.
pub unsafe fn tblis_einsum<T>(
    subscripts: &str,
    operands: &[&TblisTensor<T>],
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
    row_major: bool,
    out_tblis_tensor: Option<&mut TblisTensor<T>>,
) -> Option<(Vec<T>, TblisTensor<T>)>
where
    T: TblisFloatAPI,
{
    unsafe { tblis_einsum_f(subscripts, operands, optimize, memory_limit, row_major, out_tblis_tensor).unwrap() }
}

/// Perform einsum operation using TBLIS.
///
/// # Safety
///
/// - This function does not check tensor data validity and mutability.
///
/// # See also
///
/// - [`tblis_einsum`] for non-fallible version.
#[allow(clippy::type_complexity)]
pub unsafe fn tblis_einsum_f<T>(
    subscripts: &str,
    operands: &[&TblisTensor<T>],
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
    row_major: bool,
    out_tblis_tensor: Option<&mut TblisTensor<T>>,
) -> Result<Option<(Vec<T>, TblisTensor<T>)>, String>
where
    T: TblisFloatAPI,
{
    let shapes: Vec<Vec<usize>> = operands.iter().map(|t| t.shape.iter().map(|&s| s as usize).collect()).collect();
    let steps = tblis_einsum_prep_f(subscripts, &shapes, optimize, memory_limit)?;
    let mut tensor_list: Vec<(TblisTensor<T>, Option<Vec<T>>)> = operands.iter().map(|&t| (t.clone(), None)).collect();
    let num_steps = steps.len();
    for (idx_step, step) in steps.iter().enumerate() {
        let TblisContractStep { indices, idx_a, idx_b, idx_c, shape_c } = step;

        if let Some(idx_b) = idx_b {
            // case of tensor mult
            let mut idx_a = idx_a.clone();
            let mut idx_b = idx_b.clone();

            // handle trace here
            // - tblis_tensor_mult is not able to handle combined contract with trace
            // - trace means something like "eca, ab -> e", where 'c' in first and 'b' in second are redundant
            //   indices, should be traced before contraction
            let idx_a_set = idx_a.chars().collect::<BTreeSet<char>>();
            let idx_b_set = idx_b.chars().collect::<BTreeSet<char>>();
            let idx_c_set = idx_c.chars().collect::<BTreeSet<char>>();
            let idx_a_only = &(&idx_a_set - &idx_b_set) - &idx_c_set;
            let idx_b_only = &(&idx_b_set - &idx_a_set) - &idx_c_set;
            if !idx_a_only.is_empty() {
                let tsr_a = &tensor_list[indices[0]].0;
                let idx_a_traced = idx_a.chars().filter(|c| !idx_a_only.contains(c)).collect::<String>();
                let (vec_a_traced, tsr_a_traced) = tblis_trace_f(&idx_a, &idx_a_traced, tsr_a, row_major)?;
                tensor_list[indices[0]] = (tsr_a_traced, Some(vec_a_traced));
                idx_a = idx_a_traced;
            }
            if !idx_b_only.is_empty() {
                let tsr_b = &tensor_list[indices[1]].0;
                let idx_b_traced = idx_b.chars().filter(|c| !idx_b_only.contains(c)).collect::<String>();
                let (vec_b_traced, tsr_b_traced) = tblis_trace_f(&idx_b, &idx_b_traced, tsr_b, row_major)?;
                tensor_list[indices[1]] = (tsr_b_traced, Some(vec_b_traced));
                idx_b = idx_b_traced;
            }

            let tsr_a = &tensor_list[indices[0]].0;
            let tsr_b = &tensor_list[indices[1]].0;
            let is_last_step = idx_step == num_steps - 1;
            let (vec_c, mut tsr_c) = if is_last_step && let Some(ref out_tblis_tensor) = out_tblis_tensor {
                // final tensor with pre-allocated space
                let tsr_c = out_tblis_tensor;
                if tsr_c.shape != *shape_c {
                    return Err("Output tensor shape mismatch.".to_string());
                }
                (None, (*tsr_c).clone())
            } else {
                // intermediate tensor or final tensor without pre-allocated space
                let size_c = shape_c.iter().product::<isize>() as usize;
                let vec_c = unsafe { crate::alloc_vec::uninitialized_vec::<T>(size_c)? };
                let stride_c = shape_to_stride(shape_c, row_major);
                let tsr_c = TblisTensor::new(vec_c.as_ptr() as *mut T, shape_c, &stride_c);
                (Some(vec_c), tsr_c)
            };
            // handle empty idx_a/idx_b (scalar-like operations)
            match (idx_a.is_empty(), idx_b.is_empty()) {
                (false, false) => unsafe {
                    tblis_tensor_mult(tsr_a, &idx_a, tsr_b, &idx_b, &mut tsr_c, idx_c, None);
                },
                (true, true) => {
                    let val_a = tsr_a.clone().set_scalar(T::one()).to_scalar()?;
                    let val_b = tsr_b.clone().set_scalar(T::one()).to_scalar()?;
                    unsafe { *tsr_c.data = val_a * val_b };
                },
                (true, false) => {
                    let val_a = tsr_a.clone().set_scalar(T::one()).to_scalar()?;
                    let add_cfg = TblisAddCfgBuilder::default().alpha(val_a).beta(T::zero()).build().unwrap();
                    unsafe { tblis_tensor_add(tsr_b, &idx_b, &mut tsr_c, idx_c, Some(add_cfg)) };
                },
                (false, true) => {
                    let val_b = tsr_b.clone().set_scalar(T::one()).to_scalar()?;
                    let add_cfg = TblisAddCfgBuilder::default().alpha(val_b).beta(T::zero()).build().unwrap();
                    unsafe { tblis_tensor_add(tsr_a, &idx_a, &mut tsr_c, idx_c, Some(add_cfg)) };
                },
            };
            tensor_list.push((tsr_c, vec_c));
        } else {
            // case of tensor transpose (implement by add)
            let tsr_a = &tensor_list[indices[0]].0;
            let size_c = shape_c.iter().product::<isize>() as usize;
            let vec_c = unsafe { crate::alloc_vec::uninitialized_vec::<T>(size_c)? };
            let stride_c = shape_to_stride(shape_c, row_major);
            let mut tsr_c = TblisTensor::new(vec_c.as_ptr() as *mut T, shape_c, &stride_c);
            let cfg = TblisAddCfgBuilder::default().beta(T::zero()).build().unwrap();
            unsafe { tblis_tensor_add(tsr_a, idx_a, &mut tsr_c, idx_c, Some(cfg)) };
            tensor_list.push((tsr_c, Some(vec_c)));
        }
        // remove used tensors
        let mut indices = indices.to_vec();
        indices.sort_unstable_by(|a, b| b.cmp(a));
        for i in indices {
            tensor_list.remove(i);
        }
    }
    assert!(tensor_list.len() == 1);
    let (tsr, vec_opt) = tensor_list.pop().unwrap();
    if let Some(vec) = vec_opt { Ok(Some((vec, tsr))) } else { Err("Final tensor does not own its data.".to_string()) }
}

#[cfg(feature = "ndarray")]
pub mod ndarray_einsum {
    use crate::prelude::*;
    use duplicate::duplicate_item;
    use ndarray::prelude::*;

    #[duplicate_item(ArrayType; [ArrayView<'_, T, D>]; [ArrayViewMut<'_, T, D>]; [Array<T, D>];)]
    impl<T, D> ToTblisTensor<T> for ArrayType
    where
        T: TblisFloatAPI,
        D: Dimension,
    {
        fn to_tblis_tensor(&self) -> TblisTensor<T> {
            let shape = self.shape().iter().map(|&s| s as isize).collect::<Vec<isize>>();
            let stride = self.strides().to_vec();
            let data_ptr = self.as_ptr() as *mut T;
            TblisTensor::new(data_ptr, &shape, &stride)
        }
    }

    /// Trait to convert `(Vec<T>, TblisTensor<T>)` to [`ndarray::ArrayD`].
    pub trait ArrayFromTblisTensor {
        type Out;
        fn into_array(self) -> Self::Out;
    }

    impl<T> ArrayFromTblisTensor for (Vec<T>, TblisTensor<T>)
    where
        T: TblisFloatAPI,
    {
        type Out = ArrayD<T>;
        fn into_array(self) -> ArrayD<T> {
            let (vec, tsr) = self;
            let shape = tsr.shape.iter().map(|&s| s as usize).collect::<Vec<usize>>();
            ArrayD::from_shape_vec(IxDyn(&shape), vec).unwrap()
        }
    }

    /// Convert `(Vec<T>, TblisTensor<T>)` to [`ndarray::ArrayD`].
    pub fn array_from_tblis_tensor<T>(dat: (Vec<T>, TblisTensor<T>)) -> ArrayD<T>
    where
        T: TblisFloatAPI,
    {
        dat.into_array()
    }
}

#[cfg(feature = "ndarray")]
pub use ndarray_einsum::*;

#[cfg(test)]
#[cfg(feature = "ndarray")]
mod test_ndarray_workable {
    #[test]
    #[allow(clippy::let_and_return)]
    fn test_ndarray_workable() {
        use crate::prelude::*;
        use ndarray::prelude::*;
        let (nao, nmo): (usize, usize) = (3, 2);
        let vec_c: Vec<f64> = (0..nao * nmo).map(|x| x as f64).collect();
        let vec_e: Vec<f64> = (0..nao * nao * nao * nao).map(|x| x as f64).collect();

        let arr_c = ArrayView2::from_shape((nao, nmo), &vec_c).unwrap();
        let arr_e = ArrayView4::from_shape((nao, nao, nao, nao), &vec_e).unwrap();

        /// # Parameters
        /// - `arr_c`: coefficient matrix $C_{\mu p}$
        /// - `arr_s`: electronic integral $E_{\mu \nu \kappa \lambda}$ (in atomic orbital basis)
        ///
        /// # Returns
        /// - `arr_g`: electronic integral $G_{pqrs}$ (in molecular orbital basis)
        fn ao2mo(arr_c: ArrayView2<f64>, arr_e: ArrayView4<f64>) -> Array4<f64> {
            // transform ndarray objects to tblis objects
            let tsr_c = arr_c.to_tblis_tensor();
            let tsr_e = arr_e.to_tblis_tensor();

            // generate operands and perform contraction
            let operands = [&tsr_c, &tsr_c, &tsr_e, &tsr_c, &tsr_c];
            let out_g = unsafe {
                tblis_einsum(
                    "μi,νa,μνκλ,κj,λb->iajb", // einsum subscripts
                    &operands,                // tensors to be contracted
                    "optimal",                // contraction strategy (see crate opt-einsum-path)
                    None,                     // memory limit (None means no limit, see crate opt-einsum-path)
                    true,                     // row-major (true) or col-major (false)
                    None,                     // pre-allocated output tensor (None to allocate internally)
                )
            };
            let (vec_g, tsr_g) = out_g.unwrap(); // (underlying data, tensor shape/stride info)

            // transform tblis object back to ndarray object
            let arr_g = (vec_g, tsr_g).into_array().into_dimensionality().unwrap();
            arr_g
        }

        let arr_g = ao2mo(arr_c, arr_e);
        println!("{:?}", arr_g);
    }
}
