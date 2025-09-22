//! Implementation of einsum operations preparations.
//!
//! For crate `ndarray`, also implements einsum execution.

use crate::prelude::*;
use opt_einsum_path::typing::{ContractionType, SizeLimitType, TensorShapeType};
use opt_einsum_path::{PathOptimizer, contract_path};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TblisContractStep {
    pub indices: Vec<usize>,
    pub idx_a: String,
    pub idx_b: Option<String>,
    pub idx_c: String,
    pub shape_c: Vec<isize>,
}

pub fn tblis_einsum_prep(
    subscripts: &str,
    operands: &[TensorShapeType],
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
) -> Vec<TblisContractStep> {
    tblis_einsum_prep_f(subscripts, operands, optimize, memory_limit).unwrap()
}

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
    crate::tensor_ops::tblis_tensor_add(tsr_prev, subscript_prev, &mut tsr_traced, &subscript_traced, Some(cfg));
    Ok((vec_traced, tsr_traced))
}

pub fn tblis_einsum<T>(
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
    tblis_einsum_f(subscripts, operands, optimize, memory_limit, row_major, out_tblis_tensor).unwrap()
}

#[allow(clippy::type_complexity)]
pub fn tblis_einsum_f<T>(
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
                (false, false) => {
                    crate::tensor_ops::tblis_tensor_mult(tsr_a, &idx_a, tsr_b, &idx_b, &mut tsr_c, idx_c, None);
                },
                (true, true) => {
                    let val_a = tsr_a.clone().set_scalar(T::one()).to_scalar()?;
                    let val_b = tsr_b.clone().set_scalar(T::one()).to_scalar()?;
                    unsafe { *tsr_c.data = val_a * val_b };
                },
                (true, false) => {
                    let val_a = tsr_a.clone().set_scalar(T::one()).to_scalar()?;
                    let add_cfg = TblisAddCfgBuilder::default().alpha(val_a).beta(T::zero()).build().unwrap();
                    crate::tensor_ops::tblis_tensor_add(tsr_b, &idx_b, &mut tsr_c, idx_c, Some(add_cfg));
                },
                (false, true) => {
                    let val_b = tsr_b.clone().set_scalar(T::one()).to_scalar()?;
                    let add_cfg = TblisAddCfgBuilder::default().alpha(val_b).beta(T::zero()).build().unwrap();
                    crate::tensor_ops::tblis_tensor_add(tsr_a, &idx_a, &mut tsr_c, idx_c, Some(add_cfg));
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
            crate::tensor_ops::tblis_tensor_add(tsr_a, idx_a, &mut tsr_c, idx_c, Some(cfg));
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
        unsafe fn to_tblis_tensor(&self) -> TblisTensor<T> {
            let shape = self.shape().iter().map(|&s| s as isize).collect::<Vec<isize>>();
            let stride = self.strides().to_vec();
            let data_ptr = self.as_ptr() as *mut T;
            TblisTensor::new(data_ptr, &shape, &stride)
        }
    }

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

    pub fn array_from_tblis_tensor<T>(dat: (Vec<T>, TblisTensor<T>)) -> ArrayD<T>
    where
        T: TblisFloatAPI,
    {
        dat.into_array()
    }
}

#[cfg(feature = "ndarray")]
pub use ndarray_einsum::*;

#[test]
#[cfg(feature = "ndarray")]
fn test_ndarray_workable() {
    use ndarray::prelude::*;
    let nao: usize = 3;
    let nocc: usize = 2;
    let vec_c: Vec<f64> = (0..nao * nocc).map(|x| x as f64).collect();
    let vec_eri: Vec<f64> = (0..nao * nao * nao * nao).map(|x| x as f64).collect();

    let arr_c = ArrayView2::from_shape((nao, nocc), &vec_c).unwrap();
    let arr_eri = ArrayView4::from_shape((nao, nao, nao, nao), &vec_eri).unwrap();
    let tsr_c = unsafe { arr_c.to_tblis_tensor() };
    let tsr_eri = unsafe { arr_eri.to_tblis_tensor() };

    let (vec_t2, tsr_t2) = tblis_einsum_f(
        "μi,νa,μνκλ,κj,λb->iajb",
        &[&tsr_c, &tsr_c, &tsr_eri, &tsr_c, &tsr_c],
        "optimal",
        None,
        true,
        None,
    )
    .unwrap()
    .unwrap();

    println!("{:?}", vec_t2);
    println!("{:?}", tsr_t2);

    let arr_t2 = tblis_einsum_f(
        "μi,νa,μνκλ,κj,λb->iajb",
        &[&tsr_c, &tsr_c, &tsr_eri, &tsr_c, &tsr_c],
        "optimal",
        None,
        false,
        None,
    )
    .unwrap()
    .unwrap()
    .into_array();

    println!("{:?}", arr_t2);
}
