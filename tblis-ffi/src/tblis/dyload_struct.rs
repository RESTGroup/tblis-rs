//! Library struct definition for dynamic-loading.
//!
//! This file is generated automatically.

use super::*;

pub struct DyLoadLib {
    pub __libraries: Vec<libloading::Library>,
    pub __libraries_path: Vec<String>,
    pub tblis_init_scalar_s: Option<unsafe extern "C" fn(s: *mut tblis_scalar, value: f32)>,
    pub tblis_init_scalar_d: Option<unsafe extern "C" fn(s: *mut tblis_scalar, value: f64)>,
    pub tblis_init_scalar_c: Option<unsafe extern "C" fn(s: *mut tblis_scalar, value: scomplex)>,
    pub tblis_init_scalar_z: Option<unsafe extern "C" fn(s: *mut tblis_scalar, value: dcomplex)>,
    pub tblis_init_tensor_scaled_s: Option<
        unsafe extern "C" fn(
            t: *mut tblis_tensor,
            scalar: f32,
            ndim: c_int,
            len: *mut len_type,
            data: *mut f32,
            stride: *mut stride_type,
        ),
    >,
    pub tblis_init_tensor_scaled_d: Option<
        unsafe extern "C" fn(
            t: *mut tblis_tensor,
            scalar: f64,
            ndim: c_int,
            len: *mut len_type,
            data: *mut f64,
            stride: *mut stride_type,
        ),
    >,
    pub tblis_init_tensor_scaled_c: Option<
        unsafe extern "C" fn(
            t: *mut tblis_tensor,
            scalar: scomplex,
            ndim: c_int,
            len: *mut len_type,
            data: *mut scomplex,
            stride: *mut stride_type,
        ),
    >,
    pub tblis_init_tensor_scaled_z: Option<
        unsafe extern "C" fn(
            t: *mut tblis_tensor,
            scalar: dcomplex,
            ndim: c_int,
            len: *mut len_type,
            data: *mut dcomplex,
            stride: *mut stride_type,
        ),
    >,
    pub tblis_init_tensor_s: Option<
        unsafe extern "C" fn(
            t: *mut tblis_tensor,
            ndim: c_int,
            len: *mut len_type,
            data: *mut f32,
            stride: *mut stride_type,
        ),
    >,
    pub tblis_init_tensor_d: Option<
        unsafe extern "C" fn(
            t: *mut tblis_tensor,
            ndim: c_int,
            len: *mut len_type,
            data: *mut f64,
            stride: *mut stride_type,
        ),
    >,
    pub tblis_init_tensor_c: Option<
        unsafe extern "C" fn(
            t: *mut tblis_tensor,
            ndim: c_int,
            len: *mut len_type,
            data: *mut scomplex,
            stride: *mut stride_type,
        ),
    >,
    pub tblis_init_tensor_z: Option<
        unsafe extern "C" fn(
            t: *mut tblis_tensor,
            ndim: c_int,
            len: *mut len_type,
            data: *mut dcomplex,
            stride: *mut stride_type,
        ),
    >,
    pub tblis_get_num_threads: Option<unsafe extern "C" fn() -> c_uint>,
    pub tblis_set_num_threads: Option<unsafe extern "C" fn(num_threads: c_uint)>,
    pub tblis_tensor_add: Option<
        unsafe extern "C" fn(
            comm: *const tblis_comm,
            cntx: *const tblis_config,
            A: *const tblis_tensor,
            idx_A: *const label_type,
            B: *mut tblis_tensor,
            idx_B: *const label_type,
        ),
    >,
    pub tblis_tensor_dot: Option<
        unsafe extern "C" fn(
            comm: *const tblis_comm,
            cntx: *const tblis_config,
            A: *const tblis_tensor,
            idx_A: *const label_type,
            B: *const tblis_tensor,
            idx_B: *const label_type,
            result: *mut tblis_scalar,
        ),
    >,
    pub tblis_tensor_reduce: Option<
        unsafe extern "C" fn(
            comm: *const tblis_comm,
            cntx: *const tblis_config,
            op: reduce_t,
            A: *const tblis_tensor,
            idx_A: *const label_type,
            result: *mut tblis_scalar,
            idx: *mut len_type,
        ),
    >,
    pub tblis_tensor_scale: Option<
        unsafe extern "C" fn(
            comm: *const tblis_comm,
            cntx: *const tblis_config,
            A: *mut tblis_tensor,
            idx_A: *const label_type,
        ),
    >,
    pub tblis_tensor_set: Option<
        unsafe extern "C" fn(
            comm: *const tblis_comm,
            cntx: *const tblis_config,
            alpha: *const tblis_scalar,
            A: *mut tblis_tensor,
            idx_A: *const label_type,
        ),
    >,
    pub tblis_tensor_shift: Option<
        unsafe extern "C" fn(
            comm: *const tblis_comm,
            cntx: *const tblis_config,
            alpha: *const tblis_scalar,
            A: *mut tblis_tensor,
            idx_A: *const label_type,
        ),
    >,
    pub tblis_tensor_mult: Option<
        unsafe extern "C" fn(
            comm: *const tblis_comm,
            cntx: *const tblis_config,
            A: *const tblis_tensor,
            idx_A: *const label_type,
            B: *const tblis_tensor,
            idx_B: *const label_type,
            C: *mut tblis_tensor,
            idx_C: *const label_type,
        ),
    >,
}
