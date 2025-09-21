//! FFI function declarations for non-dynamic-loading.
//!
//! This file is generated automatically.

use super::*;

unsafe extern "C" {
    pub fn tblis_init_scalar_s(s: *mut tblis_scalar, value: f32);
    pub fn tblis_init_scalar_d(s: *mut tblis_scalar, value: f64);
    pub fn tblis_init_scalar_c(s: *mut tblis_scalar, value: scomplex);
    pub fn tblis_init_scalar_z(s: *mut tblis_scalar, value: dcomplex);
    pub fn tblis_init_tensor_scaled_s(
        t: *mut tblis_tensor,
        scalar: f32,
        ndim: c_int,
        len: *mut len_type,
        data: *mut f32,
        stride: *mut stride_type,
    );
    pub fn tblis_init_tensor_scaled_d(
        t: *mut tblis_tensor,
        scalar: f64,
        ndim: c_int,
        len: *mut len_type,
        data: *mut f64,
        stride: *mut stride_type,
    );
    pub fn tblis_init_tensor_scaled_c(
        t: *mut tblis_tensor,
        scalar: scomplex,
        ndim: c_int,
        len: *mut len_type,
        data: *mut scomplex,
        stride: *mut stride_type,
    );
    pub fn tblis_init_tensor_scaled_z(
        t: *mut tblis_tensor,
        scalar: dcomplex,
        ndim: c_int,
        len: *mut len_type,
        data: *mut dcomplex,
        stride: *mut stride_type,
    );
    pub fn tblis_init_tensor_s(
        t: *mut tblis_tensor,
        ndim: c_int,
        len: *mut len_type,
        data: *mut f32,
        stride: *mut stride_type,
    );
    pub fn tblis_init_tensor_d(
        t: *mut tblis_tensor,
        ndim: c_int,
        len: *mut len_type,
        data: *mut f64,
        stride: *mut stride_type,
    );
    pub fn tblis_init_tensor_c(
        t: *mut tblis_tensor,
        ndim: c_int,
        len: *mut len_type,
        data: *mut scomplex,
        stride: *mut stride_type,
    );
    pub fn tblis_init_tensor_z(
        t: *mut tblis_tensor,
        ndim: c_int,
        len: *mut len_type,
        data: *mut dcomplex,
        stride: *mut stride_type,
    );
    pub static tblis_single: *const tblis_comm;
    pub fn tblis_get_num_threads() -> c_uint;
    pub fn tblis_set_num_threads(num_threads: c_uint);
    pub fn tblis_tensor_add(
        comm: *const tblis_comm,
        cntx: *const tblis_config,
        A: *const tblis_tensor,
        idx_A: *const label_type,
        B: *mut tblis_tensor,
        idx_B: *const label_type,
    );
    pub fn tblis_tensor_dot(
        comm: *const tblis_comm,
        cntx: *const tblis_config,
        A: *const tblis_tensor,
        idx_A: *const label_type,
        B: *const tblis_tensor,
        idx_B: *const label_type,
        result: *mut tblis_scalar,
    );
    pub fn tblis_tensor_reduce(
        comm: *const tblis_comm,
        cntx: *const tblis_config,
        op: reduce_t,
        A: *const tblis_tensor,
        idx_A: *const label_type,
        result: *mut tblis_scalar,
        idx: *mut len_type,
    );
    pub fn tblis_tensor_scale(
        comm: *const tblis_comm,
        cntx: *const tblis_config,
        A: *mut tblis_tensor,
        idx_A: *const label_type,
    );
    pub fn tblis_tensor_set(
        comm: *const tblis_comm,
        cntx: *const tblis_config,
        alpha: *const tblis_scalar,
        A: *mut tblis_tensor,
        idx_A: *const label_type,
    );
    pub fn tblis_tensor_shift(
        comm: *const tblis_comm,
        cntx: *const tblis_config,
        alpha: *const tblis_scalar,
        A: *mut tblis_tensor,
        idx_A: *const label_type,
    );
    pub fn tblis_tensor_mult(
        comm: *const tblis_comm,
        cntx: *const tblis_config,
        A: *const tblis_tensor,
        idx_A: *const label_type,
        B: *const tblis_tensor,
        idx_B: *const label_type,
        C: *mut tblis_tensor,
        idx_C: *const label_type,
    );
}
