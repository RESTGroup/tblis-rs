//! Compatible implementation for dynamic-loading.
//!
//! This requires custom `dyload_lib` definition in mod.rs, or visible from current layer of module.
//!
//! This file is generated automatically.

use super::*;

pub unsafe fn tblis_init_scalar_s(s: *mut tblis_scalar, value: f32) {
    dyload_lib().tblis_init_scalar_s.unwrap()(s, value)
}

pub unsafe fn tblis_init_scalar_d(s: *mut tblis_scalar, value: f64) {
    dyload_lib().tblis_init_scalar_d.unwrap()(s, value)
}

pub unsafe fn tblis_init_scalar_c(s: *mut tblis_scalar, value: scomplex) {
    dyload_lib().tblis_init_scalar_c.unwrap()(s, value)
}

pub unsafe fn tblis_init_scalar_z(s: *mut tblis_scalar, value: dcomplex) {
    dyload_lib().tblis_init_scalar_z.unwrap()(s, value)
}

pub unsafe fn tblis_init_tensor_scaled_s(
    t: *mut tblis_tensor,
    scalar: f32,
    ndim: c_int,
    len: *mut len_type,
    data: *mut f32,
    stride: *mut stride_type,
) {
    dyload_lib().tblis_init_tensor_scaled_s.unwrap()(t, scalar, ndim, len, data, stride)
}

pub unsafe fn tblis_init_tensor_scaled_d(
    t: *mut tblis_tensor,
    scalar: f64,
    ndim: c_int,
    len: *mut len_type,
    data: *mut f64,
    stride: *mut stride_type,
) {
    dyload_lib().tblis_init_tensor_scaled_d.unwrap()(t, scalar, ndim, len, data, stride)
}

pub unsafe fn tblis_init_tensor_scaled_c(
    t: *mut tblis_tensor,
    scalar: scomplex,
    ndim: c_int,
    len: *mut len_type,
    data: *mut scomplex,
    stride: *mut stride_type,
) {
    dyload_lib().tblis_init_tensor_scaled_c.unwrap()(t, scalar, ndim, len, data, stride)
}

pub unsafe fn tblis_init_tensor_scaled_z(
    t: *mut tblis_tensor,
    scalar: dcomplex,
    ndim: c_int,
    len: *mut len_type,
    data: *mut dcomplex,
    stride: *mut stride_type,
) {
    dyload_lib().tblis_init_tensor_scaled_z.unwrap()(t, scalar, ndim, len, data, stride)
}

pub unsafe fn tblis_init_tensor_s(
    t: *mut tblis_tensor,
    ndim: c_int,
    len: *mut len_type,
    data: *mut f32,
    stride: *mut stride_type,
) {
    dyload_lib().tblis_init_tensor_s.unwrap()(t, ndim, len, data, stride)
}

pub unsafe fn tblis_init_tensor_d(
    t: *mut tblis_tensor,
    ndim: c_int,
    len: *mut len_type,
    data: *mut f64,
    stride: *mut stride_type,
) {
    dyload_lib().tblis_init_tensor_d.unwrap()(t, ndim, len, data, stride)
}

pub unsafe fn tblis_init_tensor_c(
    t: *mut tblis_tensor,
    ndim: c_int,
    len: *mut len_type,
    data: *mut scomplex,
    stride: *mut stride_type,
) {
    dyload_lib().tblis_init_tensor_c.unwrap()(t, ndim, len, data, stride)
}

pub unsafe fn tblis_init_tensor_z(
    t: *mut tblis_tensor,
    ndim: c_int,
    len: *mut len_type,
    data: *mut dcomplex,
    stride: *mut stride_type,
) {
    dyload_lib().tblis_init_tensor_z.unwrap()(t, ndim, len, data, stride)
}

pub unsafe fn tblis_get_num_threads() -> c_uint {
    dyload_lib().tblis_get_num_threads.unwrap()()
}

pub unsafe fn tblis_set_num_threads(num_threads: c_uint) {
    dyload_lib().tblis_set_num_threads.unwrap()(num_threads)
}

pub unsafe fn tblis_tensor_add(
    comm: *const tblis_comm,
    cntx: *const tblis_config,
    A: *const tblis_tensor,
    idx_A: *const label_type,
    B: *mut tblis_tensor,
    idx_B: *const label_type,
) {
    dyload_lib().tblis_tensor_add.unwrap()(comm, cntx, A, idx_A, B, idx_B)
}

pub unsafe fn tblis_tensor_dot(
    comm: *const tblis_comm,
    cntx: *const tblis_config,
    A: *const tblis_tensor,
    idx_A: *const label_type,
    B: *const tblis_tensor,
    idx_B: *const label_type,
    result: *mut tblis_scalar,
) {
    dyload_lib().tblis_tensor_dot.unwrap()(comm, cntx, A, idx_A, B, idx_B, result)
}

pub unsafe fn tblis_tensor_reduce(
    comm: *const tblis_comm,
    cntx: *const tblis_config,
    op: reduce_t,
    A: *const tblis_tensor,
    idx_A: *const label_type,
    result: *mut tblis_scalar,
    idx: *mut len_type,
) {
    dyload_lib().tblis_tensor_reduce.unwrap()(comm, cntx, op, A, idx_A, result, idx)
}

pub unsafe fn tblis_tensor_scale(
    comm: *const tblis_comm,
    cntx: *const tblis_config,
    A: *mut tblis_tensor,
    idx_A: *const label_type,
) {
    dyload_lib().tblis_tensor_scale.unwrap()(comm, cntx, A, idx_A)
}

pub unsafe fn tblis_tensor_set(
    comm: *const tblis_comm,
    cntx: *const tblis_config,
    alpha: *const tblis_scalar,
    A: *mut tblis_tensor,
    idx_A: *const label_type,
) {
    dyload_lib().tblis_tensor_set.unwrap()(comm, cntx, alpha, A, idx_A)
}

pub unsafe fn tblis_tensor_shift(
    comm: *const tblis_comm,
    cntx: *const tblis_config,
    alpha: *const tblis_scalar,
    A: *mut tblis_tensor,
    idx_A: *const label_type,
) {
    dyload_lib().tblis_tensor_shift.unwrap()(comm, cntx, alpha, A, idx_A)
}

pub unsafe fn tblis_tensor_mult(
    comm: *const tblis_comm,
    cntx: *const tblis_config,
    A: *const tblis_tensor,
    idx_A: *const label_type,
    B: *const tblis_tensor,
    idx_B: *const label_type,
    C: *mut tblis_tensor,
    idx_C: *const label_type,
) {
    dyload_lib().tblis_tensor_mult.unwrap()(comm, cntx, A, idx_A, B, idx_B, C, idx_C)
}
