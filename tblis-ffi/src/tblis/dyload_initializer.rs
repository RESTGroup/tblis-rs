//! Library initializer implementation for dynamic-loading.
//!
//! This file is generated automatically.

use super::*;
use libloading::{Library, Symbol};

unsafe fn get_symbol<'f, F>(libs: &'f [Library], name: &[u8]) -> Option<Symbol<'f, F>> {
    libs.iter().find_map(|lib| lib.get::<F>(name).ok())
}

impl DyLoadLib {
    pub unsafe fn new(libs: Vec<libloading::Library>, libs_path: Vec<String>) -> DyLoadLib {
        let mut result = DyLoadLib {
            __libraries: vec![],      // dummy here, set this field later
            __libraries_path: vec![], // dummy here, set this field later
            tblis_init_scalar_s: get_symbol(&libs, b"tblis_init_scalar_s\0").map(|sym| *sym),
            tblis_init_scalar_d: get_symbol(&libs, b"tblis_init_scalar_d\0").map(|sym| *sym),
            tblis_init_scalar_c: get_symbol(&libs, b"tblis_init_scalar_c\0").map(|sym| *sym),
            tblis_init_scalar_z: get_symbol(&libs, b"tblis_init_scalar_z\0").map(|sym| *sym),
            tblis_init_tensor_scaled_s: get_symbol(&libs, b"tblis_init_tensor_scaled_s\0").map(|sym| *sym),
            tblis_init_tensor_scaled_d: get_symbol(&libs, b"tblis_init_tensor_scaled_d\0").map(|sym| *sym),
            tblis_init_tensor_scaled_c: get_symbol(&libs, b"tblis_init_tensor_scaled_c\0").map(|sym| *sym),
            tblis_init_tensor_scaled_z: get_symbol(&libs, b"tblis_init_tensor_scaled_z\0").map(|sym| *sym),
            tblis_init_tensor_s: get_symbol(&libs, b"tblis_init_tensor_s\0").map(|sym| *sym),
            tblis_init_tensor_d: get_symbol(&libs, b"tblis_init_tensor_d\0").map(|sym| *sym),
            tblis_init_tensor_c: get_symbol(&libs, b"tblis_init_tensor_c\0").map(|sym| *sym),
            tblis_init_tensor_z: get_symbol(&libs, b"tblis_init_tensor_z\0").map(|sym| *sym),
            tblis_get_num_threads: get_symbol(&libs, b"tblis_get_num_threads\0").map(|sym| *sym),
            tblis_set_num_threads: get_symbol(&libs, b"tblis_set_num_threads\0").map(|sym| *sym),
            tblis_tensor_add: get_symbol(&libs, b"tblis_tensor_add\0").map(|sym| *sym),
            tblis_tensor_dot: get_symbol(&libs, b"tblis_tensor_dot\0").map(|sym| *sym),
            tblis_tensor_reduce: get_symbol(&libs, b"tblis_tensor_reduce\0").map(|sym| *sym),
            tblis_tensor_scale: get_symbol(&libs, b"tblis_tensor_scale\0").map(|sym| *sym),
            tblis_tensor_set: get_symbol(&libs, b"tblis_tensor_set\0").map(|sym| *sym),
            tblis_tensor_shift: get_symbol(&libs, b"tblis_tensor_shift\0").map(|sym| *sym),
            tblis_tensor_mult: get_symbol(&libs, b"tblis_tensor_mult\0").map(|sym| *sym),
        };
        result.__libraries = libs;
        result.__libraries_path = libs_path;
        result
    }
}
