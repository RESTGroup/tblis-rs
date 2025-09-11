#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

pub(crate) use core::ffi::*;

pub const MOD_NAME: &str = module_path!();
pub const LIB_NAME: &str = "TBLIS"; // for code, e.g. "MKL"
pub const LIB_NAME_SHOW: &str = "TBLIS"; // for display, e.g. "oneMKL"
pub const LIB_NAME_LINK: &str = "tblis"; // for linking, e.g. "mkl_rt"

#[cfg(feature = "dynamic_loading")]
mod dynamic_loading_specific {
    use super::*;
    use libloading::Library;
    use std::fmt::Debug;
    use std::sync::OnceLock;

    fn get_lib_candidates() -> Vec<String> {
        use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
        let mut candidates = vec![];

        // user defined candidates
        for paths in [format!("RSTSR_DYLOAD_{LIB_NAME}").as_str(), "RSTSR_DYLOAD"] {
            if let Ok(path) = std::env::var(paths) {
                candidates.extend(path.split(":").map(|s| s.to_string()).collect::<Vec<_>>());
            }
        }

        candidates.extend(vec![format!("{DLL_PREFIX}{LIB_NAME_LINK}{DLL_SUFFIX}")]);
        candidates
    }

    fn check_lib_loaded(lib: &DyLoadLib) -> bool {
        lib.tblis_tensor_mult.is_some()
    }

    fn panic_no_lib_found<S: Debug>(candidates: &[S], err_msg: &str) -> ! {
        panic!(
            r#"
This happens in module `{MOD_NAME}`.
Unable to dynamically load the {LIB_NAME_SHOW} (`{LIB_NAME_LINK}`) shared library.
Candidates: {candidates:#?}

Please check
- if dynamic-loading is not desired, please disable the `dynamic_loading` feature in your `Cargo.toml` (by something like --no-default-features).
- if you want to provide custom {LIB_NAME_SHOW} library, use environment variable `RSTSR_DYLOAD_{LIB_NAME}` or `RSTSR_DYLOAD` to specify the path to the library.
- if `lib{LIB_NAME_LINK}.so` (linux) or `lib{LIB_NAME_LINK}.dylib` (macOS) or `lib{LIB_NAME_LINK}.dll` (Windows) is installed on your system.
- if `LD_LIBRARY_PATH` (linux) or `DYLD_LIBRARY_PATH` (macOS) or `PATH` (Windows) environment variable is set correctly (any path that's visible to linker).
- this crate does not use things like `LD_PRELOAD` or `DYLD_INSERT_LIBRARIES` to load the library.
- this crate does not support static linking of libraries when dynamic-loading.

Error message(s):
{err_msg}
"#
        )
    }

    fn panic_condition_not_met<S: Debug>(candidates: &[S]) -> ! {
        panic!(
            r#"
This happens in module `{MOD_NAME}`.
Unable to dynamically load the {LIB_NAME_SHOW} (`{LIB_NAME_LINK}`) shared library, due to condition unfulfilled.
Condition: `tblis_tensor_mult` must be available.
Found libraries: {candidates:#?}

Please check
- if dynamic-loading is not desired, please disable the `dynamic_loading` feature in your `Cargo.toml` (by something like --no-default-features).
- if you want to provide custom {LIB_NAME_SHOW} library, use environment variable `RSTSR_DYLOAD_{LIB_NAME}` or `RSTSR_DYLOAD` to specify the path to the library.
- sequence of libraries matters: `RSTSR_DYLOAD_{LIB_NAME}` will be tried first, then `RSTSR_DYLOAD`, then system dynamic library search paths.
"#
        )
    }

    pub unsafe fn dyload_lib() -> &'static DyLoadLib {
        static LIB: OnceLock<DyLoadLib> = OnceLock::new();

        LIB.get_or_init(|| {
            let candidates = get_lib_candidates();
            let (mut libraries, mut libraries_path) = (vec![], vec![]);
            let mut err_msg = String::new();
            for candidate in &candidates {
                match Library::new(candidate) {
                    Ok(l) => {
                        libraries.push(l);
                        libraries_path.push(candidate.to_string());
                    },
                    Err(e) => err_msg.push_str(&format!(
                        "Failed to load `{candidate}`: {e}
"
                    )),
                }
            }
            let lib = DyLoadLib::new(libraries, libraries_path);
            if lib.__libraries.is_empty() {
                panic_no_lib_found(&candidates, &err_msg);
            }
            if !check_lib_loaded(&lib) {
                panic_condition_not_met(&lib.__libraries_path);
            }
            lib
        })
    }
}

#[cfg(feature = "dynamic_loading")]
pub use dynamic_loading_specific::*;

/* #region general configuration */

pub(crate) mod ffi_base;
pub use ffi_base::*;

#[cfg(not(feature = "dynamic_loading"))]
pub(crate) mod ffi_extern;
#[cfg(not(feature = "dynamic_loading"))]
pub use ffi_extern::*;

#[cfg(feature = "dynamic_loading")]
pub(crate) mod dyload_compatible;
#[cfg(feature = "dynamic_loading")]
pub(crate) mod dyload_initializer;
#[cfg(feature = "dynamic_loading")]
pub(crate) mod dyload_struct;

#[cfg(feature = "dynamic_loading")]
pub use dyload_compatible::*;
#[cfg(feature = "dynamic_loading")]
pub use dyload_struct::*;

/* #endregion */
