"""
Python script to handle dynamic loading

This is a utility python package that may be used across FFI bindings, for RSTSR-FFI project only.

This package handles
- splitting `unsafe extern` block out from FFI file;
- creating library struct for dynamic loading;
- creating FFI-compatible functions for dynamic loading.
"""

from tree_sitter import Language, Parser
import tree_sitter_rust


def dyload_parse_file(token):
    parser = Parser(Language(tree_sitter_rust.language()))
    token_transformed = token.replace("unsafe extern \"C\"", "extern \"C\"")
    parsed = parser.parse(bytes(token_transformed, "utf8"))
    parsed_ffi = []
    for node in parsed.root_node.children:
        if node.type == "foreign_mod_item":
            parsed_ffi.append(node)
    assert(len(parsed_ffi) == 1)
    return parsed, parsed_ffi[0]


def dyload_remove_extern(parsed, node_extern):
    return parsed.root_node.text.decode("utf8").replace(node_extern.text.decode("utf8"), "")


def dyload_get_ffi_fn(node):
    assert(node.type == "foreign_mod_item")
    return [node for node in node.children[-1].children if node.type == "function_signature_item"]


def dyload_fn_split(node):
    assert(node.type == "function_signature_item")
    keys = ["visibility_modifier", "identifier", "parameters", "return_type"]
    result = { key: None for key in keys }
    for (idx, child) in enumerate(node.children):
        if child.type == "->":
            result["return_type"] = node.children[idx + 1]
        elif child.type in keys:
            result[child.type] = child
    assert(result["identifier"] is not None)
    assert(result["parameters"] is not None)
    return result


def dyload_main(token, token_extra=None):
    # 1. obtain all stuffs for usual ffi use cases
    parsed, node_extern = dyload_parse_file(token)
    token_ffi_base = dyload_remove_extern(parsed, node_extern)

    # 2. prepare necessary tokens for output
    token_ffi_extern = ""
    token_dyload_struct = ""
    token_dyload_initializer = ""
    token_dyload_compatible = ""
    
    nodes_fn = dyload_get_ffi_fn(node_extern)
    nodes_fn_extra = None
    if token_extra is not None:
        parser = Parser(Language(tree_sitter_rust.language()))
        parsed_extra = parser.parse(bytes(token_extra, "utf8"))
        nodes_fn_extra = dyload_get_ffi_fn(parsed_extra.root_node.children[0])
        nodes_fn = nodes_fn + nodes_fn_extra
    identifiers_fn = []

    # 3. iterate by functions
    for node_fn in nodes_fn:
        dict_fn = dyload_fn_split(node_fn)

        visibility_modifier = dict_fn["visibility_modifier"].text.decode("utf8")
        identifier = dict_fn["identifier"].text.decode("utf8")
        
        return_type_string = ""
        if dict_fn["return_type"] is not None:
            return_type_string = " -> " + dict_fn["return_type"].text.decode("utf8")

        nodes_para = [n for n in dict_fn["parameters"].children if n.type == "parameter"]
        parameters = "(" + ", ".join([n.text.decode("utf8") for n in nodes_para]) + ")"
        parameters_called = ", ".join([n.children[0].text.decode("utf8") for n in nodes_para])

        part_dyload_struct = f"""
            {visibility_modifier} {identifier}: Option<unsafe extern "C" fn{parameters}{return_type_string}>,
        """.strip()
        part_dyload_initializer = f"""
            {identifier}: get_symbol(&libs, b"{identifier}\\0").map(|sym| *sym),
        """.strip()
        part_dyload_compatible = f"""
            {visibility_modifier} unsafe fn {identifier}{parameters}{return_type_string} {{
                dyload_lib().{identifier}.unwrap()({parameters_called})
            }}
        """.strip()

        token_dyload_struct += part_dyload_struct + "\n"
        token_dyload_initializer += part_dyload_initializer + "\n"
        token_dyload_compatible += part_dyload_compatible + "\n\n"
        identifiers_fn.append(identifier)

    # 4. finish all other stuffs
    
    # ffi_base.rs
    output_ffi_base = f"""
//! Base of current FFI.
//!
//! Declaration of types, enums, cargo feature controls, etc.
//!
//! This file is generated automatically.

{token_ffi_base}
    """

    output_ffi_extern = f"""
//! FFI function declarations for non-dynamic-loading.
//!
//! This file is generated automatically.

use super::*;

unsafe {node_extern.text.decode("utf8")}
    """
    if token_extra is not None:
        output_ffi_extern += f"unsafe {token_extra.strip()}"

    output_dyload_struct = f"""
//! Library struct definition for dynamic-loading.
//!
//! This file is generated automatically.

use super::*;

pub struct DyLoadLib {{
    pub __libraries: Vec<libloading::Library>,
    pub __libraries_path: Vec<String>,
    {token_dyload_struct}
}}
    """

    output_dyload_initializer = f"""
//! Library initializer implementation for dynamic-loading.
//!
//! This file is generated automatically.

use super::*;
use libloading::{{Library, Symbol}};

unsafe fn get_symbol<'f, F>(libs: &'f [Library], name: &[u8]) -> Option<Symbol<'f, F>> {{
    libs.iter().find_map(|lib| lib.get::<F>(name).ok())
}}

impl DyLoadLib {{
    pub unsafe fn new(libs: Vec<libloading::Library>, libs_path: Vec<String>) -> DyLoadLib {{
        let mut result = DyLoadLib {{
            __libraries: vec![], // dummy here, set this field later
            __libraries_path: vec![], // dummy here, set this field later
            {token_dyload_initializer}
        }};
        result.__libraries = libs;
        result.__libraries_path = libs_path;
        result
    }}
}}
    """

    output_dyload_compatible = f"""
//! Compatible implementation for dynamic-loading.
//!
//! This requires custom `dyload_lib` definition in mod.rs, or visible from current layer of module.
//!
//! This file is generated automatically.

use super::*;

{token_dyload_compatible}
    """

    return {
        "ffi_base": output_ffi_base,
        "ffi_extern": output_ffi_extern,
        "dyload_struct": output_dyload_struct,
        "dyload_initializer": output_dyload_initializer,
        "dyload_compatible": output_dyload_compatible,
        "mod_template": DYLOAD_MOD_TEMPLATE,
    }


DYLOAD_MOD_TEMPLATE = """/* Tips for developers:
   You may substitute the `SUBSTITUTION` for your specific library.
*/

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

pub const MOD_NAME: &str = module_path!();
pub const LIB_NAME: &str = SUBSTITUTION; // for code, e.g. "MKL"
pub const LIB_NAME_SHOW: &str = SUBSTITUTION; // for display, e.g. "oneMKL"
pub const LIB_NAME_LINK: &str = SUBSTITUTION; // for linking, e.g. "mkl_rt"

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

        candidates.extend(vec![
            format!("{DLL_PREFIX}{LIB_NAME_LINK}{DLL_SUFFIX}"),
            SUBSTITUTION, // more candidates can be added here
        ]);
        candidates
    }

    fn check_lib_loaded(lib: &DyLoadLib) -> bool {
        SUBSTITUTION // usually check some function is not `None`
                     // e.g. lib.cblas_dgemm.is_some()
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
Condition: {SUBSTITUTION}
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
                    Err(e) => err_msg.push_str(&format!("Failed to load `{candidate}`: {e}\n")),
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
"""
