# tblis-ffi: TBLIS FFI bindings

This crate contains TBLIS FFI bindings.

[TBLIS](https://github.com/MatthewsResearchGroup/tblis) (Tensor BLIS, The Tensor-Based Library Instantiation Software) can perform various tensor operations (multiplication, addition, reduction, transposition, etc.) efficiently on single-node CPU. This library can be an underlying driver for performing einsum (Einstein summation).

Current FFI version is [TBLIS 9b95712](https://github.com/MatthewsResearchGroup/tblis/commit/9b95712966cb8804be51c62bfd6207957f62bc6f) after v1.3.0. If you are using an older version of TBLIS, this crate should still work if you do not explicitly call the function that only occurs in higher version of TBLIS.

TBLIS (C++) source code is available on [github](https://github.com/MatthewsResearchGroup/tblis).

This crate is not official bindgen project. It is originally intended to serve rust tensor toolkit [RSTSR](https://github.com/RESTGroup/rstsr) and rust electronic structure toolkit [REST](https://gitee.com/RESTGroup/rest).

- **Audience**: Anyone uses TBLIS function may also find it useful, not only RSTSR or REST program developers.
- **Pure Extern or Dynamic Loading**: This crate supports either pure extern (usual FFI, requires dynamic or static linking) and dynamic-loading, by cargo feature `dynamic_loading`.

## Dynamic loading

By default, the `tblis` shared library is loaded at runtime via `libloading`. The loading process will search for the library in multiple locations, by the following order:

1. User-defined candidates via environment variables `TBLIS_DYLOAD` or `RSTSR_DYLOAD`.
2. `LD_LIBRARY_PATH` style discovery via environment variables `LD_LIBRARY_PATH` (Linux), `DYLD_LIBRARY_PATH` and `DYLD_FALLBACK_LIBRARY_PATH` (macOS), `PATH` (Windows).
3. Python interpreter path discovery: if Python is at `/path/bin/python`, the library is expected at `/path/lib/libtblis.so`. This includes `TBLIS_PYTHON_PATH`, `CONDA_PREFIX`, and Python interpreters found in `PATH`.
4. Standard system candidates such as `/usr/lib`, `/usr/local/lib`, and `/lib`.

To disable dynamic loading and use static/dynamic linking instead, disable the `dynamic_loading` cargo feature.

## Cargo features

Optional features:

- `dynamic_loading`: Supports dynamic loading.

## Crate structure

- `ffi_base.rs`: Basic type, enum, struct definitions.
- `ffi_extern.rs`: Unsafe extern "C" bindgen functions. Only activated when not dynamic loading.
- `dyload_struct.rs`: Struct `Lib` for dynamic loading.
- `dyload_initializer.rs`: The initialization function of `Lib` for dynamic loading.
- `dyload_compatible.rs`: Unsafe bindgen function that is compatible to that of `ffi_extern.rs`. Only activated when dynamic loading.
