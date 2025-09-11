# TBLIS FFI bindings

This crate contains TBLIS FFI bindings.

Current FFI version is [TBLIS 9b95712](https://github.com/MatthewsResearchGroup/tblis/commit/9b95712966cb8804be51c62bfd6207957f62bc6f) after v1.3.0. If you are using an older version of TBLIS, this crate should still work if you do not explicitly call the function that only occurs in higher version of TBLIS.

TBLIS (C++/C/ASM) source code is available on [github](https://github.com/MatthewsResearchGroup/tblis).

This crate is not official bindgen project. It is originally intended to serve rust tensor toolkit [RSTSR](https://github.com/RESTGroup/rstsr) and rust electronic structure toolkit [REST](https://gitee.com/RESTGroup/rest).

- **Audience**: Anyone uses TBLIS function may also find it useful, not only RSTSR or REST program developers.
- **Pure Extern or Dynamic Loading**: This crate supports either pure extern (usual FFI, requires dynamic or static linking) and dynamic-loading, by cargo feature `dynamic_loading`.

## Dynamic loading

This crate supports dynamic loading.

If you want to use dynamic loading, please enable cargo feature `dynamic_loading` when cargo build.

The dynamic loading will try to find proper library when your program initializes.
- This crate will automatically detect proper libraries, if these libraries are in environmental path `LD_LIBRARY_PATH` (Linux) `DYLD_LIBRARY_PATH` (Mac OS), `PATH` (Windows).
- If you want to override the library to be loaded, please set these shell environmental variable `RSTSR_DYLOAD_TBLIS` to the dynamic library path.

## Cargo features

Optional features:

- `dynamic_loading`: Supports dynamic loading.

## Crate structure

- `ffi_base.rs`: Basic type, enum, struct definitions.
- `ffi_extern.rs`: Unsafe extern "C" bindgen functions. Only activated when not dynamic loading.
- `dyload_struct.rs`: Struct `Lib` for dynamic loading.
- `dyload_initializer.rs`: The initialization function of `Lib` for dynamic loading.
- `dyload_compatible.rs`: Unsafe bindgen function that is compatible to that of `ffi_extern.rs`. Only activated when dynamic loading.
