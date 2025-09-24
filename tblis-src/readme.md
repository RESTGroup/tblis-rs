# tblis-src: Crate for TBLIS Source Code

To use crate `tblis-ffi` in rust, you may need to perform some configuration to properly link `libtblis.so` into your own program.

## Link library `tblis` with cargo crate `tblis-src`

You can link library `tblis` by cargo crate `tblis-src`. You can use crago feature `build_from_source` to automatically build TBLIS with default configuration.

### Crate features

- **`build_from_source`**: This will use CMake (cmake > 3.23, c++20 standard), and use the code from git submodule to compile tblis. Though this option can be developer-friendly (you do not need to perform any other configurations to make program compile and run by cargo), `build_from_source` does not provide customized compilation.

    CMake configurable variables (can be defined as environment variables):
    - `TBLIS_SRC`: Git repository source directory or URL. All git submodules (marray, blis, tci) should be properly downloaded.
    - `TBLIS_VER`: Git repository version (branch or tag). Default to be `develop`.

- **`static`**: This will link static libary instead of dynamic one. Please note that static linking may require additional dynamic library linking, which should be configured manually by developer in `build.rs` or environment variables `RUSTFLAGS`. Static linking can be difficult when searching symbols, and we recommend dynamic linking in most cases.

## Link library `tblis` manually

By this way, you can directly use cargo crate `tblis-ffi`, without using `tblis-src`.

It is recommended to link `libtblis.so` by dynamic linking. Making sure your library is in environment variable `LD_LIBRARY_PATH`, then

```rust
// build.rs
println!("cargo:rustc-link-lib=tblis");
```
