# TBLIS Wrapper in Rust

This crate contains TBLIS wrapper and several minimal implementations.

[TBLIS](https://github.com/MatthewsResearchGroup/tblis) (Tensor BLIS, The Tensor-Based Library Instantiation Software) can perform various tensor operations (multiplication, addition, reduction, transposition, etc.) efficiently on single-node CPU. This library can be an underlying driver for performing einsum (Einstein summation).

TBLIS (C++) source code is available on [github](https://github.com/MatthewsResearchGroup/tblis).

This crate is not official wrapper project. It is originally intended to serve rust tensor toolkit [RSTSR](https://github.com/RESTGroup/rstsr) and rust electronic structure toolkit [REST](https://gitee.com/RESTGroup/rest).

<center>

| Resources | Badges | API Document |
|--|--|--|
| Crate for Wrapper (tblis) | [![Crate](https://img.shields.io/crates/v/tblis.svg)](https://crates.io/crates/tblis) | [![API Documentation](https://docs.rs/tblis/badge.svg)](https://docs.rs/tblis) |
| Crate for FFI (tblis-ffi) | [![Crate](https://img.shields.io/crates/v/tblis-ffi.svg)](https://crates.io/crates/tblis-ffi) | [![API Documentation](https://docs.rs/tblis-ffi/badge.svg)](https://docs.rs/tblis-ffi) |
| Crate for Source (tblis-src) | [![Crate](https://img.shields.io/crates/v/tblis-src.svg)](https://crates.io/crates/tblis-src) | [![API Documentation](https://docs.rs/tblis-src/badge.svg)](https://docs.rs/tblis-src) |
| FFI Binding | [9b95712](https://github.com/MatthewsResearchGroup/tblis/commit/9b95712966cb8804be51c62bfd6207957f62bc6f) after [![v1.3.0](https://img.shields.io/github/v/release/MatthewsResearchGroup/tblis)](https://github.com/MatthewsResearchGroup/tblis/releases/tag/v1.3.0) |

</center>

## Example

The following example is to perform contraction:
$$
G_{pqrs} = \sum_{\mu \nu \kappa \lambda} C_{\mu p} C_{\nu q} E_{\mu \nu \kappa \lambda} C_{\kappa r} C_{\lambda s}
$$
This tensor contraction is utilized in electronic structure (electronic integral in atomic orbital basis $E_{\mu \nu \kappa \lambda}$ to molecular orbital basis $G_{pqrs}$).

To run the following code, you may need to
- activate crate feature `ndarray` to make conversion between `ndarray::{Array, ArrayView, ArrayMut}` and `tblis::TblisTensor`;
- properly link libtblis.so in your project (also see crate [tblis-ffi](https://docs.rs/tblis-ffi) and [tblis-src](https://docs.rs/tblis-ffi) for more information).

The following code spinnet performs this contraction. For full code, please see the collapsed section.

<details>

<summary>Collapsed code of the full example</summary>

```rust
// Must declare crate `tblis-src` if you want link tblis dynamically.
// You can also call the following code in `build.rs`, instead of using tblis-src:
//     println!("cargo:rustc-link-lib=tblis");
extern crate tblis_src;

use ndarray::prelude::*;
use tblis::prelude::*;

// dummy setting of matrix C and tensor E
let (nao, nocc): (usize, usize) = (3, 2);
let vec_c: Vec<f64> = (0..nao * nocc).map(|x| x as f64).collect();
let vec_e: Vec<f64> = (0..nao * nao * nao * nao).map(|x| x as f64).collect();

let arr_c = ArrayView2::from_shape((nao, nocc), &vec_c).unwrap();
let arr_e = ArrayView4::from_shape((nao, nao, nao, nao), &vec_e).unwrap();

/// # Parameters
/// - `arr_c`: coefficient matrix $C_{\mu p}$
/// - `arr_s`: electronic integral $E_{\mu \nu \kappa \lambda}$ (in atomic orbital basis)
///
/// # Returns
/// - `arr_g`: electronic integral $G_{pqrs}$ (in molecular orbital basis)
fn ao2mo(arr_c: ArrayView2<f64>, arr_e: ArrayView4<f64>) -> Array4<f64> {
    // transform ndarray objects to tblis objects
    let tsr_c = unsafe { arr_c.to_tblis_tensor() };
    let tsr_e = unsafe { arr_e.to_tblis_tensor() };

    // generate operands and perform contraction
    let operands = [&tsr_c, &tsr_c, &tsr_e, &tsr_c, &tsr_c];
    let out_g = unsafe {
        tblis_einsum(
            "μi,νa,μνκλ,κj,λb->iajb", // einsum subscripts
            &operands,                // tensors to be contracted
            "optimal",                // contraction strategy (see crate opt-einsum-path)
            None,                     // memory limit (None means no limit, see crate opt-einsum-path)
            true,                     // row-major (true) or col-major (false)
            None,                     // pre-allocated output tensor (None to allocate internally)
        )
    };
    let (vec_g, tsr_g) = out_g.unwrap(); // (underlying data, tensor shape/stride info)

    // transform tblis object back to ndarray object
    let arr_g = (vec_g, tsr_g).into_array().into_dimensionality().unwrap();
    arr_g
}

let arr_g = ao2mo(arr_c, arr_e);
println!("{:?}", arr_g);
```

</details>

```rust,ignore
// Must declare crate `tblis-src` if you want link tblis dynamically.
// You can also call the following code in `build.rs`, instead of using tblis-src:
//     println!("cargo:rustc-link-lib=tblis");
extern crate tblis_src;

use ndarray::prelude::*;
use tblis::prelude::*;

/// # Parameters
/// - `arr_c`: coefficient matrix $C_{\mu p}$
/// - `arr_s`: electronic integral $E_{\mu \nu \kappa \lambda}$ (in atomic orbital basis)
///
/// # Returns
/// - `arr_g`: electronic integral $G_{pqrs}$ (in molecular orbital basis)
fn ao2mo(arr_c: ArrayView2<f64>, arr_e: ArrayView4<f64>) -> Array4<f64> {
    // transform ndarray objects to tblis objects
    let tsr_c = unsafe { arr_c.to_tblis_tensor() };
    let tsr_e = unsafe { arr_e.to_tblis_tensor() };

    // generate operands and perform contraction
    let operands = [&tsr_c, &tsr_c, &tsr_e, &tsr_c, &tsr_c];
    let out_g = unsafe {
        tblis_einsum(
            "μi,νa,μνκλ,κj,λb->iajb", // einsum subscripts
            &operands,                // tensors to be contracted
            "optimal",                // contraction strategy (see crate opt-einsum-path)
            None,                     // memory limit (None means no limit, see crate opt-einsum-path)
            true,                     // row-major (true) or col-major (false)
            None,                     // pre-allocated output tensor (None to allocate internally)
        )
    };
    let (vec_g, tsr_g) = out_g.unwrap(); // (underlying data, tensor shape/stride info)

    // transform tblis object back to ndarray object
    let arr_g = (vec_g, tsr_g).into_array().into_dimensionality().unwrap();
    arr_g
}

let arr_g = ao2mo(arr_c, arr_e);
```

## Cargo features

Optional features:

- `ndarray`: Supports conversion from ndarray objects (`Array`, `ArrayView`, `ArrayMut`) to `TblisTensor`; conversion from `TblisTensor` to ndarray object (`ArrayD`).
- `dynamic_loading`: Supports dynamic loading (for dependency crate tblis-ffi).

## Installation (linkage to `libtblis.so`)

If you wish using dynamic loading (instead of dynamic/static linking), refer to the next subsection "Dynamic loading".

You can either
- link library `tblis` manually
- use cargo crate [tblis-src](https://docs.rs/tblis-src) with pre-built libtblis.so
- use cargo crate [tblis-src](https://docs.rs/tblis-src) and build-from-source

### Link library `tblis` manually

By this way, you can directly use cargo crate [tblis](https://docs.rs/tblis) or [tblis-ffi](https://docs.rs/tblis-ffi), without using [tblis-src](https://docs.rs/tblis-src).

It is recommended to link `libtblis.so` by dynamic linking. Making sure your library is in environment variable `LD_LIBRARY_PATH`, then

```rust
// build.rs
println!("cargo:rustc-link-lib=static=tblis");
```

### Use cargo crate [tblis-src](https://docs.rs/tblis-src) with pre-built libtblis.so

By this way, you need to add `tblis-src` as Cargo.toml dependency:

```toml
tblis-src = { version = "0.1" }
```

and then export this crate in your lib.rs/main.rs:

```rust
extern crate tblis-src;
```

### Use cargo crate [tblis-src](https://docs.rs/tblis-src) and build-from-source

You can use crago feature `build_from_source` to automatically build TBLIS with default configuration.

cargo crate [tblis-src](https://docs.rs/tblis-src) has the following cargo features:

- **`build_from_source`**: This will use CMake (cmake > 3.23, c++20 standard), and use the code from git submodule to compile tblis. Though this option can be developer-friendly (you do not need to perform any other configurations to make program compile and run by cargo), `build_from_source` does not provide customized compilation.

    CMake configurable variables (can be defined as environment variables):
    - `TBLIS_SRC`: Git repository source directory or URL. All git submodules (marray, blis, tci) should be properly downloaded.
    - `TBLIS_VER`: Git repository version (branch or tag). Default to be `develop`.

- **`static`**: This will link static libary instead of dynamic one. Please note that static linking may require additional dynamic library linking, which should be configured manually by developer in `build.rs` or environment variables `RUSTFLAGS`. Static linking can be difficult when searching symbols, and we recommend dynamic linking in most cases.

## Dynamic loading

This crate supports dynamic loading.

If you want to use dynamic loading, please enable cargo feature `dynamic_loading` when cargo build.

The dynamic loading will try to find proper library when your program initializes.
- This crate will automatically detect proper libraries, if these libraries are in environmental path `LD_LIBRARY_PATH` (Linux) `DYLD_LIBRARY_PATH` (Mac OS), `PATH` (Windows).
- If you want to override the library to be loaded, please set these shell environmental variable `RSTSR_DYLOAD_TBLIS` to the dynamic library path.

