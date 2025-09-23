# TBLIS Bindings for Rust

This crate contains TBLIS FFI bindings.

[TBLIS](https://github.com/MatthewsResearchGroup/tblis) (Tensor BLIS, The Tensor-Based Library Instantiation Software) can perform various tensor operations (multiplication, addition, reduction, transposition, etc.) efficiently on single-node CPU. This library can be an underlying driver for performing einsum (Einstein summation).

<center>

| Resources | Badges | API Document |
|--|--|--|
| Crate for Wrapper (tblis) | [![Crate](https://img.shields.io/crates/v/tblis.svg)](https://crates.io/crates/tblis) | [![API Documentation](https://docs.rs/tblis/badge.svg)](https://docs.rs/tblis) |
| Crate for FFI (tblis-ffi) | [![Crate](https://img.shields.io/crates/v/tblis-ffi.svg)](https://crates.io/crates/tblis-ffi) | [![API Documentation](https://docs.rs/tblis-ffi/badge.svg)](https://docs.rs/tblis-ffi) |
| Crate for Source (tblis-src) | [![Crate](https://img.shields.io/crates/v/tblis-src.svg)](https://crates.io/crates/tblis-src) | [![API Documentation](https://docs.rs/tblis-src/badge.svg)](https://docs.rs/tblis-src) |
| FFI Binding | [9b95712](https://github.com/MatthewsResearchGroup/tblis/commit/9b95712966cb8804be51c62bfd6207957f62bc6f) after [![v1.3.0](https://img.shields.io/github/v/release/MatthewsResearchGroup/tblis)](https://github.com/MatthewsResearchGroup/tblis/releases/tag/v1.3.0) |

</center>

We refer the readme file of crate [tblis](tblis/readme.md) (Minimal wrapper), [tblis-ffi](tblis-ffi/readme.md) (FFI bindings) and [tblis-src](tblis-src/readme.md) (source code build) for more information.
