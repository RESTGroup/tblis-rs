extern crate alloc;
use core::ptr::NonNull;

/// Create an uninitialized vector with the given size.
///
/// This function depends on `aligned_alloc` feature.
/// If `aligned_alloc` is enabled, it will align at 64-bit when size of vector
/// elements is larger than 128.
///
/// # Safety
///
/// Caller must ensure that the vector is properly initialized before using it.
///
/// This is not a very good function, since `set_len` on uninitialized memory is
/// undefined-behavior (UB).
/// Nevertheless, if `T` is some type of `MaybeUninit`, then this will not UB.
pub unsafe fn uninitialized_vec<T>(size: usize) -> Result<Vec<T>, String> {
    unsafe { aligned_uninitialized_vec::<T, 128>(size, 64) }
}

/// Create an unaligned uninitialized vector with the given size.
///
/// # Safety
///
/// Caller must ensure that the vector is properly initialized before using it.
///
/// This is not a very good function, since `set_len` on uninitialized memory is
/// undefined-behavior (UB).
/// Nevertheless, if `T` is some type of `MaybeUninit`, then this will not UB.
#[allow(clippy::uninit_vec)]
pub unsafe fn unaligned_uninitialized_vec<T>(size: usize) -> Result<Vec<T>, String> {
    let mut v: Vec<T> = vec![];
    v.try_reserve_exact(size).map_err(|e| format!("AllocationError: {:?}", e))?;
    unsafe { v.set_len(size) };
    Ok(v)
}

/// Create an uninitialized vector with the given size and alignment.
///
/// - Error: `LayoutError` if the layout cannot be created.
/// - Ok(None): if the size is 0 or allocation fails.
/// - Ok(Some): pointer to the allocated memory.
///
/// https://users.rust-lang.org/t/how-can-i-allocate-aligned-memory-in-rust/33293
pub fn aligned_alloc(numbytes: usize, alignment: usize) -> Result<Option<NonNull<()>>, String> {
    if numbytes == 0 {
        return Ok(None);
    }
    let layout =
        alloc::alloc::Layout::from_size_align(numbytes, alignment).map_err(|e| format!("LayoutError: {:?}", e))?;
    let pointer = NonNull::new(unsafe { alloc::alloc::alloc(layout) }).map(|p| p.cast::<()>());
    Ok(pointer)
}

/// Create an conditionally aligned uninitialized vector with the given size.
///
/// - `N`: condition for alignment; if `N < size`, then this function will not allocate aligned
///   vector.
///
/// # Safety
///
/// Caller must ensure that the vector is properly initialized before using it.
///
/// This is not a very good function, since `set_len` on uninitialized memory is
/// undefined-behavior (UB).
/// Nevertheless, if `T` is some type of `MaybeUninit`, then this will not UB.
#[allow(clippy::uninit_vec)]
pub unsafe fn aligned_uninitialized_vec<T, const N: usize>(size: usize, alignment: usize) -> Result<Vec<T>, String> {
    if size == 0 {
        Ok(vec![])
    } else if size < N {
        unsafe { unaligned_uninitialized_vec(size) }
    } else {
        let sizeof = core::mem::size_of::<T>();
        let pointer = aligned_alloc(size * sizeof, alignment)?;
        if let Some(pointer) = pointer {
            let mut v = unsafe { Vec::from_raw_parts(pointer.as_ptr() as *mut T, size, size) };
            unsafe { v.set_len(size) };
            Ok(v)
        } else {
            Err("Allocation failed (probably due to out-of-memory)".to_string())
        }
    }
}
