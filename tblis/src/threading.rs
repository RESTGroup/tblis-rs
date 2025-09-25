use core::ffi::*;

/// Get or set the number of threads used by TBLIS.
///
/// # Note
///
/// TBLIS threading is similar to BLIS, where thread numbers are controlled for each logic
/// application thread separately (for rayon framework, each spawned rayon thread session).
///
/// Also see [BLIS documentation on threading](https://github.com/flame/blis/blob/master/docs/Multithreading.md#globally-at-runtime).
/// However, note that "global" there means parallel level to each iteration level in BLIS, not
/// global to the entire program thread pool (the convention adopted by MKL).
pub fn tblis_get_num_threads() -> usize {
    unsafe { tblis_ffi::tblis::tblis_get_num_threads() as usize }
}

/// Set the number of threads used by TBLIS.
///
/// # Note
///
/// TBLIS threading is similar to BLIS, where thread numbers are controlled for each logic
/// application thread separately (for rayon framework, each spawned rayon thread session).
///
/// Also see [BLIS documentation on threading](https://github.com/flame/blis/blob/master/docs/Multithreading.md#globally-at-runtime).
/// However, note that "global" there means parallel level to each iteration level in BLIS, not
/// global to the entire program thread pool (the convention adopted by MKL).
pub fn tblis_set_num_threads(num_threads: usize) {
    unsafe { tblis_ffi::tblis::tblis_set_num_threads(num_threads as c_uint) }
}

#[test]
#[ignore]
fn check_rayon_par() {
    extern crate tblis_src;
    use rayon::prelude::*;

    (0..16).into_par_iter().for_each(|i| {
        let num_threads = rayon::current_num_threads();
        tblis_set_num_threads(1);
        println!("[THREAD] idx {i:2}, get_threads {num_threads:2}");
    });
    let num_threads = rayon::current_num_threads();
    println!("[MAIN  ] get_threads {num_threads:2}");
}
