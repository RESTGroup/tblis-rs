extern crate tblis_src;

use rayon::prelude::*;
use tblis::prelude::*;

fn gen_array(shape: &[isize]) -> Vec<f64> {
    let size = shape.iter().product();
    (0..size).into_par_iter().map(|i| (i as f64 + 0.2).cos()).collect()
}

fn fp(vec: &[f64]) -> f64 {
    vec.par_iter().enumerate().map(|(i, &x)| (i as f64).cos() * x).sum()
}

#[test]
#[ignore = "bench purpose"]
fn test() {
    // for this case, numpy can be very slow
    let nao: isize = 96;
    let shape_e = [nao, nao, nao, nao];
    let stride_e = [nao * nao * nao, nao * nao, nao, 1];
    let vec_e = gen_array(&shape_e);
    let tsr_e = TblisTensor::new(vec_e.as_ptr() as *mut f64, &shape_e, &stride_e);

    let operands = [&tsr_e, &tsr_e];

    let subscripts_list = [
        "abxy, xycd -> abcd", // naive gemm case, 2 * n^6
        "axyz, xyzb -> ab",   // naive gemm case, 2 * n^5
        "axyz, bxyz -> ab",   // naive syrk case,     n^5
        "axyz, ybzx -> ab",   // comp  gemm case, 2 * n^5
        "axby, yacx -> abc",  // batch gemm case, 2 * n^5
        "xpay, aybx -> ab",   // complicate case, 2 * n^4
    ];
    let repeat_list = [5, 20, 20, 20, 20, 20];

    for (subscripts, &nrepeat) in subscripts_list.iter().zip(repeat_list.iter()) {
        println!("Subscripts: {subscripts}");
        let time = std::time::Instant::now();
        let mut vec_g = vec![];
        for _ in 0..nrepeat {
            let (vec_g_temp, _) = unsafe { tblis_einsum(subscripts, &operands, true, None, true, None).unwrap() };
            vec_g = vec_g_temp;
        }
        println!("elapsed time: {:12.6?} (avg of {nrepeat:2} repeats)", time.elapsed() / nrepeat);
        println!("fingerprint : {:20.12}", fp(&vec_g));
    }
}
