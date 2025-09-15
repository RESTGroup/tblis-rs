#[cfg(test)]
mod tests {
    use tblis_ffi::tblis::*;

    #[test]
    fn test_workable() {
        let mut data_a = [0.0f64; 10 * 9 * 2 * 5];
        let mut data_b = [0.0f64; 7 * 5 * 9 * 8];
        let mut data_c = [0.0f64; 7 * 2 * 10 * 8];

        let a = tblis_tensor {
            type_: TYPE_DOUBLE,
            conj: 0,
            scalar: tblis_scalar { data: tblis_scalar_scalar { d: 0.0 }, type_: TYPE_DOUBLE },
            data: data_a.as_mut_ptr() as *mut _,
            ndim: 4,
            len: [10, 9, 2, 5].as_mut_ptr(),
            stride: [1, 10, 90, 180].as_mut_ptr(),
        };

        let b = tblis_tensor {
            type_: TYPE_DOUBLE,
            conj: 0,
            scalar: tblis_scalar { data: tblis_scalar_scalar { d: 0.0 }, type_: TYPE_DOUBLE },
            data: data_b.as_mut_ptr() as *mut _,
            ndim: 4,
            len: [7, 5, 9, 8].as_mut_ptr(),
            stride: [1, 7, 35, 315].as_mut_ptr(),
        };

        let mut c = tblis_tensor {
            type_: TYPE_DOUBLE,
            conj: 0,
            scalar: tblis_scalar { data: tblis_scalar_scalar { d: 0.0 }, type_: TYPE_DOUBLE },
            data: data_c.as_mut_ptr() as *mut _,
            ndim: 4,
            len: [7, 2, 10, 8].as_mut_ptr(),
            stride: [1, 7, 14, 140].as_mut_ptr(),
        };

        unsafe {
            tblis_tensor_mult(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &a,
                c"cebf".as_ptr(),
                &b,
                c"afed".as_ptr(),
                &mut c,
                c"abcd".as_ptr(),
            );
        }

        /* Test case of the following C code.

        ```C
        double data_A[10*9*2*5];
        tblis_tensor A;
        tblis_init_tensor_d(&A, 4, (len_type[]){10, 9, 2, 5},
                            data_A, (stride_type[]){1, 10, 90, 180});

        double data_B[7*5*9*8];
        tblis_tensor B;
        tblis_init_tensor_d(&B, 4, (len_type[]){7, 5, 9, 8},
                            data_B, (stride_type[]){1, 7, 35, 315});

        double data_C[7*2*10*8];
        tblis_tensor C;
        tblis_init_tensor_d(&C, 4, (len_type[]){7, 2, 10, 8},
                            data_C, (stride_type[]){1, 7, 14, 140});

        // initialize data_A and data_B...

        // this computes C[abcd] += A[cebf] B[afed]
        tblis_tensor_mult(NULL, NULL, &A, "cebf", &B, "afed", &C, "abcd");
        ```

        */
    }

    #[test]
    fn test_matmul() {
        let data_a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let data_b: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];
        let mut data_c: Vec<f64> = vec![0.0; 4];

        let a = tblis_tensor {
            type_: TYPE_DOUBLE,
            conj: 0,
            scalar: tblis_scalar { data: tblis_scalar_scalar { d: 1.0 }, type_: TYPE_DOUBLE },
            data: data_a.as_ptr() as *mut _,
            ndim: 2,
            len: [2, 2].as_mut_ptr(),
            stride: [2, 1].as_mut_ptr(),
        };

        let b = tblis_tensor {
            type_: TYPE_DOUBLE,
            conj: 0,
            scalar: tblis_scalar { data: tblis_scalar_scalar { d: 1.0 }, type_: TYPE_DOUBLE },
            data: data_b.as_ptr() as *mut _,
            ndim: 2,
            len: [2, 2].as_mut_ptr(),
            stride: [2, 1].as_mut_ptr(),
        };

        let mut c = tblis_tensor {
            type_: TYPE_DOUBLE,
            conj: 0,
            scalar: tblis_scalar { data: tblis_scalar_scalar { d: 1.0 }, type_: TYPE_DOUBLE },
            data: data_c.as_mut_ptr() as *mut _,
            ndim: 2,
            len: [2, 2].as_mut_ptr(),
            stride: [2, 1].as_mut_ptr(),
        };

        unsafe {
            tblis_tensor_mult(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &a,
                c"ij".as_ptr(),
                &b,
                c"jk".as_ptr(),
                &mut c,
                c"ik".as_ptr(),
            );
        }

        assert_eq!(data_c, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
