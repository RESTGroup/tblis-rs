use crate::prelude::*;
use duplicate::duplicate_item;
use ndarray::prelude::*;
use ndarray::Data;
use opt_einsum_path::typing::SizeLimitType;
use opt_einsum_path::PathOptimizer;

#[duplicate_item(ArrayItem; [ArrayBase<S, D>]; [&ArrayBase<S, D>])]
impl<S, T, D> ToTblisTensor<T> for ArrayItem
where
    S: Data<Elem = T>,
    T: TblisFloatAPI,
    D: Dimension,
{
    fn to_tblis_tensor(&self) -> TblisTensor<T> {
        let view = self.view();
        let shape = view.shape().iter().map(|&s| s as isize).collect::<Vec<isize>>();
        let stride = ArrayBase::strides(&view).to_vec();
        let data_ptr = view.as_ptr() as *mut T;
        TblisTensor::new(data_ptr, &shape, &stride)
    }
}

/// Trait to convert `(Vec<T>, TblisTensor<T>)` to [`ndarray::ArrayD`].
pub trait ArrayFromTblisTensor {
    type Out;
    fn into_array(self) -> Self::Out;
}

impl<T> ArrayFromTblisTensor for (Vec<T>, TblisTensor<T>)
where
    T: TblisFloatAPI,
{
    type Out = ArrayD<T>;
    fn into_array(self) -> ArrayD<T> {
        let (vec, tsr) = self;
        let shape = tsr.shape.iter().map(|&s| s as usize).collect::<Vec<usize>>();
        ArrayD::from_shape_vec(IxDyn(&shape), vec).unwrap()
    }
}

/// Convert `(Vec<T>, TblisTensor<T>)` to [`ndarray::ArrayD`].
pub fn array_from_tblis_tensor<T>(dat: (Vec<T>, TblisTensor<T>)) -> ArrayD<T>
where
    T: TblisFloatAPI,
{
    dat.into_array()
}

/// High-level Einstein summation interface for [`ndarray::ArrayBase`].
///
/// This function only works when crate feature `ndarray` is enabled.
///
/// All the parameters are the same as in [`tblis_einsum`], except for the parameter `operands`.
///
/// # Parameters
///
/// - `operands`: Owned or references to ndarray objects (tensors to be contracted).
///
/// # Example
///
/// ```rust
/// // Must declare crate `tblis-src` if you want link tblis dynamically.
/// // You can also call the following code in `build.rs`, instead of using crate `tblis-src`:
/// //     println!("cargo:rustc-link-lib=tblis");
/// extern crate tblis_src;
///
/// use ndarray::prelude::*;
/// use tblis::prelude::*;
///
/// // dummy setting of matrix C and tensor E
/// let (nao, nmo): (usize, usize) = (3, 2);
/// let vec_c: Vec<f64> = (0..nao * nmo).map(|x| x as f64).collect();
/// let vec_e: Vec<f64> = (0..nao * nao * nao * nao).map(|x| x as f64).collect();
///
/// let arr_c = ArrayView2::from_shape((nao, nmo), &vec_c).unwrap();
/// let arr_e = ArrayView4::from_shape((nao, nao, nao, nao), &vec_e).unwrap();
///
/// /// # Parameters
/// /// - `arr_c`: coefficient matrix $C_{\mu p}$
/// /// - `arr_s`: electronic integral $E_{\mu \nu \kappa \lambda}$ (in atomic orbital basis)
/// ///
/// /// # Returns
/// /// - `arr_g`: electronic integral $G_{pqrs}$ (in molecular orbital basis)
/// fn ao2mo(arr_c: ArrayView2<f64>, arr_e: ArrayView4<f64>) -> Array4<f64> {
///     let view_c = arr_c.view().into_dyn();
///     let view_e = arr_e.view().into_dyn();
///     let operands = [&view_c, &view_c, &view_e, &view_c, &view_c];
///     let arr_g = tblis_einsum_ndarray(
///         "μi,νa,μνκλ,κj,λb->iajb", // einsum subscripts
///         &operands,                // tensors to be contracted
///         "optimal",                // contraction strategy (see crate opt-einsum-path)
///         None,                     // memory limit (None means no limit, see crate opt-einsum-path)
///         true,                     // row-major (true) or col-major (false)
///         None,                     // pre-allocated output tensor (None to allocate internally)
///     )
///     .unwrap();
///
///     // transform to 4-dimensional array
///     arr_g.into_dimensionality().unwrap()
/// }
///
/// let arr_g = ao2mo(arr_c, arr_e);
/// println!("{:?}", arr_g);
/// ```
///
/// # See also
///
/// - [`tblis_einsum`] (native TBLIS interface)
/// - [`tblis_einsum_f`] (failable native TBLIS interface)
/// - [`tblis_einsum_ndarray_f`] (failable ndarray interface)
pub fn tblis_einsum_ndarray<T, A>(
    subscripts: &str,
    operands: &[&A],
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
    row_major: bool,
    out_tblis_tensor: Option<&mut TblisTensor<T>>,
) -> Option<ArrayD<T>>
where
    T: TblisFloatAPI,
    A: ToTblisTensor<T>,
{
    tblis_einsum_ndarray_f(subscripts, operands, optimize, memory_limit, row_major, out_tblis_tensor).unwrap()
}

/// High-level failable Einstein summation interface for [`ndarray::ArrayBase`].
///
/// # See also
///
/// - [`tblis_einsum`] (native TBLIS interface)
/// - [`tblis_einsum_f`] (failable native TBLIS interface)
/// - [`tblis_einsum_ndarray`] (ndarray interface)
pub fn tblis_einsum_ndarray_f<T, A>(
    subscripts: &str,
    operands: &[&A],
    optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
    row_major: bool,
    out_tblis_tensor: Option<&mut TblisTensor<T>>,
) -> Result<Option<ArrayD<T>>, String>
where
    T: TblisFloatAPI,
    A: ToTblisTensor<T>,
{
    let tblis_operands: Vec<TblisTensor<T>> = operands.iter().map(|x| x.to_tblis_tensor()).collect();
    let tblis_operands_ref: Vec<&TblisTensor<T>> = tblis_operands.iter().collect();
    let res =
        unsafe { tblis_einsum_f(subscripts, &tblis_operands_ref, optimize, memory_limit, row_major, out_tblis_tensor) };
    match res {
        Ok(Some(out)) => Ok(Some(out.into_array())),
        Ok(None) => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
#[cfg(feature = "ndarray")]
mod test_ndarray_native {
    #[test]
    #[allow(clippy::let_and_return)]
    fn test_ndarray_native() {
        use crate::prelude::*;
        use ndarray::prelude::*;
        let (nao, nmo): (usize, usize) = (3, 2);
        let vec_c: Vec<f64> = (0..nao * nmo).map(|x| x as f64).collect();
        let vec_e: Vec<f64> = (0..nao * nao * nao * nao).map(|x| x as f64).collect();

        let arr_c = ArrayView2::from_shape((nao, nmo), &vec_c).unwrap();
        let arr_e = ArrayView4::from_shape((nao, nao, nao, nao), &vec_e).unwrap();

        /// # Parameters
        /// - `arr_c`: coefficient matrix $C_{\mu p}$
        /// - `arr_s`: electronic integral $E_{\mu \nu \kappa \lambda}$ (in atomic orbital basis)
        ///
        /// # Returns
        /// - `arr_g`: electronic integral $G_{pqrs}$ (in molecular orbital basis)
        fn ao2mo(arr_c: ArrayView2<f64>, arr_e: ArrayView4<f64>) -> Array4<f64> {
            let view_c = arr_c.view().into_dyn();
            let view_e = arr_e.view().into_dyn();
            let operands = [&view_c, &view_c, &view_e, &view_c, &view_c];
            let arr_g = tblis_einsum_ndarray(
                "μi,νa,μνκλ,κj,λb->iajb", // einsum subscripts
                &operands,                // tensors to be contracted
                "optimal",                // contraction strategy (see crate opt-einsum-path)
                None,                     // memory limit (None means no limit, see crate opt-einsum-path)
                true,                     // row-major (true) or col-major (false)
                None,                     // pre-allocated output tensor (None to allocate internally)
            )
            .unwrap();

            // transform to 4-dimensional array
            arr_g.into_dimensionality().unwrap()
        }

        let arr_g = ao2mo(arr_c, arr_e);
        println!("{:?}", arr_g);
    }

    #[test]
    #[allow(clippy::let_and_return)]
    fn test_ndarray_workable() {
        use crate::prelude::*;
        use ndarray::prelude::*;
        let (nao, nmo): (usize, usize) = (3, 2);
        let vec_c: Vec<f64> = (0..nao * nmo).map(|x| x as f64).collect();
        let vec_e: Vec<f64> = (0..nao * nao * nao * nao).map(|x| x as f64).collect();

        let arr_c = ArrayView2::from_shape((nao, nmo), &vec_c).unwrap();
        let arr_e = ArrayView4::from_shape((nao, nao, nao, nao), &vec_e).unwrap();

        fn ao2mo(arr_c: ArrayView2<f64>, arr_e: ArrayView4<f64>) -> Array4<f64> {
            // transform ndarray objects to tblis objects
            let tsr_c = arr_c.to_tblis_tensor();
            let tsr_e = arr_e.to_tblis_tensor();

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
    }
}
