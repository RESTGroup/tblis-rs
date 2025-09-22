#![allow(clippy::excessive_precision)]

use rstest::rstest;
use tblis_rs::prelude::*;

fn gen_array(shape: &[isize]) -> Vec<f64> {
    let size = shape.iter().product();
    (0..size).map(|i| (i as f64 + 0.2).cos()).collect()
}

fn fp(vec: &[f64]) -> f64 {
    vec.iter().enumerate().map(|(i, &x)| (i as f64).cos() * x).sum()
}

fn build_shape(s: &str) -> Vec<isize> {
    let valid_chars = "abcdefghijklmnopqABC";
    let sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4, 9, 10, 2, 4, 5, 3, 2, 6];
    s.chars().map(|c| valid_chars.find(c).map(|i| sizes[i]).unwrap_or(1)).collect()
}

fn build_strides(shape: &[isize]) -> Vec<isize> {
    // build row-major strides
    let mut strides = vec![1isize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn build_tblis_tensor(s: &str) -> (Vec<f64>, TblisTensor<f64>) {
    let shape = build_shape(s);
    let strides = build_strides(&shape);
    let mut data = gen_array(&shape);
    let tensor = TblisTensor::new(data.as_mut_ptr(), &shape, &strides);
    (data, tensor)
}

#[rstest]
// [TYPE] scalar-like operations
#[case("a,->a"                                 , vec![2]                 ,   1.1524106074800669)]
#[case("ab,->ab"                               , vec![2, 3]              ,   2.8053551615173387)]
#[case(",ab,->ab"                              , vec![2, 3]              ,   2.7494348327775620)]
#[case(",,->"                                  , vec![]                  ,   0.9413838371083508)]
// [TYPE] hadamard-like products
#[case("a,ab,abc->abc"                         , vec![2, 3, 4]           ,   1.1706257404816276)]
#[case("a,b,ab->ab"                            , vec![2, 3]              ,   1.2343776816309531)]
// [TYPE] index-transformations
#[case("ea,fb,gc,hd,abcd->efgh"                , vec![4, 3, 2, 6]        ,  12.1332755429121821)]
#[case("ea,fb,abcd,gc,hd->efgh"                , vec![4, 3, 2, 6]        ,  12.1332755429121750)]
#[case("abcd,ea,fb,gc,hd->efgh"                , vec![4, 3, 2, 6]        ,  12.1332755429121910)]
// [TYPE] complex contractions
#[case("acdf,jbje,gihb,hfac,gfac,gifabc,hfac"  , vec![5, 4]              ,   0.1814690776365560)]
#[case("acdf,jbje,gihb,hfac,gfac,gifabc,hfac"  , vec![5, 4]              ,   0.1814690776365560)]
#[case("cd,bdhe,aidb,hgca,gc,hgibcd,hgac"      , vec![4]                 ,  -4.4698469414635698)]
#[case("abhe,hidj,jgba,hiab,gab"               , vec![5, 4]              , -16.9026985074718148)]
#[case("bde,cdh,agdb,hica,ibd,hgicd,hiac"      , vec![4]                 ,  -5.9363077640651882)]
#[case("chd,bde,agbc,hiad,hgc,hgi,hiad"        , vec![4]                 ,  -0.3797503358113263)]
#[case("chd,bde,agbc,hiad,bdi,cgh,agdb"        , vec![4]                 ,   9.3347856408443466)]
#[case("bdhe,acad,hiab,agac,hibd"              , vec![4, 2]              ,   0.0817842308123281)]
// [TYPE] collapse
#[case("ab,ab,c->"                             , vec![]                  ,  -0.7073635910335989)]
#[case("ab,ab,c->c"                            , vec![4]                 ,   6.9733109228592252)]
#[case("ab,ab,cd,cd->"                         , vec![]                  ,  30.2845404955025153)]
#[case("ab,ab,cd,cd->ac"                       , vec![2, 4]              ,   5.1155948313305153)]
#[case("ab,ab,cd,cd->cd"                       , vec![4, 5]              ,   2.9488487899924660)]
#[case("ab,ab,cd,cd,ef,ef->"                   , vec![]                  , 177.9078409242766838)]
// [TYPE] outer products
#[case("ab,cd,ef->abcdef"                      , vec![2, 3, 4, 5, 4, 3]  ,   7.2612571312420187)]
#[case("ab,cd,ef->acdf"                        , vec![2, 4, 5, 3]        ,   0.2218804695072113)]
#[case("ab,cd,de->abcde"                       , vec![2, 3, 4, 5, 4]     ,   5.6145252091023128)]
#[case("ab,cd,de->be"                          , vec![3, 4]              ,  -0.0112868166168827)]
#[case("ab,bcd,cd->abcd"                       , vec![2, 3, 4, 5]        ,   1.2455894084757055)]
#[case("ab,bcd,cd->abd"                        , vec![2, 3, 5]           ,   1.1900669479088557)]
// [TYPE] cases that have previously failed
#[case("eb,cb,fb->cef"                         , vec![4, 4, 3]           ,  -0.3948422850851117)]
#[case("dd,fb,be,cdb->cef"                     , vec![4, 4, 3]           ,   0.0433826323804478)]
#[case("bca,cdb,dbf,afc->"                     , vec![]                  ,   1.8790765302648753)]
#[case("dcc,fce,ea,dbf->ab"                    , vec![2, 3]              ,  -0.0995815485371659)]
#[case("fdf,cdd,ccd,afe->ae"                   , vec![2, 4]              ,  -1.8204695389889403)]
#[case("abcd,ad"                               , vec![3, 4]              ,  -6.2411233103176258)]
#[case("ed,fcd,ff,bcf->be"                     , vec![3, 4]              ,   0.5335137116293984)]
#[case("baa,dcf,af,cde->be"                    , vec![3, 4]              ,   6.7338928837379548)]
#[case("bd,db,eac->ace"                        , vec![2, 4, 4]           ,   0.5294245507683408)]
#[case("fff,fae,bef,def->abd"                  , vec![2, 3, 5]           ,  -6.1168007036329932)]
#[case("efc,dbc,acf,fd->abe"                   , vec![2, 3, 4]           ,   2.1288617659874238)]
// [TYPE] Inner products
#[case("ab,ab"                                 , vec![]                  ,   2.8946232078487490)]
#[case("ab,ba"                                 , vec![]                  ,   1.3829944294961767)]
#[case("abc,abc"                               , vec![]                  ,  12.0867609307166504)]
#[case("abc,bac"                               , vec![]                  ,   0.7467158730806971)]
#[case("abc,cba"                               , vec![]                  ,   1.1905629015868808)]
// [TYPE] GEMM test cases
#[case("ab,bc"                                 , vec![2, 4]              ,   3.0203306308641209)]
#[case("ab,cb"                                 , vec![2, 4]              ,   2.4279291273425985)]
#[case("ba,bc"                                 , vec![2, 4]              ,   0.6630550300956001)]
#[case("ba,cb"                                 , vec![2, 4]              ,   1.0775626157664493)]
#[case("abcd,cd"                               , vec![2, 3]              ,  28.5705554819948730)]
#[case("abcd,ab"                               , vec![4, 5]              ,  29.2315920761960335)]
#[case("abcd,cdef"                             , vec![2, 3, 4, 3]        , -13.9084614811298160)]
#[case("abcd,cdef->feba"                       , vec![3, 4, 3, 2]        ,   0.8964372441283897)]
#[case("abcd,efdc"                             , vec![2, 3, 4, 3]        ,   6.2879216163017242)]
// [TYPE] Inner than dot
#[case("aab,bc->ac"                            , vec![2, 4]              ,   2.7142044202227504)]
#[case("ab,bcc->ac"                            , vec![2, 4]              ,   0.6687831883550068)]
#[case("aab,bcc->ac"                           , vec![2, 4]              ,   0.5820900839998362)]
#[case("baa,bcc->ac"                           , vec![2, 4]              ,   5.3442670128010361)]
#[case("aab,ccb->ac"                           , vec![2, 4]              ,   0.3602374930760941)]
// [TYPE] Randomly build test caes
#[case("aab,fa,df,ecc->bde"                    , vec![3, 5, 4]           ,  -0.5103260493034085)]
#[case("ecb,fef,bad,ed->ac"                    , vec![2, 4]              ,  -1.1285608137144998)]
#[case("bcf,bbb,fbf,fc->"                      , vec![]                  ,   0.5607438675587967)]
#[case("bb,ff,be->e"                           , vec![4]                 ,   0.4876389976600121)]
#[case("bcb,bb,fc,fff->"                       , vec![]                  ,   0.1186616195257864)]
#[case("fbb,dfd,fc,fc->"                       , vec![]                  ,  -0.1797709337347626)]
#[case("afd,ba,cc,dc->bf"                      , vec![3, 3]              ,  -0.8876593529208174)]
#[case("adb,bc,fa,cfc->d"                      , vec![5]                 ,   0.7860343998485582)]
#[case("bbd,bda,fc,db->acf"                    , vec![2, 4, 3]           ,   2.2992302890561374)]
#[case("dba,ead,cad->bce"                      , vec![3, 4, 4]           ,   0.7414744179866388)]
#[case("aef,fbc,dca->bde"                      , vec![3, 5, 4]           ,   1.4158672478067489)]
// [TYPE] Transpose only
#[case("ea->ea"                                , vec![4, 2]              ,   4.2779003835689338)]
#[case("fb->fb"                                , vec![3, 3]              ,   4.3272474273084374)]
#[case("abcd->dcab"                            , vec![5, 4, 2, 3]        ,   5.0109395102756960)]
#[case("gc->cg"                                , vec![4, 2]              ,   0.2680112599856153)]
#[case("hd->dh"                                , vec![5, 6]              ,   4.4819580717183714)]
#[case("efgh->hfge"                            , vec![6, 3, 2, 4]        ,   2.2237855229147154)]
#[case("acdf->afcd"                            , vec![2, 3, 4, 5]        ,   3.2392076661644182)]
#[case("gihb->ghib"                            , vec![2, 6, 5, 3]        ,  -2.1102327279824999)]
#[case("hfac->cfah"                            , vec![4, 3, 2, 6]        ,   2.2237855229146861)]
#[case("gfac->cgaf"                            , vec![4, 2, 2, 3]        ,   4.0858816458932754)]
#[case("gifabc->abifcg"                        , vec![2, 3, 5, 3, 4, 2]  ,  -0.8209082897628117)]
#[case("hfac->cfha"                            , vec![4, 3, 6, 2]        ,  -1.3035489268162346)]
fn test_einsum(#[case] einsum_str: &str, #[case] ref_shape: Vec<isize>, #[case] ref_fp: f64) {
    let einsum_str_inp = einsum_str.split("->").next().unwrap().split(',').collect::<Vec<&str>>();
    let tensors = einsum_str_inp.iter().map(|s| build_tblis_tensor(s)).collect::<Vec<_>>();
    let tblis_tensors = tensors.iter().map(|(_, t)| t).collect::<Vec<_>>();
    let (out_data, out_tensor) = tblis_einsum(einsum_str, &tblis_tensors, true, None, true, None).unwrap();
    let out_fp = fp(&out_data);
    assert_eq!(out_tensor.shape, ref_shape);
    assert!((out_fp - ref_fp).abs() < 1e-10);
}
