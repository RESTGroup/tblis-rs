#![allow(clippy::excessive_precision)]

extern crate tblis_src;

use num::complex::ComplexFloat;
use rstest::rstest;
use tblis::prelude::*;

#[allow(non_camel_case_types)]
type c64 = num::Complex<f64>;

fn gen_array(shape: &[isize]) -> Vec<c64> {
    let size = shape.iter().product();
    (0..size).map(|i| c64::new((i as f64 + 0.2).cos(), -0.5 * (i as f64 + 0.4).sin())).collect()
}

fn fp(vec: &[c64]) -> c64 {
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

fn build_tblis_tensor(s: &str) -> (Vec<c64>, TblisTensor<c64>) {
    let shape = build_shape(s);
    let strides = build_strides(&shape);
    let mut data = gen_array(&shape);
    let tensor = TblisTensor::new(data.as_mut_ptr(), &shape, &strides);
    (data, tensor)
}

#[rstest]
// [TYPE] scalar-like operations
#[case("a,->a"                                 , vec![2]                 , (  1.0626633964750658,  -0.6806902927156129))]
#[case("ab,->ab"                               , vec![2, 3]              , (  2.6791295309972907,  -1.1926934244521328))]
#[case(",ab,->ab"                              , vec![2, 3]              , (  2.3934969629216369,  -1.6905700533121117))]
#[case(",,->"                                  , vec![]                  , (  0.8299159805637262,  -0.5536905426638488))]
// [TYPE] hadamard-like products
#[case("a,ab,abc->abc"                         , vec![2, 3, 4]           , (  1.5336587277409919,  -0.8487366664981919))]
#[case("a,b,ab->ab"                            , vec![2, 3]              , (  0.5769339284293409,  -1.5532266817002940))]
// [TYPE] index-transformations
#[case("ea,fb,gc,hd,abcd->efgh"                , vec![4, 3, 2, 6]        , (  9.2102305248174847,   6.6918481319956342))]
#[case("ea,fb,abcd,gc,hd->efgh"                , vec![4, 3, 2, 6]        , (  9.2102305248174847,   6.6918481319956342))]
#[case("abcd,ea,fb,gc,hd->efgh"                , vec![4, 3, 2, 6]        , (  9.2102305248174883,   6.6918481319956307))]
// [TYPE] complex contractions
#[case("acdf,jbje,gihb,hfac,gfac,gifabc,hfac"  , vec![5, 4]              , ( -0.2116097242816026,   0.0454024855951956))]
#[case("acdf,jbje,gihb,hfac,gfac,gifabc,hfac"  , vec![5, 4]              , ( -0.2116097242816026,   0.0454024855951956))]
#[case("cd,bdhe,aidb,hgca,gc,hgibcd,hgac"      , vec![4]                 , ( -7.7635629821390308,   8.5154775536332945))]
#[case("abhe,hidj,jgba,hiab,gab"               , vec![5, 4]              , (-21.3091287330690449, -31.1676144095302625))]
#[case("bde,cdh,agdb,hica,ibd,hgicd,hiac"      , vec![4]                 , ( -3.1103334757846168,   0.9019517948241862))]
#[case("chd,bde,agbc,hiad,hgc,hgi,hiad"        , vec![4]                 , ( -3.9154496906531921,  -0.3719312577909981))]
#[case("chd,bde,agbc,hiad,bdi,cgh,agdb"        , vec![4]                 , ( 21.8169594386688530,  -0.8392270694101933))]
#[case("bdhe,acad,hiab,agac,hibd"              , vec![4, 2]              , (  0.0714978291829508,   0.0179331909288974))]
// [TYPE] collapse
#[case("ab,ab,c->"                             , vec![]                  , ( -1.1440117473937714,  -1.7203308328855442))]
#[case("ab,ab,c->c"                            , vec![4]                 , (  4.7647955571612179,  -2.6302597509298922))]
#[case("ab,ab,cd,cd->"                         , vec![]                  , ( 15.2981766805646124, -10.6028689530129476))]
#[case("ab,ab,cd,cd->ac"                       , vec![2, 4]              , (  2.4818565990824020,  -1.6167222863631672))]
#[case("ab,ab,cd,cd->cd"                       , vec![4, 5]              , (  1.6820837207751058,  -1.6635100382781292))]
#[case("ab,ab,cd,cd,ef,ef->"                   , vec![]                  , ( 50.6236785641111808, -68.0228858281206357))]
// [TYPE] outer products
#[case("ab,cd,ef->abcdef"                      , vec![2, 3, 4, 5, 4, 3]  , (  8.0160220339714989,   9.8106351580935769))]
#[case("ab,cd,ef->acdf"                        , vec![2, 4, 5, 3]        , ( -0.3189002168829093,  -0.5698001035414051))]
#[case("ab,cd,de->abcde"                       , vec![2, 3, 4, 5, 4]     , (  3.3311677876018635,  -0.9646556190738309))]
#[case("ab,cd,de->be"                          , vec![3, 4]              , (  0.0060456268678287,   0.0315858071478905))]
#[case("ab,bcd,cd->abcd"                       , vec![2, 3, 4, 5]        , (  0.7947390762604263,  -1.8313657121740694))]
#[case("ab,bcd,cd->abd"                        , vec![2, 3, 5]           , (  1.1177831190650014,  -0.4536529682239876))]
// [TYPE] cases that have previously failed
#[case("eb,cb,fb->cef"                         , vec![4, 4, 3]           , ( -0.3417214715065875,  -0.2989636671893672))]
#[case("dd,fb,be,cdb->cef"                     , vec![4, 4, 3]           , (  0.0614607960345380,  -0.0292966395473457))]
#[case("bca,cdb,dbf,afc->"                     , vec![]                  , (-11.1976419920901726,   2.6675195489355876))]
#[case("dcc,fce,ea,dbf->ab"                    , vec![2, 3]              , ( -0.0254992770945275,   0.1291919328619402))]
#[case("fdf,cdd,ccd,afe->ae"                   , vec![2, 4]              , ( -2.3807100283394025,  -1.7309388790228271))]
#[case("abcd,ad"                               , vec![3, 4]              , ( -4.2814335847278109,  -0.2423568704494487))]
#[case("ed,fcd,ff,bcf->be"                     , vec![3, 4]              , (  0.0845024493295495,  -1.8943171230788358))]
#[case("baa,dcf,af,cde->be"                    , vec![3, 4]              , (  3.8735271235695636,  -5.4627841204381227))]
#[case("bd,db,eac->ace"                        , vec![2, 4, 4]           , (  0.7557248790243568,   0.9129313989409814))]
#[case("fff,fae,bef,def->abd"                  , vec![2, 3, 5]           , ( -3.2545155626795133,   1.4559659254341435))]
#[case("efc,dbc,acf,fd->abe"                   , vec![2, 3, 4]           , (  0.0046658477715147,  -1.0444897081753224))]
// [TYPE] Inner products
#[case("ab,ab"                                 , vec![]                  , (  2.1078679480925979,  -0.7008159330640431))]
#[case("ab,ba"                                 , vec![]                  , (  1.1922381853320942,   0.0341064650010012))]
#[case("abc,abc"                               , vec![]                  , (  9.0550386293676404,  -2.9217348330317572))]
#[case("abc,bac"                               , vec![]                  , (  0.5665941613556231,   0.1918424986448617))]
#[case("abc,cba"                               , vec![]                  , (  0.7427667439638136,  -8.0266281535986987))]
// [TYPE] GEMM test cases
#[case("ab,bc"                                 , vec![2, 4]              , (  3.0636698684468722,   2.4485748049856646))]
#[case("ab,cb"                                 , vec![2, 4]              , (  1.7712916026623675,  -0.5978413636325484))]
#[case("ba,bc"                                 , vec![2, 4]              , (  1.0343214702468668,   0.9034628404673370))]
#[case("ba,cb"                                 , vec![2, 4]              , (  1.0008955702062043,   0.0746914622620528))]
#[case("abcd,cd"                               , vec![2, 3]              , ( 21.7844254710988672,  -6.8920102056717578))]
#[case("abcd,ab"                               , vec![4, 5]              , ( 22.3929921840377375,  -5.3370424046114540))]
#[case("abcd,cdef"                             , vec![2, 3, 4, 3]        , (-17.5109719823750396,  -3.7862458273559318))]
#[case("abcd,cdef->feba"                       , vec![3, 4, 3, 2]        , (  0.8325862854248018,  -2.9856704587480039))]
#[case("abcd,efdc"                             , vec![2, 3, 4, 3]        , (  4.1316137974856151,  -2.1183100696273223))]
// [TYPE] Inner than dot
#[case("aab,bc->ac"                            , vec![2, 4]              , (  2.6122198655941657,   2.4070577835099365))]
#[case("ab,bcc->ac"                            , vec![2, 4]              , (  0.5536064032919279,   0.8165245266829665))]
#[case("aab,bcc->ac"                           , vec![2, 4]              , (  0.4283036047645465,   0.8328438191931101))]
#[case("baa,bcc->ac"                           , vec![2, 4]              , (  4.0702573007696854,  -4.8086334738516854))]
#[case("aab,ccb->ac"                           , vec![2, 4]              , (  0.2599457596087409,  -0.0972866664359448))]
// [TYPE] Randomly build test caes
#[case("aab,fa,df,ecc->bde"                    , vec![3, 5, 4]           , ( -0.6452535857516806,   0.2701805091737071))]
#[case("ecb,fef,bad,ed->ac"                    , vec![2, 4]              , ( -2.0991536924824938,  -0.6101631190896243))]
#[case("bcf,bbb,fbf,fc->"                      , vec![]                  , (  0.3689725389646942,  -0.4863879440986705))]
#[case("bb,ff,be->e"                           , vec![4]                 , (  0.1216715120397994,  -0.5219532921091159))]
#[case("bcb,bb,fc,fff->"                       , vec![]                  , (  0.0496840348532896,   0.0876400640479318))]
#[case("fbb,dfd,fc,fc->"                       , vec![]                  , ( -0.2019588423252376,   0.0520794854329894))]
#[case("afd,ba,cc,dc->bf"                      , vec![3, 3]              , ( -0.2164303183208181,   1.7498299453944053))]
#[case("adb,bc,fa,cfc->d"                      , vec![5]                 , (  0.4821755077243144,  -0.2166447771592562))]
#[case("bbd,bda,fc,db->acf"                    , vec![2, 4, 3]           , (  4.4800763340690981,   2.9909850932868034))]
#[case("dba,ead,cad->bce"                      , vec![3, 4, 4]           , (  1.2455232197722861,   0.6959454118311303))]
#[case("aef,fbc,dca->bde"                      , vec![3, 5, 4]           , (  2.6115342498937100,   0.1558997986008280))]
// [TYPE] Transpose only
#[case("ea->ea"                                , vec![4, 2]              , (  4.2779003835689338,  -1.0430003580983946))]
#[case("fb->fb"                                , vec![3, 3]              , (  4.3272474273084374,  -0.9808282730885702))]
#[case("abcd->dcab"                            , vec![5, 4, 2, 3]        , (  5.0109395102756977,  -0.9966085998460459))]
#[case("gc->cg"                                , vec![4, 2]              , (  0.2680112599856153,  -0.1269641467186604))]
#[case("hd->dh"                                , vec![5, 6]              , (  4.4819580717183714,  -3.3460245221870828))]
#[case("efgh->hfge"                            , vec![6, 3, 2, 4]        , (  2.2237855229147128,   3.8833637852830840))]
#[case("acdf->afcd"                            , vec![2, 3, 4, 5]        , (  3.2392076661644174,  -0.6296964164425971))]
#[case("gihb->ghib"                            , vec![2, 6, 5, 3]        , ( -2.1102327279825022,   0.4224646504787080))]
#[case("hfac->cfah"                            , vec![4, 3, 2, 6]        , (  2.2237855229146861,   3.8833637852830756))]
#[case("gfac->cgaf"                            , vec![4, 2, 2, 3]        , (  4.0858816458932772,  -0.5554982976507628))]
#[case("gifabc->abifcg"                        , vec![2, 3, 5, 3, 4, 2]  , ( -0.8209082897628119,   0.1283066691603100))]
#[case("hfac->cfha"                            , vec![4, 3, 6, 2]        , ( -1.3035489268162332,  -2.2966898260907733))]
fn test_einsum(#[case] einsum_str: &str, #[case] ref_shape: Vec<isize>, #[case] ref_fp: (f64, f64)) {
    let ref_fp = c64::new(ref_fp.0, ref_fp.1);
    let einsum_str_inp = einsum_str.split("->").next().unwrap().split(',').collect::<Vec<&str>>();
    let tensors = einsum_str_inp.iter().map(|s| build_tblis_tensor(s)).collect::<Vec<_>>();
    let tblis_tensors = tensors.iter().map(|(_, t)| t).collect::<Vec<_>>();
    let (out_data, out_tensor) = unsafe { tblis_einsum(einsum_str, &tblis_tensors, true, None, true, None).unwrap() };
    let out_fp = fp(&out_data);
    assert_eq!(out_tensor.shape, ref_shape);
    assert!((out_fp - ref_fp).abs() < 1e-10);
}
