# # Test reference generation

# ## Source Info

# Contains code from opt_einsum, which is licensed under the MIT License.
#
# Contains code from pytblis, which is licensed under the BSD-3 License.

# ## Preparation

import numpy as np


def gen_array(shape, dtype):
    size = np.prod(shape)
    arr = np.cos(np.arange(size, dtype=dtype) + 0.2)
    if dtype in [np.complex64, np.complex128]:
        arr += - 0.5j * np.sin(np.arange(size, dtype=dtype) + 0.4)
    return arr.reshape(shape)


def fp(a):
    return np.dot(np.cos(np.arange(a.size)), a.ravel())


sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4, 9, 10, 2, 4, 5, 3, 2, 6]
valid_chars = "abcdefghijklmnopqABC"
dim_dict = dict(zip(valid_chars, sizes))


def build_shapes(string, dimension_dict=dim_dict):
    shapes = []
    string = string.replace(" ", "")
    terms = string.split("->")[0].split(",")
    for term in terms:
        dims = [dimension_dict[x] for x in term]
        shapes.append(tuple(dims))
    return tuple(shapes)


def build_views(string, dtype, dimension_dict=dim_dict):
    views = []
    for shape in build_shapes(string, dimension_dict=dimension_dict):
        if shape:
            arr = gen_array(shape, dtype)
            views.append(arr)
        else:
            val = gen_array([], dtype)
            views.append(val)
    return tuple(views)


# ## Test data

single_array_tests = ["ea", "fb", "abcd", "gc", "hd", "efgh", "acdf", "gihb", "hfac", "gfac", "gifabc", "hfac"]

mult_array_tests = {
    "scalar-like operations": [
        "a,->a",
        "ab,->ab",
        ",ab,->ab",
        ",,->",
    ],
    "hadamard-like products": [
        "a,ab,abc->abc",
        "a,b,ab->ab",
    ],
    "index-transformations": [
        "ea,fb,gc,hd,abcd->efgh",
        "ea,fb,abcd,gc,hd->efgh",
        "abcd,ea,fb,gc,hd->efgh",
    ],
    "complex contractions": [
        "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
        "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
        "cd,bdhe,aidb,hgca,gc,hgibcd,hgac",
        "abhe,hidj,jgba,hiab,gab",
        "bde,cdh,agdb,hica,ibd,hgicd,hiac",
        "chd,bde,agbc,hiad,hgc,hgi,hiad",
        "chd,bde,agbc,hiad,bdi,cgh,agdb",
        "bdhe,acad,hiab,agac,hibd",
    ],
    "collapse": [
        "ab,ab,c->",
        "ab,ab,c->c",
        "ab,ab,cd,cd->",
        "ab,ab,cd,cd->ac",
        "ab,ab,cd,cd->cd",
        "ab,ab,cd,cd,ef,ef->",        
    ],
    "outer products": [
        "ab,cd,ef->abcdef",
        "ab,cd,ef->acdf",
        "ab,cd,de->abcde",
        "ab,cd,de->be",
        "ab,bcd,cd->abcd",
        "ab,bcd,cd->abd",  
    ],
    "cases that have previously failed": [
        "eb,cb,fb->cef",
        "dd,fb,be,cdb->cef",
        "bca,cdb,dbf,afc->",
        "dcc,fce,ea,dbf->ab",
        "fdf,cdd,ccd,afe->ae",
        "abcd,ad",
        "ed,fcd,ff,bcf->be",
        "baa,dcf,af,cde->be",
        "bd,db,eac->ace",
        "fff,fae,bef,def->abd",
        "efc,dbc,acf,fd->abe",  
    ],
    "Inner products": [
        "ab,ab",
        "ab,ba",
        "abc,abc",
        "abc,bac",
        "abc,cba",
    ],
    "GEMM test cases": [
        "ab,bc",
        "ab,cb",
        "ba,bc",
        "ba,cb",
        "abcd,cd",
        "abcd,ab",
        "abcd,cdef",
        "abcd,cdef->feba",
        "abcd,efdc",  
    ],
    "Inner than dot": [
        "aab,bc->ac",
        "ab,bcc->ac",
        "aab,bcc->ac",
        "baa,bcc->ac",
        "aab,ccb->ac",  
    ],
    "Randomly build test caes": [
        "aab,fa,df,ecc->bde",
        "ecb,fef,bad,ed->ac",
        "bcf,bbb,fbf,fc->",
        "bb,ff,be->e",
        "bcb,bb,fc,fff->",
        "fbb,dfd,fc,fc->",
        "afd,ba,cc,dc->bf",
        "adb,bc,fa,cfc->d",
        "bbd,bda,fc,db->acf",
        "dba,ead,cad->bce",
        "aef,fbc,dca->bde",  
    ],
}


# ## Ref: einsum

def test_einsum(string, dtype):
    views = build_views(string, dtype=dtype)
    result = np.einsum(string, *views)
    return result.shape, fp(result)


for test_type, test_cases in mult_array_tests.items():
    print(f"// [TYPE] {test_type}")
    for case in test_cases:
        shape, fp_val = test_einsum(case, np.float64)
        print(f"#[case({'"' + case + '"':40}, vec!{str(list(shape)):20}, {fp_val:20.16f})]")

for test_type, test_cases in mult_array_tests.items():
    print(f"// [TYPE] {test_type}")
    for case in test_cases:
        shape, fp_val = test_einsum(case, np.complex128)
        print(f"#[case({'"' + case + '"':40}, vec!{str(list(shape)):20}, ({fp_val.real:20.16f}, {fp_val.imag:20.16f}))]")


# ## Ref: transpose

def test_transpose(string, dtype, rng_val=0):
    views = build_views(string, dtype=dtype)
    arr = views[0]
    rng = np.random.default_rng(rng_val)
    perm = rng.permutation(len(arr.shape))
    string_perm = "".join(np.array(list(string))[perm])
    command_string = f"{string}->{string_perm}"
    result = np.einsum(command_string, *views)
    return command_string, result.shape, fp(result)


for count, token in enumerate(single_array_tests):
    case, shape, fp_val = test_transpose(token, np.float64, count)
    print(f"#[case({'"' + case + '"':40}, vec!{str(list(shape)):20}, {fp_val:20.16f})]")

for count, token in enumerate(single_array_tests):
    case, shape, fp_val = test_transpose(token, np.complex128, count)
    print(f"#[case({'"' + case + '"':40}, vec!{str(list(shape)):20}, ({fp_val.real:20.16f}, {fp_val.imag:20.16f}))]")
