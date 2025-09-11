#ifndef _TBLIS_IFACE_3T_MULT_H_
#define _TBLIS_IFACE_3T_MULT_H_

#include "../base/thread.h"
#include "../base/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT
void tblis_tensor_mult(const tblis_comm* comm, const tblis_config* cntx,
                       const tblis_tensor* A, const label_type* idx_A,
                       const tblis_tensor* B, const label_type* idx_B,
                             tblis_tensor* C, const label_type* idx_C);

#if TBLIS_ENABLE_CPLUSPLUS

inline
void mult(const communicator& comm,
          const scalar& alpha,
          const tensor_wrapper& A,
          const label_vector& idx_A,
          const tensor_wrapper& B,
          const label_vector& idx_B,
          const scalar& beta,
          const tensor_wrapper& C,
          const label_vector& idx_C)
{
    auto A_(A);
    A_.scalar *= alpha;

    auto C_(C);
    C_.scalar *= beta;

    tblis_tensor_mult(comm, nullptr, &A_, idx_A.data(), &B, idx_B.data(), &C_, idx_C.data());
}

inline
void mult(const communicator& comm,
          const tensor_wrapper& A,
          const label_vector& idx_A,
          const tensor_wrapper& B,
          const label_vector& idx_B,
          const scalar& beta,
          const tensor_wrapper& C,
          const label_vector& idx_C)
{
    mult(comm, {1.0, A.type}, A, idx_A, B, idx_B, beta, C, idx_C);
}

inline
void mult(const communicator& comm,
          const scalar& alpha,
          const tensor_wrapper& A,
          const label_vector& idx_A,
          const tensor_wrapper& B,
          const label_vector& idx_B,
          const tensor_wrapper& C,
          const label_vector& idx_C)
{
    mult(comm, alpha, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

inline
void mult(const communicator& comm,
          const tensor_wrapper& A,
          const label_vector& idx_A,
          const tensor_wrapper& B,
          const label_vector& idx_B,
          const tensor_wrapper& C,
          const label_vector& idx_C)
{
    mult(comm, {1.0, A.type}, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

inline
void mult(const communicator& comm,
          const scalar& alpha,
          const tensor_wrapper& A,
          const tensor_wrapper& B,
          const scalar& beta,
          const tensor_wrapper& C)
{
    label_vector idx_A, idx_B, idx_C;

    TBLIS_ASSERT((A.ndim+B.ndim+C.ndim)%2 == 0);

    auto nAB = (A.ndim+B.ndim-C.ndim)/2;
    auto nAC = (A.ndim+C.ndim-B.ndim)/2;
    auto nBC = (B.ndim+C.ndim-A.ndim)/2;

    for (auto i : range(nAC)) idx_A.push_back(i);
    for (auto i : range(nAC)) idx_C.push_back(i);
    for (auto i : range(nAB)) idx_A.push_back(nAC+i);
    for (auto i : range(nAB)) idx_B.push_back(nAC+i);
    for (auto i : range(nBC)) idx_B.push_back(nAC+nAB+i);
    for (auto i : range(nBC)) idx_C.push_back(nAC+nAB+i);

    mult(comm, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
}

inline
void mult(const communicator& comm,
          const tensor_wrapper& A,
          const tensor_wrapper& B,
          const scalar& beta,
          const tensor_wrapper& C)
{
    mult(comm, {1.0, A.type}, A, B, beta, C);
}

inline
void mult(const communicator& comm,
          const scalar& alpha,
          const tensor_wrapper& A,
          const tensor_wrapper& B,
          const tensor_wrapper& C)
{
    mult(comm, alpha, A, B, {0.0, A.type}, C);
}

inline
void mult(const communicator& comm,
          const tensor_wrapper& A,
          const tensor_wrapper& B,
          const tensor_wrapper& C)
{
    mult(comm, {1.0, A.type}, A, B, {0.0, A.type}, C);
}

TBLIS_COMPAT_INLINE
void mult(const scalar& alpha,
          const tensor_wrapper& A,
          const label_vector& idx_A,
          const tensor_wrapper& B,
          const label_vector& idx_B,
          const scalar& beta,
          const tensor_wrapper& C,
          const label_vector& idx_C)
{
    mult(*(communicator*)nullptr, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
}

inline
void mult(const tensor_wrapper& A,
          const label_vector& idx_A,
          const tensor_wrapper& B,
          const label_vector& idx_B,
          const scalar& beta,
          const tensor_wrapper& C,
          const label_vector& idx_C)
{
    mult({1.0, A.type}, A, idx_A, B, idx_B, beta, C, idx_C);
}

inline
void mult(const scalar& alpha,
          const tensor_wrapper& A,
          const label_vector& idx_A,
          const tensor_wrapper& B,
          const label_vector& idx_B,
          const tensor_wrapper& C,
          const label_vector& idx_C)
{
    mult(alpha, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

inline
void mult(const tensor_wrapper& A,
          const label_vector& idx_A,
          const tensor_wrapper& B,
          const label_vector& idx_B,
          const tensor_wrapper& C,
          const label_vector& idx_C)
{
    mult({1.0, A.type}, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

inline
void mult(const scalar& alpha,
          const tensor_wrapper& A,
          const tensor_wrapper& B,
          const scalar& beta,
          const tensor_wrapper& C)
{
    mult(*(communicator*)nullptr, alpha, A, B, beta, C);
}

inline
void mult(const tensor_wrapper& A,
          const tensor_wrapper& B,
          const scalar& beta,
          const tensor_wrapper& C)
{
    mult({1.0, A.type}, A, B, beta, C);
}

inline
void mult(const scalar& alpha,
          const tensor_wrapper& A,
          const tensor_wrapper& B,
          const tensor_wrapper& C)
{
    mult(alpha, A, B, {0.0, A.type}, C);
}

inline
void mult(const tensor_wrapper& A,
          const tensor_wrapper& B,
          const tensor_wrapper& C)
{
    mult({1.0, A.type}, A, B, {0.0, A.type}, C);
}

#ifdef MARRAY_DPD_MARRAY_HPP

template <typename T>
void mult(const communicator& comm,
          T alpha, const MArray::dpd_marray_view<const T>& A, const label_vector& idx_A,
                   const MArray::dpd_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const MArray::dpd_marray_view<      T>& C, const label_vector& idx_C);

template <typename T>
void mult(T alpha, const MArray::dpd_marray_view<const T>& A, const label_vector& idx_A,
                   const MArray::dpd_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const MArray::dpd_marray_view<      T>& C, const label_vector& idx_C)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            mult(comm, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef MARRAY_INDEXED_MARRAY_HPP

template <typename T>
void mult(const communicator& comm,
          T alpha, const MArray::indexed_marray_view<const T>& A, const label_vector& idx_A,
                   const MArray::indexed_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const MArray::indexed_marray_view<      T>& C, const label_vector& idx_C);

template <typename T>
void mult(T alpha, const MArray::indexed_marray_view<const T>& A, const label_vector& idx_A,
                   const MArray::indexed_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const MArray::indexed_marray_view<      T>& C, const label_vector& idx_C)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            mult(comm, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef MARRAY_INDEXED_DPD_MARRAY_HPP

template <typename T>
void mult(const communicator& comm,
          T alpha, const MArray::indexed_dpd_marray_view<const T>& A, const label_vector& idx_A,
                   const MArray::indexed_dpd_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const MArray::indexed_dpd_marray_view<      T>& C, const label_vector& idx_C);

template <typename T>
void mult(T alpha, const MArray::indexed_dpd_marray_view<const T>& A, const label_vector& idx_A,
                   const MArray::indexed_dpd_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const MArray::indexed_dpd_marray_view<      T>& C, const label_vector& idx_C)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            mult(comm, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
        },
        tblis_get_num_threads()
    );
}

#endif

#endif

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif
