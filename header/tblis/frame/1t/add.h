#ifndef _TBLIS_IFACE_1T_ADD_H_
#define _TBLIS_IFACE_1T_ADD_H_

#include "../base/thread.h"
#include "../base/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT
void tblis_tensor_add(const tblis_comm* comm,
                      const tblis_config* cntx,
                      const tblis_tensor* A,
                      const label_type* idx_A,
                            tblis_tensor* B,
                      const label_type* idx_B);

#if TBLIS_ENABLE_CPLUSPLUS

inline
void add(const communicator& comm,
         const scalar& alpha,
         const tensor_wrapper& A_,
         const label_vector& idx_A,
         const scalar& beta,
               tensor_wrapper&& B,
         const label_vector& idx_B)
{
    auto A(A_);
    A.scalar *= alpha.convert(A.type);
    B.scalar *= beta.convert(B.type);
    tblis_tensor_add(comm, nullptr, &A, idx_A.data(), &B, idx_B.data());
}

inline
void add(const communicator& comm,
         const scalar& alpha,
         const tensor_wrapper& A,
         const label_vector& idx_A,
               tensor_wrapper&& B,
         const label_vector& idx_B)
{
    add(comm, alpha, A, idx_A, {0.0, A.type}, std::move(B), idx_B);
}

inline
void add(const communicator& comm,
         const tensor_wrapper& A,
         const label_vector& idx_A,
         const scalar& beta,
               tensor_wrapper&& B,
         const label_vector& idx_B)
{
    add(comm, {1.0, A.type}, A, idx_A, beta, std::move(B), idx_B);
}

inline
void add(const communicator& comm,
         const tensor_wrapper& A,
         const label_vector& idx_A,
               tensor_wrapper&& B,
         const label_vector& idx_B)
{
    add(comm, {1.0, A.type}, A, idx_A, {0.0, A.type}, std::move(B), idx_B);
}

inline
void add(const communicator& comm,
         const scalar& alpha,
         const tensor_wrapper& A,
         const scalar& beta,
               tensor_wrapper&& B)
{
    add(comm, alpha, A, idx(A), beta, std::move(B), idx(B));
}

inline
void add(const communicator& comm,
         const scalar& alpha,
         const tensor_wrapper& A,
               tensor_wrapper&& B)
{
    add(comm, alpha, A, {0.0, A.type}, std::move(B));
}

inline
void add(const communicator& comm,
         const tensor_wrapper& A,
         const scalar& beta,
               tensor_wrapper&& B)
{
    add(comm, {1.0, A.type}, A, beta, std::move(B));
}

inline
void add(const communicator& comm,
         const tensor_wrapper& A,
               tensor_wrapper&& B)
{
    add(comm, {1.0, A.type}, A, {0.0, A.type}, std::move(B));
}

TBLIS_COMPAT_INLINE
void add(const scalar& alpha,
         const tensor_wrapper& A,
         const label_vector& idx_A,
         const scalar& beta,
               tensor_wrapper&& B,
         const label_vector& idx_B)
{
    add(*(communicator*)nullptr, alpha, A, idx_A, beta, std::move(B), idx_B);
}

inline
void add(const scalar& alpha,
         const tensor_wrapper& A,
         const label_vector& idx_A,
               tensor_wrapper&& B,
         const label_vector& idx_B)
{
    add(alpha, A, idx_A, {0.0, A.type}, std::move(B), idx_B);
}

inline
void add(const tensor_wrapper& A,
         const label_vector& idx_A,
         const scalar& beta,
               tensor_wrapper&& B,
         const label_vector& idx_B)
{
    add({1.0, A.type}, A, idx_A, beta, std::move(B), idx_B);
}

inline
void add(const tensor_wrapper& A,
         const label_vector& idx_A,
               tensor_wrapper&& B,
         const label_vector& idx_B)
{
    add({1.0, A.type}, A, idx_A, {0.0, A.type}, std::move(B), idx_B);
}

inline
void add(const scalar& alpha,
         const tensor_wrapper& A,
         const scalar& beta,
               tensor_wrapper&& B)
{
    add(alpha, A, idx(A), beta, std::move(B), idx(B));
}

inline
void add(const scalar& alpha,
         const tensor_wrapper& A,
               tensor_wrapper&& B)
{
    add(alpha, A, {0.0, A.type}, std::move(B));
}

inline
void add(const tensor_wrapper& A,
         const scalar& beta,
               tensor_wrapper&& B)
{
    add({1.0, A.type}, A, beta, std::move(B));
}

inline
void add(const tensor_wrapper& A,
               tensor_wrapper&& B)
{
    add({1.0, A.type}, A, {0.0, A.type}, std::move(B));
}

#ifdef MARRAY_DPD_MARRAY_HPP

template <typename T>
void add(const communicator& comm,
         T alpha, MArray::dpd_marray_view<const T> A, const label_vector& idx_A,
         T  beta, MArray::dpd_marray_view<      T> B, const label_vector& idx_B);

template <typename T>
void add(T alpha, MArray::dpd_marray_view<const T> A, const label_vector& idx_A,
         T  beta, MArray::dpd_marray_view<      T> B, const label_vector& idx_B)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            add(comm, alpha, A, idx_A, beta, B, idx_B);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef MARRAY_INDEXED_MARRAY_HPP

template <typename T>
void add(const communicator& comm,
         T alpha, MArray::indexed_marray_view<const T> A, const label_vector& idx_A,
         T  beta, MArray::indexed_marray_view<      T> B, const label_vector& idx_B);

template <typename T>
void add(T alpha, MArray::indexed_marray_view<const T> A, const label_vector& idx_A,
         T  beta, MArray::indexed_marray_view<      T> B, const label_vector& idx_B)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            add(comm, alpha, A, idx_A, beta, B, idx_B);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef MARRAY_INDEXED_DPD_MARRAY_HPP

template <typename T>
void add(const communicator& comm,
         T alpha, MArray::indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
         T  beta, MArray::indexed_dpd_marray_view<      T> B, const label_vector& idx_B);

template <typename T>
void add(T alpha, MArray::indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
         T  beta, MArray::indexed_dpd_marray_view<      T> B, const label_vector& idx_B)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            add(comm, alpha, A, idx_A, beta, B, idx_B);
        },
        tblis_get_num_threads()
    );
}

#endif

#endif

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif
