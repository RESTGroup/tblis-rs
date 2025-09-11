#ifndef _TBLIS_IFACE_1T_SHIFT_H_
#define _TBLIS_IFACE_1T_SHIFT_H_

#include "../base/thread.h"
#include "../base/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT
void tblis_tensor_shift(const tblis_comm* comm,
                        const tblis_config* cntx,
                        const tblis_scalar* alpha,
                              tblis_tensor* A,
                        const label_type* idx_A);

#if TBLIS_ENABLE_CPLUSPLUS

inline
void shift(const communicator& comm,
           const scalar& alpha_,
           const scalar& beta,
                 tensor_wrapper&& A,
           const label_vector& idx_A)
{
    auto alpha = alpha_.convert(A.type);
    A.scalar *= beta.convert(A.type);
    tblis_tensor_shift(comm, nullptr, &alpha, &A, idx_A.data());
}

inline
void shift(const communicator& comm,
           const scalar& alpha,
                 tensor_wrapper&& A,
           const label_vector& idx_A)
{
    shift(comm, alpha, {1.0, A.type}, std::move(A), idx_A);
}

inline
void shift(const communicator& comm,
           const scalar& alpha,
           const scalar& beta,
                 tensor_wrapper&& A)
{
    shift(comm, alpha, beta, std::move(A), idx(A));
}

inline
void shift(const communicator& comm,
           const scalar& alpha,
                 tensor_wrapper&& A)
{
    shift(comm, alpha, {1.0, A.type}, std::move(A));
}

TBLIS_COMPAT_INLINE
void shift(const scalar& alpha,
           const scalar& beta,
                 tensor_wrapper&& A,
           const label_vector& idx_A)
{
    shift(*(communicator*)nullptr, alpha, beta, std::move(A), idx_A);
}

inline
void shift(const scalar& alpha,
                 tensor_wrapper&& A,
           const label_vector& idx_A)
{
    shift(alpha, {1.0, A.type}, std::move(A), idx_A);
}

inline
void shift(const scalar& alpha,
           const scalar& beta,
                 tensor_wrapper&& A)
{
    shift(alpha, beta, std::move(A), idx(A));
}

inline
void shift(const scalar& alpha,
                 tensor_wrapper&& A)
{
    shift(alpha, {1.0, A.type}, std::move(A));
}

#ifdef MARRAY_DPD_MARRAY_HPP

template <typename T>
void shift(const communicator& comm,
         T alpha, T beta, MArray::dpd_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void shift(T alpha, T beta, MArray::dpd_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            shift(comm, alpha, beta, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef MARRAY_INDEXED_MARRAY_HPP

template <typename T>
void shift(const communicator& comm,
         T alpha, T beta, MArray::indexed_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void shift(T alpha, T beta, MArray::indexed_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            shift(comm, alpha, beta, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef MARRAY_INDEXED_DPD_MARRAY_HPP

template <typename T>
void shift(const communicator& comm,
         T alpha, T beta, MArray::indexed_dpd_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void shift(T alpha, T beta, MArray::indexed_dpd_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            shift(comm, alpha, beta, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#endif

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif
