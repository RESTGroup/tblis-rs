#ifndef _TBLIS_IFACE_1T_SCALE_H_
#define _TBLIS_IFACE_1T_SCALE_H_

#include "../base/thread.h"
#include "../base/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT

void tblis_tensor_scale(const tblis_comm* comm,
                        const tblis_config* cntx,
                              tblis_tensor* A,
                        const label_type* idx_A);

#if TBLIS_ENABLE_CPLUSPLUS

inline
void scale(const communicator& comm,
           const scalar& alpha,
                 tensor_wrapper&& A,
           const label_vector& idx_A)
{
    A.scalar *= alpha.convert(A.type);
    tblis_tensor_scale(comm, nullptr, &A, idx_A.data());
}

inline
void scale(const communicator& comm,
                 tensor_wrapper&& A,
           const label_vector& idx_A)
{
    scale(comm, {1.0, A.type}, std::move(A), idx_A);
}

inline
void scale(const communicator& comm,
           const scalar& alpha,
                 tensor_wrapper&& A)
{
    scale(comm, alpha, std::move(A), idx(A));
}

inline
void scale(const communicator& comm,
                 tensor_wrapper&& A)
{
    scale(comm, {1.0, A.type}, std::move(A));
}

TBLIS_COMPAT_INLINE
void scale(const scalar& alpha,
                 tensor_wrapper&& A,
           const label_vector& idx_A)
{
    scale(*(communicator*)nullptr, alpha, std::move(A), idx_A);
}

inline
void scale(      tensor_wrapper&& A,
           const label_vector& idx_A)
{
    scale({1.0, A.type}, std::move(A), idx_A);
}

inline
void scale(const scalar& alpha,
                 tensor_wrapper&& A)
{
    scale(alpha, std::move(A), idx(A));
}

inline
void scale(      tensor_wrapper&& A)
{
    scale({1.0, A.type}, std::move(A));
}

#ifdef MARRAY_DPD_MARRAY_HPP

template <typename T>
void scale(const communicator& comm,
           T alpha, MArray::dpd_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void scale(T alpha, MArray::dpd_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            scale(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef MARRAY_INDEXED_MARRAY_HPP

template <typename T>
void scale(const communicator& comm,
           T alpha, MArray::indexed_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void scale(T alpha, MArray::indexed_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            scale(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef MARRAY_INDEXED_DPD_MARRAY_HPP

template <typename T>
void scale(const communicator& comm,
           T alpha, MArray::indexed_dpd_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void scale(T alpha, MArray::indexed_dpd_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            scale(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#endif

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif
