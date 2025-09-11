#ifndef _TBLIS_IFACE_1T_SET_H_
#define _TBLIS_IFACE_1T_SET_H_

#include "../base/thread.h"
#include "../base/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT
void tblis_tensor_set(const tblis_comm* comm,
                      const tblis_config* cntx,
                      const tblis_scalar* alpha,
                            tblis_tensor* A,
                      const label_type* idx_A);

#if TBLIS_ENABLE_CPLUSPLUS

inline
void set(const communicator& comm,
         const scalar& alpha_,
               tensor_wrapper&& A,
         const label_vector& idx_A)
{
    auto alpha = alpha_.convert(A.type);
    tblis_tensor_set(comm, nullptr, &alpha, &A, idx_A.data());
}

inline
void set(const communicator& comm,
         const scalar& alpha,
               tensor_wrapper&& A)
{
    set(comm, alpha, std::move(A), idx(A));
}

TBLIS_COMPAT_INLINE
void set(const scalar& alpha,
               tensor_wrapper&& A,
         const label_vector& idx_A)
{
    set(*(communicator*)nullptr, alpha, std::move(A), idx_A);
}

inline
void set(const scalar& alpha,
               tensor_wrapper&& A)
{
    set(alpha, std::move(A), idx(A));
}

#ifdef MARRAY_DPD_MARRAY_HPP

template <typename T>
void set(const communicator& comm,
         T alpha, MArray::dpd_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void set(T alpha, MArray::dpd_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            set(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef MARRAY_INDEXED_MARRAY_HPP

template <typename T>
void set(const communicator& comm,
         T alpha, MArray::indexed_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void set(T alpha, MArray::indexed_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            set(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef MARRAY_INDEXED_DPD_MARRAY_HPP

template <typename T>
void set(const communicator& comm,
         T alpha, MArray::indexed_dpd_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void set(T alpha, MArray::indexed_dpd_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            set(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#endif

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif
