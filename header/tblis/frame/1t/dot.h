#ifndef _TBLIS_IFACE_1T_DOT_H_
#define _TBLIS_IFACE_1T_DOT_H_

#include "../base/thread.h"
#include "../base/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT
void tblis_tensor_dot(const tblis_comm* comm,
                      const tblis_config* cntx,
                      const tblis_tensor* A,
                      const label_type* idx_A,
                      const tblis_tensor* B,
                      const label_type* idx_B,
                      tblis_scalar* result);

#if TBLIS_ENABLE_CPLUSPLUS

inline
void dot(const communicator& comm,
         const tensor_wrapper& A,
         const label_vector& idx_A,
         const tensor_wrapper& B,
         const label_vector& idx_B,
         tblis_scalar& result)
{
    tblis_tensor_dot(comm, nullptr, &A, idx_A.data(), &B, idx_B.data(), &result);
}

template <typename T>
void dot(const communicator& comm,
         const tensor_wrapper& A,
         const label_vector& idx_A,
         const tensor_wrapper& B,
         const label_vector& idx_B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(comm, A, idx_A, B, idx_B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const communicator& comm,
                 const tensor_wrapper& A,
                 const label_vector& idx_A,
                 const tensor_wrapper& B,
                 const label_vector& idx_B)
{
    tblis_scalar result(0.0, A.type);
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      const tensor_wrapper& A,
      const label_vector& idx_A,
      const tensor_wrapper& B,
      const label_vector& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

inline
void dot(const communicator& comm,
         const tensor_wrapper& A,
         const tensor_wrapper& B,
         tblis_scalar& result)
{
    dot(comm, A, idx(A), B, idx(B), result);
}

template <typename T>
void dot(const communicator& comm,
         const tensor_wrapper& A,
         const tensor_wrapper& B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(comm, A, B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const communicator& comm,
                 const tensor_wrapper& A,
                 const tensor_wrapper& B)
{
    tblis_scalar result(0.0, A.type);
    dot(comm, A, B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      const tensor_wrapper& A,
      const tensor_wrapper& B)
{
    T result;
    dot(comm, A, B, result);
    return result;
}

inline
void dot(const tensor_wrapper& A,
         const label_vector& idx_A,
         const tensor_wrapper& B,
         const label_vector& idx_B,
         tblis_scalar& result)
{
    dot(*(communicator*)nullptr, A, idx_A, B, idx_B, result);
}

template <typename T>
void dot(const tensor_wrapper& A,
         const label_vector& idx_A,
         const tensor_wrapper& B,
         const label_vector& idx_B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(A, idx_A, B, idx_B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const tensor_wrapper& A,
                 const label_vector& idx_A,
                 const tensor_wrapper& B,
                 const label_vector& idx_B)
{
    tblis_scalar result(0.0, A.type);
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const tensor_wrapper& A,
      const label_vector& idx_A,
      const tensor_wrapper& B,
      const label_vector& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

inline
void dot(const tensor_wrapper& A,
         const tensor_wrapper& B,
         tblis_scalar& result)
{
    dot(A, idx(A), B, idx(B), result);
}

template <typename T>
void dot(const tensor_wrapper& A,
         const tensor_wrapper& B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(A, B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const tensor_wrapper& A,
                 const tensor_wrapper& B)
{
    tblis_scalar result(0.0, A.type);
    dot(A, B, result);
    return result;
}

template <typename T>
T dot(const tensor_wrapper& A,
      const tensor_wrapper& B)
{
    T result;
    dot(A, B, result);
    return result;
}

#ifdef MARRAY_DPD_MARRAY_HPP

template <typename T>
void dot(const communicator& comm,
         MArray::dpd_marray_view<const T> A, const label_vector& idx_A,
         MArray::dpd_marray_view<const T> B, const label_vector& idx_B, T& result);

template <typename T>
void dot(MArray::dpd_marray_view<const T> A, const label_vector& idx_A,
         MArray::dpd_marray_view<const T> B, const label_vector& idx_B, T& result)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            dot(comm, A, idx_A, B, idx_B, result);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
T dot(MArray::dpd_marray_view<const T> A, const label_vector& idx_A,
      MArray::dpd_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      MArray::dpd_marray_view<const T> A, const label_vector& idx_A,
      MArray::dpd_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

#endif

#ifdef MARRAY_INDEXED_MARRAY_HPP

template <typename T>
void dot(const communicator& comm,
         MArray::indexed_marray_view<const T> A, const label_vector& idx_A,
         MArray::indexed_marray_view<const T> B, const label_vector& idx_B, T& result);

template <typename T>
void dot(MArray::indexed_marray_view<const T> A, const label_vector& idx_A,
         MArray::indexed_marray_view<const T> B, const label_vector& idx_B, T& result)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            dot(comm, A, idx_A, B, idx_B, result);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
T dot(MArray::indexed_marray_view<const T> A, const label_vector& idx_A,
      MArray::indexed_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      MArray::indexed_marray_view<const T> A, const label_vector& idx_A,
      MArray::indexed_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

#endif

#ifdef MARRAY_INDEXED_DPD_MARRAY_HPP

template <typename T>
void dot(const communicator& comm,
         MArray::indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
         MArray::indexed_dpd_marray_view<const T> B, const label_vector& idx_B, T& result);

template <typename T>
void dot(MArray::indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
         MArray::indexed_dpd_marray_view<const T> B, const label_vector& idx_B, T& result)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            dot(comm, A, idx_A, B, idx_B, result);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
T dot(MArray::indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
      MArray::indexed_dpd_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      MArray::indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
      MArray::indexed_dpd_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

#endif

#endif

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif
