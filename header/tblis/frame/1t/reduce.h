#ifndef _TBLIS_IFACE_1T_REDUCE_H_
#define _TBLIS_IFACE_1T_REDUCE_H_

#include "../base/thread.h"
#include "../base/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT

void tblis_tensor_reduce(const tblis_comm* comm,
                         const tblis_config* cntx,
                         reduce_t op,
                         const tblis_tensor* A,
                         const label_type* idx_A,
                         tblis_scalar* result,
                         len_type* idx);

#if TBLIS_ENABLE_CPLUSPLUS

template <typename T=scalar>
struct reduce_result
{
    T value;
    len_type idx;

    reduce_result(type_t)
    : value(), idx() {}

    operator const T&() const { return value; }
};

template <>
struct reduce_result<scalar>
{
    scalar value;
    len_type idx;

    reduce_result(type_t type)
    : value(0.0, type), idx() {}

    operator const scalar&() const { return value; }
};

inline
void reduce(const communicator& comm,
            reduce_t op,
            const tensor_wrapper& A,
            const label_vector& idx_A,
            scalar& result,
            len_type& idx)
{
    tblis_tensor_reduce(comm, nullptr, op, &A, idx_A.data(), &result, &idx);
}

template <typename T>
void reduce(const communicator& comm,
            reduce_t op,
            const tensor_wrapper& A,
            const label_vector& idx_A,
            T& result,
            len_type& idx)
{
    scalar result_(0.0, A.type);
    reduce(comm, op, A, idx_A, result_, idx);
    result = result_.get<T>();
}

template <typename T=scalar>
reduce_result<T> reduce(const communicator& comm,
                        reduce_t op,
                        const tensor_wrapper& A,
                        const label_vector& idx_A)
{
    reduce_result<T> result(A.type);
    reduce(comm, op, A, idx_A, result.value, result.idx);
    return result;
}

inline
void reduce(const communicator& comm,
            reduce_t op,
            const tensor_wrapper& A,
            scalar& result,
            len_type& idx)
{
    reduce(comm, op, A, tblis::idx(A), result, idx);
}

template <typename T>
void reduce(const communicator& comm,
            reduce_t op,
            const tensor_wrapper& A,
            T& result,
            len_type& idx)
{
    scalar result_(0.0, A.type);
    reduce(comm, op, A, result_, idx);
    result = result_.get<T>();
}

template <typename T=scalar>
reduce_result<T> reduce(const communicator& comm,
                        reduce_t op,
                        const tensor_wrapper& A)
{
    reduce_result<T> result(A.type);
    reduce(comm, op, A, result.value, result.idx);
    return result;
}

inline
void reduce(reduce_t op,
            const tensor_wrapper& A,
            const label_vector& idx_A,
            scalar& result,
            len_type& idx)
{
    reduce(*(communicator*)nullptr, op, A, idx_A, result, idx);
}

template <typename T>
void reduce(reduce_t op,
            const tensor_wrapper& A,
            const label_vector& idx_A,
            T& result,
            len_type& idx)
{
    scalar result_(0.0, A.type);
    reduce(op, A, idx_A, result_, idx);
    result = result_.get<T>();
}

template <typename T=scalar>
reduce_result<T> reduce(reduce_t op,
                        const tensor_wrapper& A,
                        const label_vector& idx_A)
{
    reduce_result<T> result(A.type);
    reduce(op, A, idx_A, result.value, result.idx);
    return result;
}

inline
void reduce(reduce_t op,
            const tensor_wrapper& A,
            scalar& result,
            len_type& idx)
{
    reduce(op, A, tblis::idx(A), result, idx);
}

template <typename T>
void reduce(reduce_t op,
            const tensor_wrapper& A,
            T& result,
            len_type& idx)
{
    scalar result_(0.0, A.type);
    reduce(op, A, result_, idx);
    result = result_.get<T>();
}

template <typename T=scalar>
reduce_result<T> reduce(reduce_t op, const tensor_wrapper& A)
{
    reduce_result<T> result(A.type);
    reduce(op, A, result.value, result.idx);
    return result;
}

#ifdef MARRAY_DPD_MARRAY_HPP

template <typename T>
void reduce(const communicator& comm, reduce_t op, MArray::dpd_marray_view<const T> A,
            const label_vector& idx_A, T& result, len_type& idx);

template <typename T>
void reduce(reduce_t op, MArray::dpd_marray_view<const T> A, const label_vector& idx_A,
            T& result, len_type& idx)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            reduce(comm, op, A, idx_A, result, idx);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
reduce_result<T> reduce(reduce_t op, MArray::dpd_marray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result(type_tag<T>::value);
    reduce(op, A, idx_A, result.value, result.idx);
    return result;
}

template <typename T>
reduce_result<T> reduce(const communicator& comm, reduce_t op,
                        MArray::dpd_marray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result;
    reduce(comm, op, A, idx_A, result.value, result.idx);
    return result;
}

#endif

#ifdef MARRAY_INDEXED_MARRAY_HPP

template <typename T>
void reduce(const communicator& comm, reduce_t op, MArray::indexed_marray_view<const T> A,
            const label_vector& idx_A, T& result, len_type& idx);

template <typename T>
void reduce(reduce_t op, MArray::indexed_marray_view<const T> A, const label_vector& idx_A,
            T& result, len_type& idx)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            reduce(comm, op, A, idx_A, result, idx);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
reduce_result<T> reduce(reduce_t op, MArray::indexed_marray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result(type_tag<T>::value);
    reduce(op, A, idx_A, result.value, result.idx);
    return result;
}

template <typename T>
reduce_result<T> reduce(const communicator& comm, reduce_t op,
                        MArray::indexed_marray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result;
    reduce(comm, op, A, idx_A, result.value, result.idx);
    return result;
}

#endif

#ifdef MARRAY_INDEXED_DPD_MARRAY_HPP

template <typename T>
void reduce(const communicator& comm, reduce_t op, MArray::indexed_dpd_marray_view<const T> A,
            const label_vector& idx_A, T& result, len_type& idx);

template <typename T>
void reduce(reduce_t op, MArray::indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
            T& result, len_type& idx)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            reduce(comm, op, A, idx_A, result, idx);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
reduce_result<T> reduce(reduce_t op, MArray::indexed_dpd_marray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result(type_tag<T>::value);
    reduce(op, A, idx_A, result.value, result.idx);
    return result;
}

template <typename T>
reduce_result<T> reduce(const communicator& comm, reduce_t op,
                        MArray::indexed_dpd_marray_view<const T> A, const label_vector& idx_A)
{
    reduce_result<T> result;
    reduce(comm, op, A, idx_A, result.value, result.idx);
    return result;
}

#endif

namespace internal
{

template <typename Tensor>
struct data_type_helper
{
    static void check(...);

    static scalar check(scalar&);

    #ifdef MARRAY_MARRAY_BASE_HPP

    template <typename T, int N, typename D, bool O>
    static std::decay_t<T> check(MArray::marray_base<T,N,D,O>&);

    #endif

    #ifdef MARRAY_MARRAY_SLICE_HPP

    template <typename T, int N, int I, typename... D>
    static std::decay_t<T> check(MArray::marray_slice<T,N,I,D...>&);

    #endif

    #ifdef MARRAY_DPD_MARRAY_HPP

    template <typename T, typename D, bool O>
    static std::decay_t<T> check(MArray::dpd_marray_base<T,D,O>&);

    #endif

    #ifdef MARRAY_INDEXED_MARRAY_HPP

    template <typename T, typename D, bool O>
    static std::decay_t<T> check(MArray::indexed_marray_base<T,D,O>&);

    #endif

    #ifdef MARRAY_INDEXED_DPD_MARRAY_HPP

    template <typename T, typename D, bool O>
    static std::decay_t<T> check(MArray::indexed_dpd_marray_base<T,D,O>&);

    #endif

    #if defined(EIGEN_CXX11_TENSOR_TENSOR_FORWARD_DECLARATIONS_H)

    template <typename D, int A>
    static std::decay_t<Eigen::TensorBase<D,A>::Scalar> check(Eigen::TensorBase<D,A>&);

    #endif

    typedef decltype(check(std::declval<Tensor&>())) type;
};

template <typename T>
struct data_type_helper2 { typedef T type; };

template <>
struct data_type_helper2<void> {};

template <typename T>
using data_type = typename data_type_helper2<typename data_type_helper<std::decay_t<T>>::type>::type;

}

#define TBLIS_ALIAS_REDUCTION(name, op, which) \
\
template <typename Tensor, typename... Args> \
inline auto name(const Tensor& t, Args&&... args) \
-> decltype(reduce<internal::data_type<Tensor>>(op, t, std::forward<Args>(args)...)which) \
{ \
    return reduce<internal::data_type<Tensor>>(op, t, std::forward<Args>(args)...)which; \
} \
\
template <typename Tensor, typename... Args> \
inline auto name(const communicator& comm, const Tensor& t, Args&&... args) \
-> decltype(reduce<internal::data_type<Tensor>>(comm, op, t, std::forward<Args>(args)...)which) \
{ \
    return reduce<internal::data_type<Tensor>>(comm, op, t, std::forward<Args>(args)...)which; \
} \
\
template <typename... Args> \
inline auto name(const tensor_wrapper& t, Args&&... args) \
{ \
    return reduce(op, t, std::forward<Args>(args)...)which; \
} \
\
template <typename... Args> \
inline auto name(const communicator& comm, const tensor_wrapper& t, Args&&... args) \
{ \
    return reduce(comm, op, t, std::forward<Args>(args)...)which; \
}

TBLIS_ALIAS_REDUCTION(asum, REDUCE_SUM_ABS, .value)
TBLIS_ALIAS_REDUCTION(norm, REDUCE_NORM_2, .value)
TBLIS_ALIAS_REDUCTION(amaxv, REDUCE_MAX_ABS, .value)
TBLIS_ALIAS_REDUCTION(iamax, REDUCE_MAX_ABS, .idx)
TBLIS_ALIAS_REDUCTION(amax, REDUCE_MAX_ABS, )

#undef TBLIS_ALIAS_REDUCTION

#endif

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif
