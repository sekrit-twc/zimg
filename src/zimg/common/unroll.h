#pragma once

#ifndef ZIMG_UNROLL_H_
#define ZIMG_UNROLL_H_

#include <cstddef>
#include <type_traits>
#include <utility>
#include "ccdep.h"

namespace zimg {

namespace detail {

template <class T, size_t ...Idx>
inline FORCE_INLINE void do_unroll(T f, std::index_sequence<Idx...>)
{
	(f(std::integral_constant<size_t, Idx>{}), ...);
}

} // namespace detail


template <size_t N, class T>
inline FORCE_INLINE void unroll(T f)
{
	detail::do_unroll(f, std::make_index_sequence<N>{});
}

#define ZIMG_UNROLL_FUNC(var) [&](auto var)

} // namespace zimg

#endif // ZIMG_UNROLL_H_
