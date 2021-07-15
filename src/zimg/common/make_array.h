#pragma once

#ifndef ZIMG_MAKE_ARRAY_H_
#define ZIMG_MAKE_ARRAY_H_

#include <array>
#include <type_traits>

namespace zimg {

template<typename... T>
std::array<typename std::common_type<T...>::type, sizeof...(T)>
constexpr make_array(T &&...t)
{
    return { std::forward<T>(t)... };
}

} // namespace zimg

#endif // ZIMG_MAKE_ARRAY_H_
