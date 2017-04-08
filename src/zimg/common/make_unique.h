#pragma once

#ifndef ZIMG_MAKE_UNIQUE_H_
#define ZIMG_MAKE_UNIQUE_H_

#if __cplusplus >= 201402L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201402L)
#include <memory>
namespace ztd {
using std::make_unique;
} // namespace ztd
#else
#include <memory>
#include <type_traits>
#include <utility>

namespace ztd {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique_helper(std::false_type, Args&&... args) {
	return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T, typename... Args>
std::unique_ptr<T> make_unique_helper(std::true_type, Args&&... args) {
	static_assert(std::extent<T>::value == 0,
	              "make_unique<T[N]>() is forbidden, please use make_unique<T[]>().");

	typedef typename std::remove_extent<T>::type U;
	return std::unique_ptr<T>(new U[sizeof...(Args)]{ std::forward<Args>(args)... });
}

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
	return make_unique_helper<T>(std::is_array<T>(), std::forward<Args>(args)...);
}

} // namespace ztd
#endif // __cplusplus >= 201402L

#endif // ZIMG_MAKE_UNIQUE_H_
