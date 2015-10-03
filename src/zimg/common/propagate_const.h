#pragma once

#ifndef ZIMG_PROPAGATE_CONST_H_
#define ZIMG_PROPAGATE_CONST_H_

#include <type_traits>

namespace zimg {;

template <class T, class U, bool Value = std::is_const<T>::value>
struct propagate_const;

template <class T, class U>
struct propagate_const<T, U, true> {
	typedef typename std::add_const<U>::type type;
};

template <class T, class U>
struct propagate_const<T, U, false> {
	typedef typename std::remove_const<U>::type type;
};

} // namespace zimg

#endif // ZIMG_PROPAGATE_CONST_H
