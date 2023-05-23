#pragma once

#ifndef ZIMG_TEST_DYNAMIC_TYPE_H_
#define ZIMG_TEST_DYNAMIC_TYPE_H_

#ifndef GOOGLETEST_INCLUDE_GTEST_GTEST_H_
  #error gtest not included
#endif

#include <typeinfo>
#include <type_traits>

template <class T, class U>
bool assert_different_dynamic_type(const T *a, const U *b)
{
	static_assert(std::is_polymorphic_v<T>, "must be virtual");
	static_assert(std::is_polymorphic_v<U>, "must be virtual");

	const auto &tid_a = typeid(*a);
	const auto &tid_b = typeid(*b);

	if (tid_a == tid_b) {
		ADD_FAILURE() << "expected different types: " << tid_a.name() << " vs " << tid_b.name();
		return false;
	}
	return true;
}

#endif // ZIMG_TEST_DYNAMIC_TYPE_H_
