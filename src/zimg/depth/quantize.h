#pragma once

#ifndef ZIMG_DEPTH_QUANTIZE_H_
#define ZIMG_DEPTH_QUANTIZE_H_

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include "common/pixel.h"

namespace zimg {
namespace depth {

template <class T, class U>
T bit_cast(const U &x) noexcept
{
	static_assert(sizeof(T) == sizeof(U), "object sizes must match");
	static_assert(std::is_pod<T>::value && std::is_pod<U>::value, "object types must be POD");

	T ret;
	std::copy_n(reinterpret_cast<const char *>(&x), sizeof(x), reinterpret_cast<char *>(&ret));
	return ret;
}

constexpr int32_t numeric_max(int bits) noexcept
{
	return (1L << bits) - 1;
}

constexpr int32_t integer_offset(const PixelFormat &format) noexcept
{
	return pixel_is_float(format.type) ? 0
		: format.chroma ? 1L << (format.depth - 1)
		: !format.fullrange ? 16L << (format.depth - 8)
		: 0;
}

constexpr int32_t integer_range(const PixelFormat &format) noexcept
{
	return pixel_is_float(format.type) ? 1
		: format.fullrange ? numeric_max(format.depth)
		: format.chroma && !format.ycgco ? 224L << (format.depth - 8)
		: 219L << (format.depth - 8);
}

float half_to_float(uint16_t f16w) noexcept;
uint16_t float_to_half(float f32) noexcept;

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_QUANTIZE_H_
