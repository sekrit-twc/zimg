#pragma once

#ifndef ZIMG_DEPTH_QUANTIZE_H_
#define ZIMG_DEPTH_QUANTIZE_H_

#include <algorithm>
#include <cstdint>
#include "common/pixel.h"

namespace zimg {
namespace depth {

template <class T>
T identity(T x)
{
	return x;
}

template <class T, class U>
T bit_cast(const U &x)
{
	static_assert(sizeof(T) == sizeof(U), "object sizes must match");

	T ret;
	std::copy_n(reinterpret_cast<const char *>(&x), sizeof(x), reinterpret_cast<char *>(&ret));
	return ret;
}

inline int32_t numeric_max(int bits)
{
	return (1L << bits) - 1;
}

inline int32_t integer_offset(const PixelFormat &format)
{
	if (pixel_is_float(format.type))
		return 0;
	else if (format.chroma)
		return 1L << (format.depth - 1);
	else if (!format.fullrange)
		return 16L << (format.depth - 8);
	else
		return 0;
}

inline int32_t integer_range(const PixelFormat &format)
{
	if (pixel_is_float(format.type))
		return 1;
	else if (format.fullrange)
		return numeric_max(format.depth);
	else if (format.chroma && !format.ycgco)
		return 224L << (format.depth - 8);
	else
		return 219L << (format.depth - 8);
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
