#pragma once

#ifndef ZIMG_DEPTH_QUANTIZE_H_
#define ZIMG_DEPTH_QUANTIZE_H_

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include "common/pixel.h"

namespace zimg {
namespace depth {

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

inline std::pair<float, float> get_scale_offset(const PixelFormat &pixel_in, const PixelFormat &pixel_out)
{
	double range_in = integer_range(pixel_in);
	double offset_in = integer_offset(pixel_in);
	double range_out = integer_range(pixel_out);
	double offset_out = integer_offset(pixel_out);

	float scale = static_cast<float>(range_out / range_in);
	float offset = static_cast<float>(-offset_in * range_out / range_in + offset_out);

	return{ scale, offset };
}

float half_to_float(uint16_t f16w) noexcept;

uint16_t float_to_half(float f32) noexcept;

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_QUANTIZE_H_
