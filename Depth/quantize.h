#pragma once

#ifndef ZIMG_DEPTH_QUANTIZE_H_
#define ZIMG_DEPTH_QUANTIZE_H_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include "Common/pixel.h"

namespace zimg {;
namespace depth {;

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

	std::memcpy(&ret, &x, sizeof(ret));
	return ret;
}

template <class T>
T clamp(T x, T low, T high)
{
	return std::min(std::max(x, low), high);
}

inline int32_t numeric_max(int bits)
{
	return (1L << bits) - 1;
}

inline int32_t integer_offset(int bits, bool fullrange, bool chroma)
{
	if (chroma)
		return 1L << (bits - 1);
	else if (!fullrange)
		return 16L << (bits - 8);
	else
		return 0;
}

inline int32_t integer_range(int bits, bool fullrange, bool chroma)
{
	if (!fullrange && chroma)
		return 224L << (bits - 8);
	else if (!fullrange)
		return 219L << (bits - 8);
	else
		return numeric_max(bits);
}


/**
 * Half precision conversion routines adapted from public domain code:
 * https://gist.github.com/rygorous/2156668
 *
 * Thanks to Fabian "ryg" Giesen.
 */
inline float half_to_float(uint16_t x)
{
	float magic = bit_cast<float>((uint32_t)113 << 23);
	uint32_t shift_exp = 0x7C00UL << 13;
	uint32_t exp;
	uint32_t ret;

	ret = ((uint32_t)x & 0x7FFF) << 13;
	exp = shift_exp & ret;
	ret += (127UL - 15UL) << 23;

	if (exp == shift_exp) {
		ret += (127UL - 15UL) << 23;
	} else if (!exp) {
		ret += 1UL << 23;
		ret = bit_cast<uint32_t>(bit_cast<float>(ret) - magic);
	}

	ret |= ((uint32_t)x & 0x8000U) << 16;
	return bit_cast<float>(ret);
}

inline uint16_t float_to_half(float x)
{
	float magic = bit_cast<float>((uint32_t)15 << 23);
	uint32_t inf = 255UL << 23;
	uint32_t f16inf = 31UL << 23;
	uint32_t sign_mask = 0x80000000UL;
	uint32_t round_mask = ~0x0FFFU;

	uint32_t f;
	uint32_t sign;
	uint16_t ret;

	f = bit_cast<uint32_t>(x);
	sign = f & sign_mask;
	f ^= sign;

	if (f >= inf) {
		ret = f > inf ? 0x7E00 : 0x7C00;
	} else {
		f &= round_mask;
		f = bit_cast<uint32_t>(bit_cast<float>(f) * magic);
		f -= round_mask;

		if (bit_cast<uint32_t>(f) > f16inf)
			f = f16inf;

		ret = (uint16_t)(f >> 13);
	}

	ret |= (uint16_t)(sign >> 16);
	return ret;
}

template <class T>
class IntegerToFloat {
	float offset;
	float scale;
public:
	IntegerToFloat(int bits, bool fullrange, bool chroma) :
		offset{ (float)integer_offset(bits, fullrange, chroma) },
		scale{ 1.0f / (float)integer_range(bits, fullrange, chroma) }
	{}

	float operator()(T x) const
	{
		return (static_cast<float>(x) - offset) * scale;
	}
};

template <class T>
class FloatToInteger {
	float offset;
	float scale;
public:
	FloatToInteger(int bits, bool fullrange, bool chroma) :
		offset{ (float)integer_offset(bits, fullrange, chroma) },
		scale{ (float)integer_range(bits, fullrange, chroma) }
	{}

	T operator()(float x) const
	{
		int32_t u = (int32_t)(x * scale + offset + 0.5f);

		return static_cast<T>(clamp(u, (int32_t)0, (int32_t)std::numeric_limits<T>::max()));
	}
};

template <class T>
IntegerToFloat<T> make_integer_to_float(const PixelFormat &fmt)
{
	return{ fmt.depth, fmt.fullrange, fmt.chroma };
}

template <class T>
FloatToInteger<T> make_float_to_integer(const PixelFormat &fmt)
{
	return{ fmt.depth, fmt.fullrange, fmt.chroma };
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_QUANTIZE_H_
