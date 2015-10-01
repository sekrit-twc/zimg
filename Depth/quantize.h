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

#define FLOAT_HALF_MANT_SHIFT (23 - 10)
#define FLOAT_HALF_EXP_ADJUST (127 - 15)
inline float half_to_float(uint16_t f16w)
{
	const unsigned exp_nonfinite_f16 = 0x1F;
	const unsigned exp_nonfinite_f32 = 0xFF;

	const uint32_t mant_qnan_f32 = 0x00400000UL;

	uint16_t sign = (f16w & 0x8000U) >> 15;
	uint16_t exp = (f16w & 0x7C00U) >> 10;
	uint16_t mant = (f16w & 0x03FFU) >> 0;

	uint32_t f32dw;
	uint32_t exp_f32;
	uint32_t mant_f32;

	// Non-finite.
	if (exp == exp_nonfinite_f16) {
		exp_f32 = exp_nonfinite_f32;

		// Zero extend mantissa and convert sNaN to qNaN.
		if (mant)
			mant_f32 = (mant << FLOAT_HALF_MANT_SHIFT) | mant_qnan_f32;
		else
			mant_f32 = 0;
	} else {
		uint16_t mant_adjust;

		// Denormal.
		if (exp == 0) {
			// Special zero denorm.
			if (mant == 0) {
				mant_adjust = 0;
				exp_f32 = 0;
			} else {
				unsigned renorm = 0;
				mant_adjust = mant;

				while ((mant_adjust & 0x0400) == 0) {
					mant_adjust <<= 1;
					++renorm;
				}

				mant_adjust &= ~0x0400;
				exp_f32 = FLOAT_HALF_EXP_ADJUST - renorm + 1;
			}
		} else {
			mant_adjust = mant;
			exp_f32 = exp + FLOAT_HALF_EXP_ADJUST;
		}

		mant_f32 = (uint32_t)mant_adjust << FLOAT_HALF_MANT_SHIFT;
	}

	f32dw = ((uint32_t)sign << 31) | (exp_f32 << 23) | (mant_f32);
	return bit_cast<float>(f32dw);
}

inline uint16_t float_to_half(float f32)
{
	const unsigned exp_nonfinite_f32 = 0xFF;
	const unsigned exp_nonfinite_f16 = 0x1F;

	const unsigned mant_qnan_f16 = 0x0200;
	const unsigned mant_max_f16 = 0x03FF;

	uint32_t f32dw = bit_cast<uint32_t>(f32);
	uint32_t sign = (f32dw & 0x80000000UL) >> 31;
	uint32_t exp = (f32dw & 0x7F800000UL) >> 23;
	uint32_t mant = (f32dw & 0x007FFFFFUL) >> 0;

	uint32_t exp_f16;
	uint32_t mant_f16;

	// Non-finite.
	if (exp == exp_nonfinite_f32) {
		exp_f16 = exp_nonfinite_f16;

		// Truncate mantissa and convert sNaN to qNaN.
		if (mant)
			mant_f16 = (mant >> FLOAT_HALF_MANT_SHIFT) | mant_qnan_f16;
		else
			mant_f16 = 0;
	} else {
		uint32_t mant_adjust;
		uint32_t shift;
		uint32_t half;

		// Denormal.
		if (exp <= FLOAT_HALF_EXP_ADJUST) {
			shift = FLOAT_HALF_MANT_SHIFT + FLOAT_HALF_EXP_ADJUST - exp + 1;

			if (shift > 31)
				shift = 31;

			mant_adjust = mant | (1UL << 23);
			exp_f16 = 0;
		} else {
			shift = FLOAT_HALF_MANT_SHIFT;
			mant_adjust = mant;
			exp_f16 = exp - FLOAT_HALF_EXP_ADJUST;
		}

		half = 1UL << (shift - 1);

		// Round half to even.
		mant_f16 = (mant_adjust + half - 1 + ((mant_adjust >> shift) & 1)) >> shift;

		// Detect overflow.
		if (mant_f16 > mant_max_f16) {
			mant_f16 &= mant_max_f16;
			exp_f16 += 1;
		}
		if (exp_f16 >= exp_nonfinite_f16) {
			exp_f16 = exp_nonfinite_f16;
			mant_f16 = 0;
		}
	}

	return ((uint16_t)sign << 15) | ((uint16_t)exp_f16 << 10) | ((uint16_t)mant_f16);
}
#undef FLOAT_HALF_MANT_SHIFT
#undef FLOAT_HALF_EXP_ADJUST

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_QUANTIZE_H_
