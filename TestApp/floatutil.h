#ifndef FLOATUTIL_H_
#define FLOATUTIL_H_

#include <cstdint>
#include <emmintrin.h>
#include <immintrin.h>

inline uint16_t float_to_half(float x)
{
	__m128  ss = _mm_set_ps1(x);
	__m128i sh = _mm_cvtps_ph(ss, 0);
	return _mm_extract_epi16(sh, 0);
}

inline float half_to_float(uint16_t x)
{
	__m128i sh = _mm_set1_epi16(x);
	__m128  ss = _mm_cvtph_ps(sh);
	return _mm_cvtss_f32(ss);
}

float normalize_float(float x, float max)
{
	x /= max;
	x = x < 0.0f ? 0.0f : x;
	x = x > 1.0f ? 1.0f : x;
	return x;
}

float denormalize_float(float x, float max)
{
	x *= max;
	x = x < 0.0f ? 0.0f : x;
	x = x > max ? max : x;
	return x;
}

#endif // FLOATUTIL_H_
