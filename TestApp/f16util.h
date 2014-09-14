#ifndef F16UTIL_H_
#define F16UTIL_H_

#include <algorithm>
#include <cstdint>
#include <intrin.h>

inline uint16_t cvt_u8_to_f16(uint8_t x)
{
	__m128 ss = _mm_set_ps1((float)x / (float)UINT8_MAX);
	__m128i sh = _mm_cvtps_ph(ss, 0);
	return _mm_extract_epi16(sh, 0);
}

inline uint8_t cvt_f16_to_u8(uint16_t x)
{
	__m128i sh = _mm_set1_epi16(x);
	__m128 ss = _mm_cvtph_ps(sh);
	float y = std::round(_mm_cvtss_f32(ss) * (float)UINT8_MAX);

	return (uint8_t)std::min(std::max(y, 0.f), (float)UINT8_MAX);
}

#endif // F16UTIL_H_
