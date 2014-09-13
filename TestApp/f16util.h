#ifndef F16UTIL_H_
#define F16UTIL_H_

#include <algorithm>
#include <cstdint>
#include <intrin.h>

inline void u8_to_f16(const uint8_t *src, uint16_t *dst, int n)
{
	for (int i = 0; i < n; ++i) {
		float x = src[i] / (float)UINT8_MAX;
		__m128 ss = _mm_set_ps1(x);
		__m128i sh = _mm_cvtps_ph(ss, 0);
		dst[i] = _mm_extract_epi16(sh, 0);
	}
}

inline void f16_to_u8(const uint16_t *src, uint8_t *dst, int n)
{
	for (int i = 0; i < n; ++i) {
		__m128i sh = _mm_set1_epi16(src[i]);
		__m128 ss = _mm_cvtph_ps(sh);
		float x = _mm_cvtss_f32(ss);

		x = std::round(x * (float)UINT8_MAX);
		dst[i] = (uint8_t)std::min(std::max(x, 0.f), (float)UINT8_MAX);
	}
}

#endif // F16UTIL_H_
