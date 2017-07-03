#ifdef ZIMG_X86

#include <cstdint>
#include <emmintrin.h>
#include "common/align.h"
#include "common/ccdep.h"

#define HAVE_CPU_SSE2
  #include "common/x86util.h"
#undef HAVE_CPU_SSE2

#include "dither_x86.h"

namespace zimg {
namespace depth {

namespace {

// Convert unsigned 16-bit to single precision.
inline FORCE_INLINE void mm_cvtepu16_ps(__m128i x, __m128 &lo, __m128 &hi)
{
	__m128i lo_dw = _mm_unpacklo_epi16(x, _mm_setzero_si128());
	__m128i hi_dw = _mm_unpackhi_epi16(x, _mm_setzero_si128());

	lo = _mm_cvtepi32_ps(lo_dw);
	hi = _mm_cvtepi32_ps(hi_dw);
}

// Convert unsigned 8-bit to single precision.
inline FORCE_INLINE void mm_cvtepu8_ps(__m128i x, __m128 &lolo, __m128 &lohi, __m128 &hilo, __m128 &hihi)
{
	__m128i lo_w = _mm_unpacklo_epi8(x, _mm_setzero_si128());
	__m128i hi_w = _mm_unpackhi_epi8(x, _mm_setzero_si128());

	mm_cvtepu16_ps(lo_w, lolo, lohi);
	mm_cvtepu16_ps(hi_w, hilo, hihi);
}

// Saturated convert single precision to unsigned 16-bit, biased by INT16_MIN.
inline FORCE_INLINE __m128i mm_cvtps_epu16_bias(__m128 lo, __m128 hi)
{
	__m128i lo_dw = _mm_cvtps_epi32(lo);
	__m128i hi_dw = _mm_cvtps_epi32(hi);

	return mm_packus_epi32_bias(lo_dw, hi_dw);
}

// Saturated convert single precision to unsigned 16-bit.
inline FORCE_INLINE __m128i mm_cvtps_epu16(__m128 lo, __m128 hi)
{
	__m128i lo_dw = _mm_cvtps_epi32(lo);
	__m128i hi_dw = _mm_cvtps_epi32(hi);

	return mm_packus_epi32(lo_dw, hi_dw);
}

// Saturated convert single precision to unsigned 8-bit.
inline FORCE_INLINE __m128i mm_cvtps_epu8(const __m128 &lolo, const __m128 &lohi, const __m128 &hilo, const __m128 &hihi)
{
	__m128i lo_w = mm_cvtps_epu16(lolo, lohi);
	__m128i hi_w = mm_cvtps_epu16(hilo, hihi);

	return _mm_packus_epi16(lo_w, hi_w);
}


inline FORCE_INLINE __m128i ordered_dither_b2b_sse2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                          const uint8_t *src_p, __m128 scale, __m128 offset, __m128i out_max)
{
	__m128i x = _mm_load_si128((const __m128i *)(src_p + j));
	__m128 lolo, lohi, hilo, hihi;
	__m128 dith;

	mm_cvtepu8_ps(x, lolo, lohi, hilo, hihi);

	lolo = _mm_mul_ps(lolo, scale);
	lolo = _mm_add_ps(lolo, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lolo = _mm_add_ps(lolo, dith);

	lohi = _mm_mul_ps(lohi, scale);
	lohi = _mm_add_ps(lohi, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 4) & dither_mask));
	lohi = _mm_add_ps(lohi, dith);

	hilo = _mm_mul_ps(hilo, scale);
	hilo = _mm_add_ps(hilo, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 8) & dither_mask));
	hilo = _mm_add_ps(hilo, dith);

	hihi = _mm_mul_ps(hihi, scale);
	hihi = _mm_add_ps(hihi, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 12) & dither_mask));
	hihi = _mm_add_ps(hihi, dith);

	x = mm_cvtps_epu8(lolo, lohi, hilo, hihi);
	x = _mm_min_epu8(x, out_max);

	return x;
}

inline FORCE_INLINE void ordered_dither_b2w_sse2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                       const uint8_t *src_p, __m128 scale, __m128 offset, __m128i out_max, __m128i &lo, __m128i &hi)
{
	const __m128i i16_min_epi16 = _mm_set1_epi16(INT16_MIN);

	__m128i x = _mm_load_si128((const __m128i *)(src_p + j));
	__m128 lolo, lohi, hilo, hihi;
	__m128 dith;

	mm_cvtepu8_ps(x, lolo, lohi, hilo, hihi);

	lolo = _mm_mul_ps(lolo, scale);
	lolo = _mm_add_ps(lolo, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lolo = _mm_add_ps(lolo, dith);

	lohi = _mm_mul_ps(lohi, scale);
	lohi = _mm_add_ps(lohi, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 4) & dither_mask));
	lohi = _mm_add_ps(lohi, dith);

	hilo = _mm_mul_ps(hilo, scale);
	hilo = _mm_add_ps(hilo, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 8) & dither_mask));
	hilo = _mm_add_ps(hilo, dith);

	hihi = _mm_mul_ps(hihi, scale);
	hihi = _mm_add_ps(hihi, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 12) & dither_mask));
	hihi = _mm_add_ps(hihi, dith);

	x = mm_cvtps_epu16_bias(lolo, lohi);
	x = _mm_min_epi16(x, out_max);
	x = _mm_sub_epi16(x, i16_min_epi16);
	lo = x;

	x = mm_cvtps_epu16_bias(hilo, hihi);
	x = _mm_min_epi16(x, out_max);
	x = _mm_sub_epi16(x, i16_min_epi16);
	hi = x;
}

inline FORCE_INLINE __m128i ordered_dither_w2b_sse2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                          const uint16_t *src_p, __m128 scale, __m128 offset, __m128i out_max)
{
	__m128i x0 = _mm_load_si128((const __m128i *)(src_p + j + 0));
	__m128i x1 = _mm_load_si128((const __m128i *)(src_p + j + 8));
	__m128 lolo, lohi, hilo, hihi;
	__m128 dith;

	mm_cvtepu16_ps(x0, lolo, lohi);
	mm_cvtepu16_ps(x1, hilo, hihi);

	lolo = _mm_mul_ps(lolo, scale);
	lolo = _mm_add_ps(lolo, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lolo = _mm_add_ps(lolo, dith);

	lohi = _mm_mul_ps(lohi, scale);
	lohi = _mm_add_ps(lohi, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 4) & dither_mask));
	lohi = _mm_add_ps(lohi, dith);

	hilo = _mm_mul_ps(hilo, scale);
	hilo = _mm_add_ps(hilo, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 8) & dither_mask));
	hilo = _mm_add_ps(hilo, dith);

	hihi = _mm_mul_ps(hihi, scale);
	hihi = _mm_add_ps(hihi, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 12) & dither_mask));
	hihi = _mm_add_ps(hihi, dith);

	x0 = mm_cvtps_epu8(lolo, lohi, hilo, hihi);
	x0 = _mm_min_epu8(x0, out_max);

	return x0;
}

inline FORCE_INLINE __m128i ordered_dither_w2w_sse2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                          const uint16_t *src_p, __m128 scale, __m128 offset, __m128i out_max)
{
	const __m128i i16_min_epi16 = _mm_set1_epi16(INT16_MIN);

	__m128i x = _mm_load_si128((const __m128i *)(src_p + j));
	__m128 lo, hi, di;

	mm_cvtepu16_ps(x, lo, hi);

	lo = _mm_mul_ps(lo, scale);
	lo = _mm_add_ps(lo, offset);
	di = _mm_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lo = _mm_add_ps(lo, di);

	hi = _mm_mul_ps(hi, scale);
	hi = _mm_add_ps(hi, offset);
	di = _mm_load_ps(dither + ((dither_offset + j + 4) & dither_mask));
	hi = _mm_add_ps(hi, di);

	x = mm_cvtps_epu16_bias(lo, hi);
	x = _mm_min_epi16(x, out_max);
	x = _mm_sub_epi16(x, i16_min_epi16);

	return x;
}

inline FORCE_INLINE __m128i ordered_dither_f2b_sse2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                          const float *src_p, __m128 scale, __m128 offset, __m128i out_max)
{
	__m128 lolo = _mm_load_ps(src_p + j + 0);
	__m128 lohi = _mm_load_ps(src_p + j + 4);
	__m128 hilo = _mm_load_ps(src_p + j + 8);
	__m128 hihi = _mm_load_ps(src_p + j + 12);
	__m128 dith;
	__m128i x;

	lolo = _mm_mul_ps(lolo, scale);
	lolo = _mm_add_ps(lolo, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lolo = _mm_add_ps(lolo, dith);

	lohi = _mm_mul_ps(lohi, scale);
	lohi = _mm_add_ps(lohi, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 4) & dither_mask));
	lohi = _mm_add_ps(lohi, dith);

	hilo = _mm_mul_ps(hilo, scale);
	hilo = _mm_add_ps(hilo, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 8) & dither_mask));
	hilo = _mm_add_ps(hilo, dith);

	hihi = _mm_mul_ps(hihi, scale);
	hihi = _mm_add_ps(hihi, offset);
	dith = _mm_load_ps(dither + ((dither_offset + j + 12) & dither_mask));
	hihi = _mm_add_ps(hihi, dith);

	x = mm_cvtps_epu8(lolo, lohi, hilo, hihi);
	x = _mm_min_epu8(x, out_max);

	return x;
}

inline FORCE_INLINE __m128i ordered_dither_f2w_sse2_xiter(unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                          const float *src_p, __m128 scale, __m128 offset, __m128i out_max)
{
	const __m128i i16_min_epi16 = _mm_set1_epi16(INT16_MIN);

	__m128 lo = _mm_load_ps(src_p + j + 0);
	__m128 hi = _mm_load_ps(src_p + j + 4);
	__m128 di;
	__m128i x;

	lo = _mm_mul_ps(lo, scale);
	lo = _mm_add_ps(lo, offset);
	di = _mm_load_ps(dither + ((dither_offset + j + 0) & dither_mask));
	lo = _mm_add_ps(lo, di);

	hi = _mm_mul_ps(hi, scale);
	hi = _mm_add_ps(hi, offset);
	di = _mm_load_ps(dither + ((dither_offset + j + 4) & dither_mask));
	hi = _mm_add_ps(hi, di);

	x = mm_cvtps_epu16_bias(lo, hi);
	x = _mm_min_epi16(x, out_max);
	x = _mm_sub_epi16(x, i16_min_epi16);

	return x;
}

} // namespace


void ordered_dither_b2b_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);
	const __m128i out_max = _mm_set1_epi8(static_cast<uint8_t>((1 << bits) - 1));

#define XITER ordered_dither_b2b_sse2_xiter
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m128i x = XITER(vec_left - 16, XARGS);
		mm_store_idxhi_epi8((__m128i *)(dst_p + vec_left - 16), x, left % 16);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = XITER(j, XARGS);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = XITER(vec_right, XARGS);
		mm_store_idxlo_epi8((__m128i *)(dst_p + vec_right), x, right % 16);
	}
#undef XITER
#undef XARGS
}

void ordered_dither_b2w_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);
	const __m128i out_max = _mm_set1_epi16((int16_t)((1UL << bits) - 1) + INT16_MIN);

	__m128i lo, hi;

#define XITER ordered_dither_b2w_sse2_xiter
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max, lo, hi
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);

		if (vec_left - left > 8) {
			mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 16), lo, left % 8);
			_mm_store_si128((__m128i *)(dst_p + vec_left - 8), hi);
		} else {
			mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 8), hi, left % 8);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);

		_mm_store_si128((__m128i *)(dst_p + j + 0), lo);
		_mm_store_si128((__m128i *)(dst_p + j + 8), hi);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		if (right - vec_right > 8) {
			_mm_store_si128((__m128i *)(dst_p + vec_right), lo);
			mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right + 8), hi, right % 8);
		} else {
			mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right), lo, right % 8);
		}
	}
#undef XITER
#undef XARGS
}

void ordered_dither_w2b_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);
	const __m128i out_max = _mm_set1_epi8(static_cast<uint8_t>((1 << bits) - 1));

#define XITER ordered_dither_w2b_sse2_xiter
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m128i x = XITER(vec_left - 16, XARGS);
		mm_store_idxhi_epi8((__m128i *)(dst_p + vec_left - 16), x, left % 16);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = XITER(j, XARGS);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = XITER(vec_right, XARGS);
		mm_store_idxlo_epi8((__m128i *)(dst_p + vec_right), x, right % 16);
	}
#undef XITER
#undef XARGS
}

void ordered_dither_w2w_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);
	const __m128i out_max = _mm_set1_epi16((int16_t)((1UL << bits) - 1) + INT16_MIN);

#define XITER ordered_dither_w2w_sse2_xiter
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m128i x = XITER(vec_left - 8, XARGS);
		mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 8), x, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m128i x = XITER(j, XARGS);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = XITER(vec_right, XARGS);
		mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right), x, right % 8);
	}
#undef XITER
#undef XARGS
}

void ordered_dither_f2b_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const float *src_p = static_cast<const float *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);
	const __m128i out_max = _mm_set1_epi8(static_cast<uint8_t>((1 << bits) - 1));

#define XITER ordered_dither_f2b_sse2_xiter
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m128i x = XITER(vec_left - 16, XARGS);
		mm_store_idxhi_epi8((__m128i *)(dst_p + vec_left - 16), x, left % 16);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = XITER(j, XARGS);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = XITER(vec_right, XARGS);
		mm_store_idxlo_epi8((__m128i *)(dst_p + vec_right), x, right % 16);
	}
#undef XITER
#undef XARGS
}

void ordered_dither_f2w_sse2(const float *dither, unsigned dither_offset, unsigned dither_mask,
                             const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);
	const __m128i out_max = _mm_set1_epi16((int16_t)((1UL << bits) - 1) + INT16_MIN);

#define XITER ordered_dither_f2w_sse2_xiter
#define XARGS dither, dither_offset, dither_mask, src_p, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m128i x = XITER(vec_left - 8, XARGS);
		mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 8), x, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m128i x = XITER(j, XARGS);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = XITER(vec_right, XARGS);
		mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right), x, right % 8);
	}
#undef XITER
#undef XARGS
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
