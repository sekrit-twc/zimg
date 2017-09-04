#ifdef ZIMG_X86

#include <algorithm>
#include <cstdint>
#include <emmintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "depth_convert_x86.h"

#include "common/x86/sse_util.h"
#include "common/x86/sse2_util.h"

namespace zimg {
namespace depth {

namespace {

inline FORCE_INLINE __m128i mm_sll_epi8(__m128i x, __m128i count)
{
	__m128i lo = _mm_unpacklo_epi8(x, _mm_setzero_si128());
	__m128i hi = _mm_unpackhi_epi8(x, _mm_setzero_si128());

	lo = _mm_sll_epi16(lo, count);
	hi = _mm_sll_epi16(hi, count);

	return _mm_packus_epi16(lo, hi);
}

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

inline FORCE_INLINE void depth_convert_b2f_sse2_xiter(unsigned j, const uint8_t *src_p, __m128 scale, __m128 offset,
                                                      __m128 &lolo_out, __m128 &lohi_out, __m128 &hilo_out, __m128 &hihi_out)
{
	__m128i x = _mm_load_si128((const __m128i *)(src_p + j));
	__m128 lolo, lohi, hilo, hihi;

	mm_cvtepu8_ps(x, lolo, lohi, hilo, hihi);

	lolo = _mm_mul_ps(lolo, scale);
	lolo = _mm_add_ps(lolo, offset);

	lohi = _mm_mul_ps(lohi, scale);
	lohi = _mm_add_ps(lohi, offset);

	hilo = _mm_mul_ps(hilo, scale);
	hilo = _mm_add_ps(hilo, offset);

	hihi = _mm_mul_ps(hihi, scale);
	hihi = _mm_add_ps(hihi, offset);

	lolo_out = lolo;
	lohi_out = lohi;
	hilo_out = hilo;
	hihi_out = hihi;
}

inline FORCE_INLINE void depth_convert_w2f_sse2_xiter(unsigned j, const uint16_t *src_p, __m128 scale, __m128 offset,
                                                      __m128 &lo_out, __m128 &hi_out)
{
	__m128i x = _mm_load_si128((const __m128i *)(src_p + j));
	__m128 lo, hi;

	mm_cvtepu16_ps(x, lo, hi);

	lo = _mm_mul_ps(lo, scale);
	lo = _mm_add_ps(lo, offset);

	hi = _mm_mul_ps(hi, scale);
	hi = _mm_add_ps(hi, offset);

	lo_out = lo;
	hi_out = hi;
}

} // namespace


void left_shift_b2b_sse2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	__m128i count = _mm_set1_epi64x(shift);

	if (left != vec_left) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + vec_left - 16));
		x = mm_sll_epi8(x, count);
		mm_store_idxhi_epi8((__m128i *)(dst_p + vec_left - 16), x, left % 16);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + j));
		x = mm_sll_epi8(x, count);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + vec_right));
		x = mm_sll_epi8(x, count);
		mm_store_idxlo_epi8((__m128i *)(dst_p + vec_right), x, right % 16);
	}
}

void left_shift_b2w_sse2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	__m128i count = _mm_set1_epi64x(shift);

	if (left != vec_left) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + vec_left - 16));
		__m128i lo = _mm_unpacklo_epi8(x, _mm_setzero_si128());
		__m128i hi = _mm_unpackhi_epi8(x, _mm_setzero_si128());
		lo = _mm_sll_epi16(lo, count);
		hi = _mm_sll_epi16(hi, count);

		if (vec_left - left > 8) {
			mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 16), lo, left % 8);
			_mm_store_si128((__m128i *)(dst_p + vec_left - 8), hi);
		} else {
			mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 8), hi, left % 8);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + j));
		__m128i lo = _mm_unpacklo_epi8(x, _mm_setzero_si128());
		__m128i hi = _mm_unpackhi_epi8(x, _mm_setzero_si128());
		lo = _mm_sll_epi16(lo, count);
		hi = _mm_sll_epi16(hi, count);

		_mm_store_si128((__m128i *)(dst_p + j + 0), lo);
		_mm_store_si128((__m128i *)(dst_p + j + 8), hi);
	}

	if (right != vec_right) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + vec_right));
		__m128i lo = _mm_unpacklo_epi8(x, _mm_setzero_si128());
		__m128i hi = _mm_unpackhi_epi8(x, _mm_setzero_si128());
		lo = _mm_sll_epi16(lo, count);
		hi = _mm_sll_epi16(hi, count);

		if (right - vec_right > 8) {
			_mm_store_si128((__m128i *)(dst_p + vec_right), lo);
			mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right + 8), hi, right % 8);
		} else {
			// Modulo does not handle the case where there are exactly 8 remaining pixels.
			mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right), lo, right - vec_right);
		}
	}
}

void left_shift_w2b_sse2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	__m128i count = _mm_set1_epi64x(shift);

	if (left != vec_left) {
		__m128i lo = _mm_load_si128((const __m128i *)(src_p + vec_left - 16));
		__m128i hi = _mm_load_si128((const __m128i *)(src_p + vec_left - 8));
		lo = _mm_sll_epi16(lo, count);
		hi = _mm_sll_epi16(hi, count);
		lo = _mm_packus_epi16(lo, hi);
		mm_store_idxhi_epi8((__m128i *)(dst_p + vec_left - 16), lo, left % 16);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i lo = _mm_load_si128((const __m128i *)(src_p + j + 0));
		__m128i hi = _mm_load_si128((const __m128i *)(src_p + j + 8));
		lo = _mm_sll_epi16(lo, count);
		hi = _mm_sll_epi16(hi, count);
		lo = _mm_packus_epi16(lo, hi);
		_mm_store_si128((__m128i *)(dst_p + j), lo);
	}

	if (right != vec_right) {
		__m128i lo = _mm_load_si128((const __m128i *)(src_p + vec_right + 0));
		__m128i hi = _mm_load_si128((const __m128i *)(src_p + vec_right + 8));
		lo = _mm_sll_epi16(lo, count);
		hi = _mm_sll_epi16(hi, count);
		lo = _mm_packus_epi16(lo, hi);
		mm_store_idxlo_epi8((__m128i *)(dst_p + vec_right), lo, right % 16);
	}
}

void left_shift_w2w_sse2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	__m128i count = _mm_set1_epi64x(shift);

	if (left != vec_left) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + vec_left - 8));
		x = _mm_sll_epi16(x, count);
		mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 8), x, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + j));
		x = _mm_sll_epi16(x, count);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + vec_right));
		x = _mm_sll_epi16(x, count);
		mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right), x, right % 8);
	}
}

void depth_convert_b2f_sse2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);

	__m128 lolo, lohi, hilo, hihi;

#define XITER depth_convert_b2f_sse2_xiter
#define XARGS src_p, scale_ps, offset_ps, lolo, lohi, hilo, hihi
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);

		if (vec_left - left > 12) {
			mm_store_idxhi_ps(dst_p + vec_left - 16, lolo, left % 4);
			_mm_store_ps(dst_p + vec_left - 12, lohi);
			_mm_store_ps(dst_p + vec_left - 8, hilo);
			_mm_store_ps(dst_p + vec_left - 4, hihi);
		} else if (vec_left - left > 8) {
			mm_store_idxhi_ps(dst_p + vec_left - 12, lohi, left % 4);
			_mm_store_ps(dst_p + vec_left - 8, hilo);
			_mm_store_ps(dst_p + vec_left - 4, hihi);
		} else if (vec_left - left > 4) {
			mm_store_idxhi_ps(dst_p + vec_left - 8, hilo, left % 4);
			_mm_store_ps(dst_p + vec_left - 4, hihi);
		} else {
			mm_store_idxhi_ps(dst_p + vec_left - 4, hihi, left % 4);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);

		_mm_store_ps(dst_p + j + 0, lolo);
		_mm_store_ps(dst_p + j + 4, lohi);
		_mm_store_ps(dst_p + j + 8, hilo);
		_mm_store_ps(dst_p + j + 12, hihi);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		if (right - vec_right > 12) {
			_mm_store_ps(dst_p + vec_right + 0, lolo);
			_mm_store_ps(dst_p + vec_right + 4, lohi);
			_mm_store_ps(dst_p + vec_right + 8, hilo);
			mm_store_idxlo_ps(dst_p + vec_right + 12, hihi, right % 4);
		} else if (right - vec_right > 8) {
			_mm_store_ps(dst_p + vec_right + 0, lolo);
			_mm_store_ps(dst_p + vec_right + 4, lohi);
			mm_store_idxlo_ps(dst_p + vec_right + 8, hilo, right % 4);
		} else if (right - vec_right > 4) {
			_mm_store_ps(dst_p + vec_right + 0, lolo);
			mm_store_idxlo_ps(dst_p + vec_right + 4, lohi, right % 4);
		} else {
			// Modulo does not handle the case where there are exactly 4 remaining pixels.
			mm_store_idxlo_ps(dst_p + vec_right, lolo, right - vec_right);
		}
	}
#undef XITER
#undef XARGS
}

void depth_convert_w2f_sse2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const __m128 scale_ps = _mm_set_ps1(scale);
	const __m128 offset_ps = _mm_set_ps1(offset);

	__m128 lo, hi;

#define XITER depth_convert_w2f_sse2_xiter
#define XARGS src_p, scale_ps, offset_ps, lo, hi
	if (left != vec_left) {
		XITER(vec_left - 8, XARGS);

		if (vec_left - left > 4) {
			mm_store_idxhi_ps(dst_p + vec_left - 8, lo, left % 4);
			_mm_store_ps(dst_p + vec_left - 4, hi);
		} else {
			mm_store_idxhi_ps(dst_p + vec_left - 4, hi, left % 4);
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		XITER(j, XARGS);

		_mm_store_ps(dst_p + j + 0, lo);
		_mm_store_ps(dst_p + j + 4, hi);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		if (right - vec_right > 4) {
			_mm_store_ps(dst_p + vec_right + 0, lo);
			mm_store_idxlo_ps(dst_p + vec_right + 4, hi, right % 4);
		} else {
			// Modulo does not handle the case where there are exactly 4 remaining pixels.
			mm_store_idxlo_ps(dst_p + vec_right, lo, right - vec_right);
		}
	}
#undef XITER
#undef XARGS
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
