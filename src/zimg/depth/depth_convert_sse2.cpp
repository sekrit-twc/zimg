#ifdef ZIMG_X86

#include <cstdint>
#include <emmintrin.h>
#include "common/align.h"

#define HAVE_CPU_SSE2
  #include "common/x86util.h"
#undef HAVE_CPU_SSE2

#include "depth_convert_x86.h"

namespace zimg {;
namespace depth {
;

namespace {
;

inline FORCE_INLINE void mm_store_left_epi8(uint8_t *dst, __m128i x, unsigned count)
{
	mm_store_left_si128((__m128i *)dst, x, count);
}

inline FORCE_INLINE void mm_store_right_epi8(uint8_t *dst, __m128i x, unsigned count)
{
	mm_store_right_si128((__m128i *)dst, x, count);
}

inline FORCE_INLINE void mm_store_left_epi16(uint16_t *dst, __m128i x, unsigned count)
{
	mm_store_left_si128((__m128i *)dst, x, count * 2);
}

inline FORCE_INLINE void mm_store_right_epi16(uint16_t *dst, __m128i x, unsigned count)
{
	mm_store_right_si128((__m128i *)dst, x, count * 2);
}

inline FORCE_INLINE __m128i mm_sll_epi8(__m128i x, __m128i count)
{
	__m128i lo = _mm_unpacklo_epi8(x, _mm_setzero_si128());
	__m128i hi = _mm_unpackhi_epi8(x, _mm_setzero_si128());

	lo = _mm_sll_epi16(lo, count);
	hi = _mm_sll_epi16(hi, count);

	return _mm_packus_epi16(lo, hi);
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
		mm_store_left_epi8(dst_p + vec_left - 16, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + j));
		x = mm_sll_epi8(x, count);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + vec_right));
		x = mm_sll_epi8(x, count);
		mm_store_right_epi8(dst_p + vec_right, x, right - vec_right);
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
			mm_store_left_epi16(dst_p + vec_left - 16, lo, (vec_left - left) % 8);
			_mm_store_si128((__m128i *)(dst_p + vec_left - 8), hi);
		} else {
			mm_store_left_epi16(dst_p + vec_left - 8, hi, vec_left - left);
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
			mm_store_right_epi16(dst_p + vec_right + 8, hi, (right - vec_right) % 8);
		} else {
			mm_store_right_epi16(dst_p + vec_right, lo, right - vec_right);
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
		mm_store_left_epi8(dst_p + vec_left - 16, lo, vec_left - left);
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
		mm_store_right_epi8(dst_p + vec_right, lo, right - vec_right);
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
		mm_store_left_epi16(dst_p + vec_left - 8, x, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + j));
		x = _mm_sll_epi16(x, count);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = _mm_load_si128((const __m128i *)(src_p + vec_right));
		x = _mm_sll_epi16(x, count);
		mm_store_right_epi16(dst_p + vec_right, x, right - vec_right);
	}
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
