#ifdef ZIMG_X86_AVX512

#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "resize/resize_impl.h"
#include "resize_impl_x86.h"

#include "common/x86/sse_util.h"
#include "common/x86/sse2_util.h"
#include "common/x86/avx_util.h"
#include "common/x86/avx2_util.h"
#include "common/x86/avx512_util.h"

namespace zimg {
namespace resize {

namespace {

struct f16_traits {
	typedef __m256i vec16_type;
	typedef uint16_t pixel_type;

	static constexpr PixelType type_constant = PixelType::HALF;

	static inline FORCE_INLINE vec16_type load16_raw(const pixel_type *ptr)
	{
		return _mm256_loadu_si256((const __m256i *)ptr);
	}

	static inline FORCE_INLINE void store16_raw(pixel_type *ptr, vec16_type x)
	{
		_mm256_storeu_si256((__m256i *)ptr, x);
	}

	static inline FORCE_INLINE __m512 load16(const pixel_type *ptr)
	{
		return _mm512_cvtph_ps(load16_raw(ptr));
	}

	static inline FORCE_INLINE __m512 maskz_load16(__mmask16 mask, const pixel_type *ptr)
	{
		return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, ptr));
	}

	static inline FORCE_INLINE void store16(pixel_type *ptr, __m512 x)
	{
		store16_raw(ptr, _mm512_cvtps_ph(x, 0));
	}

	static inline FORCE_INLINE void mask_store16(pixel_type *ptr, __mmask16 mask, __m512 x)
	{
		_mm256_mask_storeu_epi16(ptr, mask, _mm512_cvtps_ph(x, 0));
	}

	static inline FORCE_INLINE void transpose16(vec16_type &x0, vec16_type &x1, vec16_type &x2, vec16_type &x3,
	                                            vec16_type &x4, vec16_type &x5, vec16_type &x6, vec16_type &x7,
	                                            vec16_type &x8, vec16_type &x9, vec16_type &x10, vec16_type &x11,
	                                            vec16_type &x12, vec16_type &x13, vec16_type &x14, vec16_type &x15)
	{
		mm256_transpose16_epi16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
	}

	static inline FORCE_INLINE void scatter16(pixel_type *dst0, pixel_type *dst1, pixel_type *dst2, pixel_type *dst3,
	                                          pixel_type *dst4, pixel_type *dst5, pixel_type *dst6, pixel_type *dst7,
	                                          pixel_type *dst8, pixel_type *dst9, pixel_type *dst10, pixel_type *dst11,
	                                          pixel_type *dst12, pixel_type *dst13, pixel_type *dst14, pixel_type *dst15, __m512 x)
	{
		__m256i y = _mm512_cvtps_ph(x, 0);
		mm_scatter_epi16(dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7, _mm256_castsi256_si128(y));
		mm_scatter_epi16(dst8, dst9, dst10, dst11, dst12, dst13, dst14, dst15, _mm256_extracti128_si256(y, 1));
	}
};

struct f32_traits {
	typedef __m512 vec16_type;
	typedef float pixel_type;

	static constexpr PixelType type_constant = PixelType::FLOAT;

	static inline FORCE_INLINE vec16_type load16_raw(const pixel_type *ptr)
	{
		return _mm512_loadu_ps(ptr);
	}

	static inline FORCE_INLINE void store16_raw(pixel_type *ptr, vec16_type x)
	{
		_mm512_store_ps(ptr, x);
	}

	static inline FORCE_INLINE __m512 load16(const pixel_type *ptr)
	{
		return load16_raw(ptr);
	}

	static inline FORCE_INLINE __m512 maskz_load16(__mmask16 mask, const pixel_type *ptr)
	{
		return _mm512_maskz_loadu_ps(mask, ptr);
	}

	static inline FORCE_INLINE void store16(pixel_type *ptr, __m512 x)
	{
		store16_raw(ptr, x);
	}

	static inline FORCE_INLINE void mask_store16(pixel_type *ptr, __mmask16 mask, __m512 x)
	{
		_mm512_mask_store_ps(ptr, mask, x);
	}

	static inline FORCE_INLINE void transpose16(vec16_type &x0, vec16_type &x1, vec16_type &x2, vec16_type &x3,
	                                            vec16_type &x4, vec16_type &x5, vec16_type &x6, vec16_type &x7,
	                                            vec16_type &x8, vec16_type &x9, vec16_type &x10, vec16_type &x11,
	                                            vec16_type &x12, vec16_type &x13, vec16_type &x14, vec16_type &x15)
	{
		mm512_transpose16_ps(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);
	}

	static inline FORCE_INLINE void scatter16(pixel_type *dst0, pixel_type *dst1, pixel_type *dst2, pixel_type *dst3,
	                                          pixel_type *dst4, pixel_type *dst5, pixel_type *dst6, pixel_type *dst7,
	                                          pixel_type *dst8, pixel_type *dst9, pixel_type *dst10, pixel_type *dst11,
	                                          pixel_type *dst12, pixel_type *dst13, pixel_type *dst14, pixel_type *dst15, __m512 x)
	{
		mm_scatter_ps(dst0, dst1, dst2, dst3, _mm512_castps512_ps128(x));
		mm_scatter_ps(dst4, dst5, dst6, dst7, _mm512_extractf32x4_ps(x, 1));
		mm_scatter_ps(dst8, dst9, dst10, dst11, _mm512_extractf32x4_ps(x, 2));
		mm_scatter_ps(dst12, dst13, dst14, dst15, _mm512_extractf32x4_ps(x, 3));
	}
};


inline FORCE_INLINE __m256i export_i30_u16(__m512i x)
{
	const __m512i round = _mm512_set1_epi32(1 << 13);
	x = _mm512_add_epi32(x, round);
	x = _mm512_srai_epi32(x, 14);
	return _mm512_cvtsepi32_epi16(x);
}

inline FORCE_INLINE __m512i export2_i30_u16(__m512i lo, __m512i hi)
{
	const __m512i round = _mm512_set1_epi32(1 << 13);

	lo = _mm512_add_epi32(lo, round);
	hi = _mm512_add_epi32(hi, round);

	lo = _mm512_srai_epi32(lo, 14);
	hi = _mm512_srai_epi32(hi, 14);

	lo = _mm512_packs_epi32(lo, hi);

	return lo;
}


template <class Traits, class T>
void transpose_line_16x16(T * RESTRICT dst, const T * const * RESTRICT src, unsigned left, unsigned right)
{
	typedef typename Traits::vec16_type vec16_type;

	for (unsigned j = left; j < right; j += 16) {
		vec16_type x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

		x0 = Traits::load16_raw(src[0] + j);
		x1 = Traits::load16_raw(src[1] + j);
		x2 = Traits::load16_raw(src[2] + j);
		x3 = Traits::load16_raw(src[3] + j);
		x4 = Traits::load16_raw(src[4] + j);
		x5 = Traits::load16_raw(src[5] + j);
		x6 = Traits::load16_raw(src[6] + j);
		x7 = Traits::load16_raw(src[7] + j);
		x8 = Traits::load16_raw(src[8] + j);
		x9 = Traits::load16_raw(src[9] + j);
		x10 = Traits::load16_raw(src[10] + j);
		x11 = Traits::load16_raw(src[11] + j);
		x12 = Traits::load16_raw(src[12] + j);
		x13 = Traits::load16_raw(src[13] + j);
		x14 = Traits::load16_raw(src[14] + j);
		x15 = Traits::load16_raw(src[15] + j);

		Traits::transpose16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);

		Traits::store16_raw(dst + 0, x0);
		Traits::store16_raw(dst + 16, x1);
		Traits::store16_raw(dst + 32, x2);
		Traits::store16_raw(dst + 48, x3);
		Traits::store16_raw(dst + 64, x4);
		Traits::store16_raw(dst + 80, x5);
		Traits::store16_raw(dst + 96, x6);
		Traits::store16_raw(dst + 112, x7);
		Traits::store16_raw(dst + 128, x8);
		Traits::store16_raw(dst + 144, x9);
		Traits::store16_raw(dst + 160, x10);
		Traits::store16_raw(dst + 176, x11);
		Traits::store16_raw(dst + 192, x12);
		Traits::store16_raw(dst + 208, x13);
		Traits::store16_raw(dst + 224, x14);
		Traits::store16_raw(dst + 240, x15);

		dst += 256;
	}
}

void transpose_line_32x32_epi16(uint16_t * RESTRICT dst, const uint16_t * const * RESTRICT src, unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; j += 32) {
		__m512i x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
		__m512i x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31;

		x0 = _mm512_load_si512(src[0] + j);
		x1 = _mm512_load_si512(src[1] + j);
		x2 = _mm512_load_si512(src[2] + j);
		x3 = _mm512_load_si512(src[3] + j);
		x4 = _mm512_load_si512(src[4] + j);
		x5 = _mm512_load_si512(src[5] + j);
		x6 = _mm512_load_si512(src[6] + j);
		x7 = _mm512_load_si512(src[7] + j);
		x8 = _mm512_load_si512(src[8] + j);
		x9 = _mm512_load_si512(src[9] + j);
		x10 = _mm512_load_si512(src[10] + j);
		x11 = _mm512_load_si512(src[11] + j);
		x12 = _mm512_load_si512(src[12] + j);
		x13 = _mm512_load_si512(src[13] + j);
		x14 = _mm512_load_si512(src[14] + j);
		x15 = _mm512_load_si512(src[15] + j);
		x16 = _mm512_load_si512(src[16] + j);
		x17 = _mm512_load_si512(src[17] + j);
		x18 = _mm512_load_si512(src[18] + j);
		x19 = _mm512_load_si512(src[19] + j);
		x20 = _mm512_load_si512(src[20] + j);
		x21 = _mm512_load_si512(src[21] + j);
		x22 = _mm512_load_si512(src[22] + j);
		x23 = _mm512_load_si512(src[23] + j);
		x24 = _mm512_load_si512(src[24] + j);
		x25 = _mm512_load_si512(src[25] + j);
		x26 = _mm512_load_si512(src[26] + j);
		x27 = _mm512_load_si512(src[27] + j);
		x28 = _mm512_load_si512(src[28] + j);
		x29 = _mm512_load_si512(src[29] + j);
		x30 = _mm512_load_si512(src[30] + j);
		x31 = _mm512_load_si512(src[31] + j);

		mm512_transpose32_epi16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,
		                        x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31);

		_mm512_store_si512(dst + 0, x0);
		_mm512_store_si512(dst + 32, x1);
		_mm512_store_si512(dst + 64, x2);
		_mm512_store_si512(dst + 96, x3);
		_mm512_store_si512(dst + 128, x4);
		_mm512_store_si512(dst + 160, x5);
		_mm512_store_si512(dst + 192, x6);
		_mm512_store_si512(dst + 224, x7);
		_mm512_store_si512(dst + 256, x8);
		_mm512_store_si512(dst + 288, x9);
		_mm512_store_si512(dst + 320, x10);
		_mm512_store_si512(dst + 352, x11);
		_mm512_store_si512(dst + 384, x12);
		_mm512_store_si512(dst + 416, x13);
		_mm512_store_si512(dst + 448, x14);
		_mm512_store_si512(dst + 480, x15);
		_mm512_store_si512(dst + 512, x16);
		_mm512_store_si512(dst + 544, x17);
		_mm512_store_si512(dst + 576, x18);
		_mm512_store_si512(dst + 608, x19);
		_mm512_store_si512(dst + 640, x20);
		_mm512_store_si512(dst + 672, x21);
		_mm512_store_si512(dst + 704, x22);
		_mm512_store_si512(dst + 736, x23);
		_mm512_store_si512(dst + 768, x24);
		_mm512_store_si512(dst + 800, x25);
		_mm512_store_si512(dst + 832, x26);
		_mm512_store_si512(dst + 864, x27);
		_mm512_store_si512(dst + 896, x28);
		_mm512_store_si512(dst + 928, x29);
		_mm512_store_si512(dst + 960, x30);
		_mm512_store_si512(dst + 992, x31);

		dst += 1024;
	}
}


template <bool DoLoop, unsigned Tail>
inline FORCE_INLINE __m512i resize_line16_h_u16_avx512_xiter(unsigned j,
                                                             const unsigned * RESTRICT filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                                             const uint16_t * RESTRICT src, unsigned src_base, uint16_t limit)
{
	const __m512i i16_min = _mm512_set1_epi16(INT16_MIN);
	const __m512i lim = _mm512_set1_epi16(limit + INT16_MIN);

	const int16_t *filter_coeffs = filter_data + j * filter_stride;
	const uint16_t *src_p = src + (filter_left[j] - src_base) * 32;

	__m512i accum_lo = _mm512_setzero_si512();
	__m512i accum_hi = _mm512_setzero_si512();
	__m512i x0, x1, xl, xh, c, coeffs;

	unsigned k_end = DoLoop ? floor_n(filter_width + 1, 8) : 0;

	for (unsigned k = 0; k < k_end; k += 8) {
		coeffs = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)(filter_coeffs + k)));

		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_AAAA);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 0));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 32));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c, xl);
		xh = _mm512_madd_epi16(c, xh);

		accum_lo = _mm512_add_epi32(accum_lo, xl);
		accum_hi = _mm512_add_epi32(accum_hi, xh);

		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_BBBB);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 64));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 96));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c, xl);
		xh = _mm512_madd_epi16(c, xh);

		accum_lo = _mm512_add_epi32(accum_lo, xl);
		accum_hi = _mm512_add_epi32(accum_hi, xh);

		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_CCCC);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 128));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 160));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c, xl);
		xh = _mm512_madd_epi16(c, xh);

		accum_lo = _mm512_add_epi32(accum_lo, xl);
		accum_hi = _mm512_add_epi32(accum_hi, xh);

		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_DDDD);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 192));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 224));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c, xl);
		xh = _mm512_madd_epi16(c, xh);

		accum_lo = _mm512_add_epi32(accum_lo, xl);
		accum_hi = _mm512_add_epi32(accum_hi, xh);

		src_p += 256;
	}

	if (Tail >= 2) {
		coeffs = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)(filter_coeffs + k_end)));

		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_AAAA);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 0));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 32));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c, xl);
		xh = _mm512_madd_epi16(c, xh);

		accum_lo = _mm512_add_epi32(accum_lo, xl);
		accum_hi = _mm512_add_epi32(accum_hi, xh);
	}

	if (Tail >= 4) {
		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_BBBB);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 64));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 96));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c, xl);
		xh = _mm512_madd_epi16(c, xh);

		accum_lo = _mm512_add_epi32(accum_lo, xl);
		accum_hi = _mm512_add_epi32(accum_hi, xh);
	}

	if (Tail >= 6) {
		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_CCCC);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 128));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 160));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c, xl);
		xh = _mm512_madd_epi16(c, xh);

		accum_lo = _mm512_add_epi32(accum_lo, xl);
		accum_hi = _mm512_add_epi32(accum_hi, xh);
	}

	if (Tail >= 8) {
		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_DDDD);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 192));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 224));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c, xl);
		xh = _mm512_madd_epi16(c, xh);

		accum_lo = _mm512_add_epi32(accum_lo, xl);
		accum_hi = _mm512_add_epi32(accum_hi, xh);
	}

	accum_lo = export2_i30_u16(accum_lo, accum_hi);
	accum_lo = _mm512_min_epi16(accum_lo, lim);
	accum_lo = _mm512_sub_epi16(accum_lo, i16_min);
	return accum_lo;
}

template <bool DoLoop, unsigned Tail>
void resize_line16_h_u16_avx512(const unsigned * RESTRICT filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                const uint16_t * RESTRICT src, uint16_t * const * RESTRICT dst, unsigned src_base, unsigned left, unsigned right, uint16_t limit)
{
	unsigned vec_left = ceil_n(left, 32);
	unsigned vec_right = floor_n(right, 32);

#define XITER resize_line16_h_u16_avx512_xiter<DoLoop, Tail>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src, src_base, limit
	for (unsigned j = left; j < vec_left; ++j) {
		__m512i x = XITER(j, XARGS);

		mm_scatter_epi16(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, dst[4] + j, dst[5] + j, dst[6] + j, dst[7] + j, _mm512_castsi512_si128(x));
		mm_scatter_epi16(dst[8] + j, dst[9] + j, dst[10] + j, dst[11] + j, dst[12] + j, dst[13] + j, dst[14] + j, dst[15] + j, _mm512_extracti32x4_epi32(x, 1));
		mm_scatter_epi16(dst[16] + j, dst[17] + j, dst[18] + j, dst[19] + j, dst[20] + j, dst[21] + j, dst[22] + j, dst[23] + j, _mm512_extracti32x4_epi32(x, 2));
		mm_scatter_epi16(dst[24] + j, dst[25] + j, dst[26] + j, dst[27] + j, dst[28] + j, dst[29] + j, dst[30] + j, dst[31] + j, _mm512_extracti32x4_epi32(x, 3));
	}

	for (unsigned j = vec_left; j < vec_right; j += 32) {
		uint16_t cache alignas(64)[32][32];
		__m512i x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
		__m512i x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31;

		for (unsigned jj = j; jj < j + 32; ++jj) {
			__m512i x = XITER(jj, XARGS);
			_mm512_store_si512(cache[jj - j], x);
		}

		x0 = _mm512_load_si512(cache[0]);
		x1 = _mm512_load_si512(cache[1]);
		x2 = _mm512_load_si512(cache[2]);
		x3 = _mm512_load_si512(cache[3]);
		x4 = _mm512_load_si512(cache[4]);
		x5 = _mm512_load_si512(cache[5]);
		x6 = _mm512_load_si512(cache[6]);
		x7 = _mm512_load_si512(cache[7]);
		x8 = _mm512_load_si512(cache[8]);
		x9 = _mm512_load_si512(cache[9]);
		x10 = _mm512_load_si512(cache[10]);
		x11 = _mm512_load_si512(cache[11]);
		x12 = _mm512_load_si512(cache[12]);
		x13 = _mm512_load_si512(cache[13]);
		x14 = _mm512_load_si512(cache[14]);
		x15 = _mm512_load_si512(cache[15]);
		x16 = _mm512_load_si512(cache[16]);
		x17 = _mm512_load_si512(cache[17]);
		x18 = _mm512_load_si512(cache[18]);
		x19 = _mm512_load_si512(cache[19]);
		x20 = _mm512_load_si512(cache[20]);
		x21 = _mm512_load_si512(cache[21]);
		x22 = _mm512_load_si512(cache[22]);
		x23 = _mm512_load_si512(cache[23]);
		x24 = _mm512_load_si512(cache[24]);
		x25 = _mm512_load_si512(cache[25]);
		x26 = _mm512_load_si512(cache[26]);
		x27 = _mm512_load_si512(cache[27]);
		x28 = _mm512_load_si512(cache[28]);
		x29 = _mm512_load_si512(cache[29]);
		x30 = _mm512_load_si512(cache[30]);
		x31 = _mm512_load_si512(cache[31]);

		mm512_transpose32_epi16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,
		                        x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31);

		_mm512_store_si512(dst[0] + j, x0);
		_mm512_store_si512(dst[1] + j, x1);
		_mm512_store_si512(dst[2] + j, x2);
		_mm512_store_si512(dst[3] + j, x3);
		_mm512_store_si512(dst[4] + j, x4);
		_mm512_store_si512(dst[5] + j, x5);
		_mm512_store_si512(dst[6] + j, x6);
		_mm512_store_si512(dst[7] + j, x7);
		_mm512_store_si512(dst[8] + j, x8);
		_mm512_store_si512(dst[9] + j, x9);
		_mm512_store_si512(dst[10] + j, x10);
		_mm512_store_si512(dst[11] + j, x11);
		_mm512_store_si512(dst[12] + j, x12);
		_mm512_store_si512(dst[13] + j, x13);
		_mm512_store_si512(dst[14] + j, x14);
		_mm512_store_si512(dst[15] + j, x15);
		_mm512_store_si512(dst[16] + j, x16);
		_mm512_store_si512(dst[17] + j, x17);
		_mm512_store_si512(dst[18] + j, x18);
		_mm512_store_si512(dst[19] + j, x19);
		_mm512_store_si512(dst[20] + j, x20);
		_mm512_store_si512(dst[21] + j, x21);
		_mm512_store_si512(dst[22] + j, x22);
		_mm512_store_si512(dst[23] + j, x23);
		_mm512_store_si512(dst[24] + j, x24);
		_mm512_store_si512(dst[25] + j, x25);
		_mm512_store_si512(dst[26] + j, x26);
		_mm512_store_si512(dst[27] + j, x27);
		_mm512_store_si512(dst[28] + j, x28);
		_mm512_store_si512(dst[29] + j, x29);
		_mm512_store_si512(dst[30] + j, x30);
		_mm512_store_si512(dst[31] + j, x31);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		__m512i x = XITER(j, XARGS);

		mm_scatter_epi16(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, dst[4] + j, dst[5] + j, dst[6] + j, dst[7] + j, _mm512_castsi512_si128(x));
		mm_scatter_epi16(dst[8] + j, dst[9] + j, dst[10] + j, dst[11] + j, dst[12] + j, dst[13] + j, dst[14] + j, dst[15] + j, _mm512_extracti32x4_epi32(x, 1));
		mm_scatter_epi16(dst[16] + j, dst[17] + j, dst[18] + j, dst[19] + j, dst[20] + j, dst[21] + j, dst[22] + j, dst[23] + j, _mm512_extracti32x4_epi32(x, 2));
		mm_scatter_epi16(dst[24] + j, dst[25] + j, dst[26] + j, dst[27] + j, dst[28] + j, dst[29] + j, dst[30] + j, dst[31] + j, _mm512_extracti32x4_epi32(x, 3));
	}
#undef XITER
#undef XARGS
}

const decltype(&resize_line16_h_u16_avx512<false, 0>) resize_line16_h_u16_avx512_jt_small[] = {
	resize_line16_h_u16_avx512<false, 2>,
	resize_line16_h_u16_avx512<false, 2>,
	resize_line16_h_u16_avx512<false, 4>,
	resize_line16_h_u16_avx512<false, 4>,
	resize_line16_h_u16_avx512<false, 6>,
	resize_line16_h_u16_avx512<false, 6>,
	resize_line16_h_u16_avx512<false, 8>,
	resize_line16_h_u16_avx512<false, 8>,
};

const decltype(&resize_line16_h_u16_avx512<false, 0>) resize_line16_h_u16_avx512_jt_large[] = {
	resize_line16_h_u16_avx512<true, 0>,
	resize_line16_h_u16_avx512<true, 2>,
	resize_line16_h_u16_avx512<true, 2>,
	resize_line16_h_u16_avx512<true, 4>,
	resize_line16_h_u16_avx512<true, 4>,
	resize_line16_h_u16_avx512<true, 6>,
	resize_line16_h_u16_avx512<true, 6>,
	resize_line16_h_u16_avx512<true, 0>,
};



template <class Traits, unsigned FWidth, unsigned Tail>
inline FORCE_INLINE __m512 resize_line16_h_fp_avx512_xiter(unsigned j,
                                                           const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                                           const typename Traits::pixel_type * RESTRICT src, unsigned src_base)
{
	typedef typename Traits::pixel_type pixel_type;

	const float *filter_coeffs = filter_data + j * filter_stride;
	const pixel_type *src_p = src + (filter_left[j] - src_base) * 16;

	__m512 accum0 = _mm512_setzero_ps();
	__m512 accum1 = _mm512_setzero_ps();
	__m512 x, c, coeffs;

	unsigned k_end = FWidth ? FWidth - Tail : floor_n(filter_width, 4);

	for (unsigned k = 0; k < k_end; k += 4) {
		coeffs = _mm512_broadcast_f32x4(_mm_load_ps(filter_coeffs + k));

		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_PERM_AAAA);
		x = Traits::load16(src_p + 0);
		accum0 = _mm512_fmadd_ps(c, x, accum0);

		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_PERM_BBBB);
		x = Traits::load16(src_p + 16);
		accum1 = _mm512_fmadd_ps(c, x, accum1);

		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_PERM_CCCC);
		x = Traits::load16(src_p + 32);
		accum0 = _mm512_fmadd_ps(c, x, accum0);

		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_PERM_DDDD);
		x = Traits::load16(src_p + 48);
		accum1 = _mm512_fmadd_ps(c, x, accum1);

		src_p += 64;
	}

	if (Tail >= 1) {
		coeffs = _mm512_broadcast_f32x4(_mm_load_ps(filter_coeffs + k_end));

		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_PERM_AAAA);
		x = Traits::load16(src_p + 0);
		accum0 = _mm512_fmadd_ps(c, x, accum0);
	}
	if (Tail >= 2) {
		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_PERM_BBBB);
		x = Traits::load16(src_p + 16);
		accum1 = _mm512_fmadd_ps(c, x, accum1);
	}
	if (Tail >= 3) {
		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_PERM_CCCC);
		x = Traits::load16(src_p + 32);
		accum0 = _mm512_fmadd_ps(c, x, accum0);
	}
	if (Tail >= 4) {
		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_PERM_DDDD);
		x = Traits::load16(src_p + 48);
		accum1 = _mm512_fmadd_ps(c, x, accum1);
	}

	if (!FWidth || FWidth >= 2)
		accum0 = _mm512_add_ps(accum0, accum1);

	return accum0;
}

template <class Traits, unsigned FWidth, unsigned Tail>
void resize_line16_h_fp_avx512(const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                               const typename Traits::pixel_type * RESTRICT src, typename Traits::pixel_type * const * RESTRICT dst, unsigned src_base, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

#define XITER resize_line16_h_fp_avx512_xiter<Traits, FWidth, Tail>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src, src_base
	for (unsigned j = left; j < vec_left; ++j) {
		__m512 x = XITER(j, XARGS);
		Traits::scatter16(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, dst[4] + j, dst[5] + j, dst[6] + j, dst[7] + j,
		                  dst[8] + j, dst[9] + j, dst[10] + j, dst[11] + j, dst[12] + j, dst[13] + j, dst[14] + j, dst[15] + j, x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		float cache alignas(64)[16][16];
		__m512 x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

		for (unsigned jj = j; jj < j + 16; ++jj) {
			__m512 x = XITER(jj, XARGS);
			_mm512_store_ps(cache[jj - j], x);
		}

		x0 = _mm512_load_ps(cache[0]);
		x1 = _mm512_load_ps(cache[1]);
		x2 = _mm512_load_ps(cache[2]);
		x3 = _mm512_load_ps(cache[3]);
		x4 = _mm512_load_ps(cache[4]);
		x5 = _mm512_load_ps(cache[5]);
		x6 = _mm512_load_ps(cache[6]);
		x7 = _mm512_load_ps(cache[7]);
		x8 = _mm512_load_ps(cache[8]);
		x9 = _mm512_load_ps(cache[9]);
		x10 = _mm512_load_ps(cache[10]);
		x11 = _mm512_load_ps(cache[11]);
		x12 = _mm512_load_ps(cache[12]);
		x13 = _mm512_load_ps(cache[13]);
		x14 = _mm512_load_ps(cache[14]);
		x15 = _mm512_load_ps(cache[15]);

		mm512_transpose16_ps(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);

		Traits::store16(dst[0] + j, x0);
		Traits::store16(dst[1] + j, x1);
		Traits::store16(dst[2] + j, x2);
		Traits::store16(dst[3] + j, x3);
		Traits::store16(dst[4] + j, x4);
		Traits::store16(dst[5] + j, x5);
		Traits::store16(dst[6] + j, x6);
		Traits::store16(dst[7] + j, x7);
		Traits::store16(dst[8] + j, x8);
		Traits::store16(dst[9] + j, x9);
		Traits::store16(dst[10] + j, x10);
		Traits::store16(dst[11] + j, x11);
		Traits::store16(dst[12] + j, x12);
		Traits::store16(dst[13] + j, x13);
		Traits::store16(dst[14] + j, x14);
		Traits::store16(dst[15] + j, x15);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		__m512 x = XITER(j, XARGS);
		Traits::scatter16(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, dst[4] + j, dst[5] + j, dst[6] + j, dst[7] + j,
		                  dst[8] + j, dst[9] + j, dst[10] + j, dst[11] + j, dst[12] + j, dst[13] + j, dst[14] + j, dst[15] + j, x);
	}
#undef XITER
#undef XARGS
}

template <class Traits>
struct resize_line16_h_fp_avx512_jt {
	typedef decltype(&resize_line16_h_fp_avx512<Traits, 0, 0>) func_type;

	static const func_type small[8];
	static const func_type large[4];
};

template <class Traits>
const typename resize_line16_h_fp_avx512_jt<Traits>::func_type resize_line16_h_fp_avx512_jt<Traits>::small[8] = {
	resize_line16_h_fp_avx512<Traits, 1, 1>,
	resize_line16_h_fp_avx512<Traits, 2, 2>,
	resize_line16_h_fp_avx512<Traits, 3, 3>,
	resize_line16_h_fp_avx512<Traits, 4, 4>,
	resize_line16_h_fp_avx512<Traits, 5, 1>,
	resize_line16_h_fp_avx512<Traits, 6, 2>,
	resize_line16_h_fp_avx512<Traits, 7, 3>,
	resize_line16_h_fp_avx512<Traits, 8, 4>
};

template <class Traits>
const typename resize_line16_h_fp_avx512_jt<Traits>::func_type resize_line16_h_fp_avx512_jt<Traits>::large[4] = {
	resize_line16_h_fp_avx512<Traits, 0, 0>,
	resize_line16_h_fp_avx512<Traits, 0, 1>,
	resize_line16_h_fp_avx512<Traits, 0, 2>,
	resize_line16_h_fp_avx512<Traits, 0, 3>
};


template <unsigned N>
void resize_line_h_perm_u16_avx512(const unsigned * RESTRICT permute_left, const uint16_t * RESTRICT permute_mask, const int16_t * RESTRICT filter_data, unsigned input_width,
                                   const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, unsigned left, unsigned right, uint16_t limit)
{
	static_assert(N <= 16, "permuted resampler only supports up to 16 taps");

	const __m512i i16_min = _mm512_set1_epi16(INT16_MIN);
	const __m512i lim = _mm512_set1_epi16(limit + INT16_MIN);

	unsigned vec_right = floor_n(right, 16);
	unsigned fallback_idx = vec_right;

	for (unsigned j = floor_n(left, 16); j < vec_right; j += 16) {
		unsigned left = permute_left[j / 16];

		if (input_width - left < 64) {
			fallback_idx = j;
			break;
		}

		const __m512i mask = _mm512_load_si512(permute_mask + j * 2);
		const int16_t *data = filter_data + j * N;

		__m512i accum0 = _mm512_setzero_si512();
		__m512i accum1 = _mm512_setzero_si512();
		__m512i x, x0, x8, x16, coeffs;

		if (N >= 2) {
			x0 = _mm512_loadu_si512(src + left + 0);
			x0 = _mm512_add_epi16(x0, i16_min);

			x = x0;
			x =_mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 0 * 16);
			x = _mm512_madd_epi16(coeffs, x);
			accum0 = _mm512_add_epi32(accum0, x);
		}
		if (N >= 4) {
			x8 = _mm512_loadu_si512(src + left + 8);
			x8 = _mm512_add_epi16(x8, i16_min);

			x = _mm512_alignr_epi8(x8, x0, 4);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 2 * 16);
			x = _mm512_madd_epi16(coeffs, x);
			accum1 = _mm512_add_epi32(accum1, x);
		}
		if (N >= 6) {
			x = _mm512_alignr_epi8(x8, x0, 8);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 4 * 16);
			x = _mm512_madd_epi16(coeffs, x);
			accum0 = _mm512_add_epi32(accum0, x);
		}
		if (N >= 8) {
			x = _mm512_alignr_epi8(x8, x0, 12);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 6 * 16);
			x = _mm512_madd_epi16(coeffs, x);
			accum1 = _mm512_add_epi32(accum1, x);
		}
		if (N >= 10) {
			x = x8;
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 8 * 16);
			x = _mm512_madd_epi16(coeffs, x);
			accum0 = _mm512_add_epi32(accum0, x);
		}
		if (N >= 12) {
			x16 = _mm512_loadu_si512(src + left + 16);
			x16 = _mm512_add_epi16(x16, i16_min);

			x = _mm512_alignr_epi8(x16, x8, 4);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 10 * 16);
			x = _mm512_madd_epi16(coeffs, x);
			accum1 = _mm512_add_epi32(accum1, x);
		}
		if (N >= 14) {
			x = _mm512_alignr_epi8(x16, x8, 8);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 12 * 16);
			x = _mm512_madd_epi16(coeffs, x);
			accum0 = _mm512_add_epi32(accum0, x);
		}
		if (N >= 16) {
			x = _mm512_alignr_epi8(x16, x8, 12);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 14 * 16);
			x = _mm512_madd_epi16(coeffs, x);
			accum1 = _mm512_add_epi32(accum1, x);
		}

		accum0 = _mm512_add_epi32(accum0, accum1);

		__m256i out = export_i30_u16(accum0);
		out = _mm256_min_epi16(out, _mm512_castsi512_si256(lim));
		out = _mm256_sub_epi16(out, _mm512_castsi512_si256(i16_min));
		_mm256_store_si256((__m256i *)(dst + j), out);
	}
	for (unsigned j = fallback_idx; j < right; j += 16) {
		unsigned left = permute_left[j / 16];

		const __m512i mask = _mm512_load_si512(permute_mask + j * 2);
		const int16_t *data = filter_data + j * N;

		__m512i accum = _mm512_setzero_si512();
		__m512i x, coeffs;

		for (unsigned k = 0; k < N; k += 2) {
			unsigned num_load = std::min(input_width - left - k, 32U);
			__mmask32 load_mask = 0xFFFFFFFFU >> (32 - num_load);

			x = _mm512_maskz_loadu_epi16(load_mask, src + left + k);
			x = _mm512_permutexvar_epi16(mask, x);
			x = _mm512_add_epi16(x, i16_min);
			coeffs = _mm512_load_si512(data + k * 16);
			x = _mm512_madd_epi16(coeffs, x);
			accum = _mm512_add_epi32(accum, x);
		}

		__m256i out = export_i30_u16(accum);
		out = _mm256_min_epi16(out, _mm512_castsi512_si256(lim));
		out = _mm256_sub_epi16(out, _mm512_castsi512_si256(i16_min));
		_mm256_store_si256((__m256i *)(dst + j), out);
	}
}

struct resize_line_h_perm_u16_avx512_jt {
	typedef decltype(&resize_line_h_perm_u16_avx512<2>) func_type;
	static const func_type table[8];
};

const typename resize_line_h_perm_u16_avx512_jt::func_type resize_line_h_perm_u16_avx512_jt::table[8] = {
	resize_line_h_perm_u16_avx512<2>,
	resize_line_h_perm_u16_avx512<4>,
	resize_line_h_perm_u16_avx512<6>,
	resize_line_h_perm_u16_avx512<8>,
	resize_line_h_perm_u16_avx512<10>,
	resize_line_h_perm_u16_avx512<12>,
	resize_line_h_perm_u16_avx512<14>,
	resize_line_h_perm_u16_avx512<16>,
};


template <class Traits, unsigned N>
void resize_line_h_perm_fp_avx512(const unsigned * RESTRICT permute_left, const unsigned * RESTRICT permute_mask, const float * RESTRICT filter_data, unsigned input_width,
                                  const typename Traits::pixel_type * RESTRICT src, typename Traits::pixel_type * RESTRICT dst, unsigned left, unsigned right)
{
	static_assert(N <= 16, "permuted resampler only supports up to 16 taps");

	unsigned vec_right = floor_n(right, 16);
	unsigned fallback_idx = vec_right;

#define mm512_alignr_epi8_ps(a, b, imm) _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512((a)), _mm512_castps_si512((b)), (imm)))
	for (unsigned j = floor_n(left, 16); j < vec_right; j += 16) {
		unsigned left = permute_left[j / 16];

		if (input_width - left < 32) {
			fallback_idx = j;
			break;
		}

		const __m512i mask = _mm512_load_si512(permute_mask + j);
		const float *data = filter_data + j * N;

		__m512 accum0 = _mm512_setzero_ps();
		__m512 accum1 = _mm512_setzero_ps();
		__m512 x, x0, x4, x8, x12, x16, coeffs;

		if (N >= 1) {
			x0 = Traits::load16(src + left + 0);

			x = x0;
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 0 * 16);
			accum0 = _mm512_fmadd_ps(coeffs, x, accum0);
		}
		if (N >= 2) {
			x4 = Traits::load16(src + left + 4);

			x = mm512_alignr_epi8_ps(x4, x0, 4);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 1 * 16);
			accum1 = _mm512_fmadd_ps(coeffs, x, accum1);
		}
		if (N >= 3) {
			x = mm512_alignr_epi8_ps(x4, x0, 8);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 2 * 16);
			accum0 = _mm512_fmadd_ps(coeffs, x, accum0);
		}
		if (N >= 4) {
			x = mm512_alignr_epi8_ps(x4, x0, 12);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 3 * 16);
			accum1 = _mm512_fmadd_ps(coeffs, x, accum1);
		}
		if (N >= 5) {
			x = x4;
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 4 * 16);
			accum0 = _mm512_fmadd_ps(coeffs, x, accum0);
		}
		if (N >= 6) {
			x8 = Traits::load16(src + left + 8);

			x = mm512_alignr_epi8_ps(x8, x4, 4);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 5 * 16);
			accum1 = _mm512_fmadd_ps(coeffs, x, accum1);
		}
		if (N >= 7) {
			x = mm512_alignr_epi8_ps(x8, x4, 8);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 6 * 16);
			accum0 = _mm512_fmadd_ps(coeffs, x, accum0);
		}
		if (N >= 8) {
			x = mm512_alignr_epi8_ps(x8, x4, 12);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 7 * 16);
			accum1 = _mm512_fmadd_ps(coeffs, x, accum1);
		}
		if (N >= 9) {
			x = x8;
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 8 * 16);
			accum0 = _mm512_fmadd_ps(coeffs, x, accum0);
		}
		if (N >= 10) {
			x12 = Traits::load16(src + left + 12);

			x = mm512_alignr_epi8_ps(x12, x8, 4);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 9 * 16);
			accum1 = _mm512_fmadd_ps(coeffs, x, accum1);
		}
		if (N >= 11) {
			x = mm512_alignr_epi8_ps(x12, x8, 8);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 10 * 16);
			accum0 = _mm512_fmadd_ps(coeffs, x, accum0);
		}
		if (N >= 12) {
			x = mm512_alignr_epi8_ps(x12, x8, 12);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 11 * 16);
			accum1 = _mm512_fmadd_ps(coeffs, x, accum1);
		}
		if (N >= 13) {
			x = x12;
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 12 * 16);
			accum0 = _mm512_fmadd_ps(coeffs, x, accum0);
		}
		if (N >= 14) {
			x16 = Traits::load16(src + left + 16);

			x = mm512_alignr_epi8_ps(x16, x12, 4);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 13 * 16);
			accum1 = _mm512_fmadd_ps(coeffs, x, accum1);
		}
		if (N >= 15) {
			x = mm512_alignr_epi8_ps(x16, x12, 8);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 14 * 16);
			accum0 = _mm512_fmadd_ps(coeffs, x, accum0);
		}
		if (N >= 16) {
			x = mm512_alignr_epi8_ps(x16, x12, 12);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + 15 * 16);
			accum1 = _mm512_fmadd_ps(coeffs, x, accum1);
		}

		accum0 = _mm512_add_ps(accum0, accum1);
		Traits::store16(dst + j, accum0);
	}
#undef mm512_alignr_epi8_ps
	for (unsigned j = fallback_idx; j < right; j += 16) {
		unsigned left = permute_left[j / 16];

		const __m512i mask = _mm512_load_si512(permute_mask + j);
		const float *data = filter_data + j * N;

		__m512 accum0 = _mm512_setzero_ps();
		__m512 accum1 = _mm512_setzero_ps();
		__m512 x, coeffs;

		for (unsigned k = 0; k < N; ++k) {
			unsigned num_load = std::min(input_width - left - k, 16U);
			__mmask16 load_mask = 0xFFFFU >> (16 - num_load);

			x = Traits::maskz_load16(load_mask, src + left + k);
			x = _mm512_permutexvar_ps(mask, x);
			coeffs = _mm512_load_ps(data + k * 16);

			if (k % 2)
				accum1 = _mm512_fmadd_ps(coeffs, x, accum1);
			else
				accum0 = _mm512_fmadd_ps(coeffs, x, accum0);
		}
		accum0 = _mm512_add_ps(accum0, accum1);
		Traits::store16(dst + j, accum0);
	}
}

template <class Traits>
struct resize_line_h_perm_fp_avx512_jt {
	typedef decltype(&resize_line_h_perm_fp_avx512<Traits, 1>) func_type;
	static const func_type table[16];
};

template <class Traits>
const typename resize_line_h_perm_fp_avx512_jt<Traits>::func_type resize_line_h_perm_fp_avx512_jt<Traits>::table[16] = {
	resize_line_h_perm_fp_avx512<Traits, 1>,
	resize_line_h_perm_fp_avx512<Traits, 2>,
	resize_line_h_perm_fp_avx512<Traits, 3>,
	resize_line_h_perm_fp_avx512<Traits, 4>,
	resize_line_h_perm_fp_avx512<Traits, 5>,
	resize_line_h_perm_fp_avx512<Traits, 6>,
	resize_line_h_perm_fp_avx512<Traits, 7>,
	resize_line_h_perm_fp_avx512<Traits, 8>,
	resize_line_h_perm_fp_avx512<Traits, 9>,
	resize_line_h_perm_fp_avx512<Traits, 10>,
	resize_line_h_perm_fp_avx512<Traits, 11>,
	resize_line_h_perm_fp_avx512<Traits, 12>,
	resize_line_h_perm_fp_avx512<Traits, 13>,
	resize_line_h_perm_fp_avx512<Traits, 14>,
	resize_line_h_perm_fp_avx512<Traits, 15>,
	resize_line_h_perm_fp_avx512<Traits, 16>,
};


template <unsigned N, bool ReadAccum, bool WriteToAccum>
inline FORCE_INLINE __m512i resize_line_v_u16_avx512_xiter(unsigned j, unsigned accum_base,
                                                           const uint16_t *src_p0, const uint16_t *src_p1, const uint16_t *src_p2, const uint16_t *src_p3,
                                                           const uint16_t *src_p4, const uint16_t *src_p5, const uint16_t *src_p6, const uint16_t *src_p7,
                                                           uint32_t * RESTRICT accum_p, const __m512i &c01, const __m512i &c23, const __m512i &c45, const __m512i &c67, uint16_t limit)
{
	const __m512i i16_min = _mm512_set1_epi16(INT16_MIN);
	const __m512i lim = _mm512_set1_epi16(limit + INT16_MIN);

	__m512i accum_lo = _mm512_setzero_si512();
	__m512i accum_hi = _mm512_setzero_si512();
	__m512i x0, x1, xl, xh;

	if (N >= 0) {
		x0 = _mm512_load_si512(src_p0 + j);
		x1 = _mm512_load_si512(src_p1 + j);
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c01, xl);
		xh = _mm512_madd_epi16(c01, xh);

		if (ReadAccum) {
			accum_lo = _mm512_add_epi32(_mm512_load_si512(accum_p + j - accum_base + 0), xl);
			accum_hi = _mm512_add_epi32(_mm512_load_si512(accum_p + j - accum_base + 16), xh);
		} else {
			accum_lo = xl;
			accum_hi = xh;
		}
	}
	if (N >= 2) {
		x0 = _mm512_load_si512(src_p2 + j);
		x1 = _mm512_load_si512(src_p3 + j);
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c23, xl);
		xh = _mm512_madd_epi16(c23, xh);

		accum_lo = _mm512_add_epi32(accum_lo, xl);
		accum_hi = _mm512_add_epi32(accum_hi, xh);
	}
	if (N >= 4) {
		x0 = _mm512_load_si512(src_p4 + j);
		x1 = _mm512_load_si512(src_p5 + j);
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c45, xl);
		xh = _mm512_madd_epi16(c45, xh);

		accum_lo = _mm512_add_epi32(accum_lo, xl);
		accum_hi = _mm512_add_epi32(accum_hi, xh);
	}
	if (N >= 6) {
		x0 = _mm512_load_si512(src_p6 + j);
		x1 = _mm512_load_si512(src_p7 + j);
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		xl = _mm512_madd_epi16(c67, xl);
		xh = _mm512_madd_epi16(c67, xh);

		accum_lo = _mm512_add_epi32(accum_lo, xl);
		accum_hi = _mm512_add_epi32(accum_hi, xh);
	}

	if (WriteToAccum) {
		_mm512_store_si512(accum_p + j - accum_base + 0, accum_lo);
		_mm512_store_si512(accum_p + j - accum_base + 16, accum_hi);
		return _mm512_setzero_si512();
	} else {
		accum_lo = export2_i30_u16(accum_lo, accum_hi);
		accum_lo = _mm512_min_epi16(accum_lo, lim);
		accum_lo = _mm512_sub_epi16(accum_lo, i16_min);

		return accum_lo;
	}
}

template <unsigned N, bool ReadAccum, bool WriteToAccum>
void resize_line_v_u16_avx512(const int16_t * RESTRICT filter_data, const uint16_t * const * RESTRICT src, uint16_t * RESTRICT dst, uint32_t * RESTRICT accum,
                              unsigned left, unsigned right, uint16_t limit)
{
	const uint16_t *src_p0 = src[0];
	const uint16_t *src_p1 = src[1];
	const uint16_t *src_p2 = src[2];
	const uint16_t *src_p3 = src[3];
	const uint16_t *src_p4 = src[4];
	const uint16_t *src_p5 = src[5];
	const uint16_t *src_p6 = src[6];
	const uint16_t *src_p7 = src[7];

	unsigned vec_left = ceil_n(left, 32);
	unsigned vec_right = floor_n(right, 32);
	unsigned accum_base = floor_n(left, 32);

	const __m512i c01 = _mm512_unpacklo_epi16(_mm512_set1_epi16(filter_data[0]), _mm512_set1_epi16(filter_data[1]));
	const __m512i c23 = _mm512_unpacklo_epi16(_mm512_set1_epi16(filter_data[2]), _mm512_set1_epi16(filter_data[3]));
	const __m512i c45 = _mm512_unpacklo_epi16(_mm512_set1_epi16(filter_data[4]), _mm512_set1_epi16(filter_data[5]));
	const __m512i c67 = _mm512_unpacklo_epi16(_mm512_set1_epi16(filter_data[6]), _mm512_set1_epi16(filter_data[7]));

	__m512i out;

#define XITER resize_line_v_u16_avx512_xiter<N, ReadAccum, WriteToAccum>
#define XARGS accum_base, src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, accum, c01, c23, c45, c67, limit
	if (left != vec_left) {
		out = XITER(vec_left - 32, XARGS);

		if (!WriteToAccum)
			_mm512_mask_storeu_epi16(dst + vec_left - 32, mmask32_set_hi(vec_left - left), out);
	}

	for (unsigned j = vec_left; j < vec_right; j += 32) {
		out = XITER(j, XARGS);

		if (!WriteToAccum)
			_mm512_store_si512(dst + j, out);
	}

	if (right != vec_right) {
		out = XITER(vec_right, XARGS);

		if (!WriteToAccum)
			_mm512_mask_storeu_epi16(dst + vec_right, mmask32_set_lo(right - vec_right), out);
	}
#undef XITER
#undef XARGS
}

const decltype(&resize_line_v_u16_avx512<0, false, false>) resize_line_v_u16_avx512_jt_a[] = {
	resize_line_v_u16_avx512<0, false, false>,
	resize_line_v_u16_avx512<0, false, false>,
	resize_line_v_u16_avx512<2, false, false>,
	resize_line_v_u16_avx512<2, false, false>,
	resize_line_v_u16_avx512<4, false, false>,
	resize_line_v_u16_avx512<4, false, false>,
	resize_line_v_u16_avx512<6, false, false>,
	resize_line_v_u16_avx512<6, false, false>,
};

const decltype(&resize_line_v_u16_avx512<0, false, false>) resize_line_v_u16_avx512_jt_b[] = {
	resize_line_v_u16_avx512<0, true, false>,
	resize_line_v_u16_avx512<0, true, false>,
	resize_line_v_u16_avx512<2, true, false>,
	resize_line_v_u16_avx512<2, true, false>,
	resize_line_v_u16_avx512<4, true, false>,
	resize_line_v_u16_avx512<4, true, false>,
	resize_line_v_u16_avx512<6, true, false>,
	resize_line_v_u16_avx512<6, true, false>,
};


template <class Traits, unsigned N, bool UpdateAccum, class T = typename Traits::pixel_type>
inline FORCE_INLINE __m512 resize_line_v_fp_avx512_xiter(unsigned j,
                                                         const T *src_p0, const T *src_p1, const T *src_p2, const T *src_p3,
                                                         const T *src_p4, const T *src_p5, const T *src_p6, const T *src_p7, T * RESTRICT accum_p,
                                                         const __m512 &c0, const __m512 &c1, const __m512 &c2, const __m512 &c3,
                                                         const __m512 &c4, const __m512 &c5, const __m512 &c6, const __m512 &c7)
{
	typedef typename Traits::pixel_type pixel_type;
	static_assert(std::is_same<pixel_type, T>::value, "must not specify T");

	__m512 accum0 = _mm512_setzero_ps();
	__m512 accum1 = _mm512_setzero_ps();
	__m512 x;

	if (N >= 0) {
		x = Traits::load16(src_p0 + j);
		accum0 = UpdateAccum ? _mm512_fmadd_ps(c0, x, Traits::load16(accum_p + j)) : _mm512_mul_ps(c0, x);
	}
	if (N >= 1) {
		x = Traits::load16(src_p1 + j);
		accum1 = _mm512_mul_ps(c1, x);
	}
	if (N >= 2) {
		x = Traits::load16(src_p2 + j);
		accum0 = _mm512_fmadd_ps(c2, x, accum0);
	}
	if (N >= 3) {
		x = Traits::load16(src_p3 + j);
		accum1 = _mm512_fmadd_ps(c3, x, accum1);
	}
	if (N >= 4) {
		x = Traits::load16(src_p4 + j);
		accum0 = _mm512_fmadd_ps(c4, x, accum0);
	}
	if (N >= 5) {
		x = Traits::load16(src_p5 + j);
		accum1 = _mm512_fmadd_ps(c5, x, accum1);
	}
	if (N >= 6) {
		x = Traits::load16(src_p6 + j);
		accum0 = _mm512_fmadd_ps(c6, x, accum0);
	}
	if (N >= 7) {
		x = Traits::load16(src_p7 + j);
		accum1 = _mm512_fmadd_ps(c7, x, accum1);
	}

	accum0 = (N >= 1) ? _mm512_add_ps(accum0, accum1) : accum0;
	return accum0;
}

template <class Traits, unsigned N, bool UpdateAccum>
void resize_line_v_fp_avx512(const float * RESTRICT filter_data, const typename Traits::pixel_type * const * RESTRICT src, typename Traits::pixel_type * RESTRICT dst,
                             unsigned left, unsigned right)
{
	typedef typename Traits::pixel_type pixel_type;

	const pixel_type *src_p0 = src[0];
	const pixel_type *src_p1 = src[1];
	const pixel_type *src_p2 = src[2];
	const pixel_type *src_p3 = src[3];
	const pixel_type *src_p4 = src[4];
	const pixel_type *src_p5 = src[5];
	const pixel_type *src_p6 = src[6];
	const pixel_type *src_p7 = src[7];

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m512 c0 = _mm512_broadcastss_ps(_mm_load_ss(filter_data + 0));
	const __m512 c1 = _mm512_broadcastss_ps(_mm_load_ss(filter_data + 1));
	const __m512 c2 = _mm512_broadcastss_ps(_mm_load_ss(filter_data + 2));
	const __m512 c3 = _mm512_broadcastss_ps(_mm_load_ss(filter_data + 3));
	const __m512 c4 = _mm512_broadcastss_ps(_mm_load_ss(filter_data + 4));
	const __m512 c5 = _mm512_broadcastss_ps(_mm_load_ss(filter_data + 5));
	const __m512 c6 = _mm512_broadcastss_ps(_mm_load_ss(filter_data + 6));
	const __m512 c7 = _mm512_broadcastss_ps(_mm_load_ss(filter_data + 7));

	__m512 accum;

#define XITER resize_line_v_fp_avx512_xiter<Traits, N, UpdateAccum>
#define XARGS src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, dst, c0, c1, c2, c3, c4, c5, c6, c7
	if (left != vec_left) {
		accum = XITER(vec_left - 16, XARGS);
		Traits::mask_store16(dst + vec_left - 16, mmask16_set_hi(vec_left - left), accum);
	}
	for (unsigned j = vec_left; j < vec_right; j += 16) {
		accum = XITER(j, XARGS);
		Traits::mask_store16(dst + j, 0xFFFFU, accum);
	}
	if (right != vec_right) {
		accum = XITER(vec_right, XARGS);
		Traits::mask_store16(dst + vec_right, mmask16_set_lo(right - vec_right), accum);
	}
#undef XITER
#undef XARGS
}

template <class Traits>
struct resize_line_v_fp_avx512_jt {
	typedef decltype(&resize_line_v_fp_avx512<Traits, 0, false>) func_type;

	static const func_type table_a[8];
	static const func_type table_b[8];
};

template <class Traits>
const typename resize_line_v_fp_avx512_jt<Traits>::func_type resize_line_v_fp_avx512_jt<Traits>::table_a[8] = {
	resize_line_v_fp_avx512<Traits, 0, false>,
	resize_line_v_fp_avx512<Traits, 1, false>,
	resize_line_v_fp_avx512<Traits, 2, false>,
	resize_line_v_fp_avx512<Traits, 3, false>,
	resize_line_v_fp_avx512<Traits, 4, false>,
	resize_line_v_fp_avx512<Traits, 5, false>,
	resize_line_v_fp_avx512<Traits, 6, false>,
	resize_line_v_fp_avx512<Traits, 7, false>,
};

template <class Traits>
const typename resize_line_v_fp_avx512_jt<Traits>::func_type resize_line_v_fp_avx512_jt<Traits>::table_b[8] = {
	resize_line_v_fp_avx512<Traits, 0, true>,
	resize_line_v_fp_avx512<Traits, 1, true>,
	resize_line_v_fp_avx512<Traits, 2, true>,
	resize_line_v_fp_avx512<Traits, 3, true>,
	resize_line_v_fp_avx512<Traits, 4, true>,
	resize_line_v_fp_avx512<Traits, 5, true>,
	resize_line_v_fp_avx512<Traits, 6, true>,
	resize_line_v_fp_avx512<Traits, 7, true>,
};


inline FORCE_INLINE void calculate_line_address(void *dst, const void *src, ptrdiff_t stride, unsigned mask, unsigned i, unsigned height)
{
	__m512i idx = _mm512_set1_epi64(i);
	__m512i m = _mm512_set1_epi64(mask);
	__m512i p = _mm512_set1_epi64(reinterpret_cast<intptr_t>(src));

	idx = _mm512_add_epi64(idx, _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0));
	idx = _mm512_min_epi64(idx, _mm512_set1_epi64(height - 1));
	idx = _mm512_and_epi64(idx, m);
	idx = _mm512_mullo_epi64(idx, _mm512_set1_epi64(stride));

	p = _mm512_add_epi64(p, idx);
#if defined(_M_X64) || defined(__x86_64__)
	_mm512_store_si512(dst, p);
#else
	_mm256_store_si256((__m256i *)dst, _mm512_cvtepi64_epi32(p));
#endif
}


class ResizeImplH_U16_AVX512 final : public ResizeImplH {
	decltype(&resize_line16_h_u16_avx512<false, 0>) m_func;
	uint16_t m_pixel_max;
public:
	ResizeImplH_U16_AVX512(const FilterContext &filter, unsigned height, unsigned depth) :
		ResizeImplH(filter, image_attributes{ filter.filter_rows, height, PixelType::WORD }),
		m_func{},
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{
		if (filter.filter_width > 8)
			m_func = resize_line16_h_u16_avx512_jt_large[filter.filter_width % 8];
		else
			m_func = resize_line16_h_u16_avx512_jt_small[filter.filter_width - 1];
	}

	unsigned get_simultaneous_lines() const override { return 32; }

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		auto range = get_required_col_range(left, right);

		try {
			checked_size_t size = (static_cast<checked_size_t>(range.second) - floor_n(range.first, 32) + 32) * sizeof(uint16_t) * 32;
			return size.get();
		} catch (const std::overflow_error &) {
			error::throw_<error::OutOfMemory>();
		}
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		auto range = get_required_col_range(left, right);

		alignas(64) const uint16_t *src_ptr[32];
		alignas(64) uint16_t *dst_ptr[32];
		uint16_t *transpose_buf = static_cast<uint16_t *>(tmp);
		unsigned height = get_image_attributes().height;

		calculate_line_address(src_ptr + 0, src->data(), src->stride(), src->mask(), i + 0, height);
		calculate_line_address(src_ptr + 8, src->data(), src->stride(), src->mask(), i + std::min(8U, height - i - 1), height);
		calculate_line_address(src_ptr + 16, src->data(), src->stride(), src->mask(), i + std::min(16U, height - i - 1), height);
		calculate_line_address(src_ptr + 24, src->data(), src->stride(), src->mask(), i + std::min(24U, height - i - 1), height);

		transpose_line_32x32_epi16(transpose_buf, src_ptr, floor_n(range.first, 32), ceil_n(range.second, 32));

		calculate_line_address(dst_ptr + 0, dst->data(), dst->stride(), dst->mask(), i + 0, height);
		calculate_line_address(dst_ptr + 8, dst->data(), dst->stride(), dst->mask(), i + std::min(8U, height - i - 1), height);
		calculate_line_address(dst_ptr + 16, dst->data(), dst->stride(), dst->mask(), i + std::min(16U, height - i - 1), height);
		calculate_line_address(dst_ptr + 24, dst->data(), dst->stride(), dst->mask(), i + std::min(24U, height - i - 1), height);

		m_func(m_filter.left.data(), m_filter.data_i16.data(), m_filter.stride_i16, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 32), left, right, m_pixel_max);
	}
};

template <class Traits>
class ResizeImplH_FP_AVX512 final : public ResizeImplH {
	typedef typename Traits::pixel_type pixel_type;
	typedef typename resize_line16_h_fp_avx512_jt<Traits>::func_type func_type;

	func_type m_func;
public:
	ResizeImplH_FP_AVX512(const FilterContext &filter, unsigned height) :
		ResizeImplH(filter, image_attributes{ filter.filter_rows, height, Traits::type_constant }),
		m_func{}
	{
		if (filter.filter_width <= 8)
			m_func = resize_line16_h_fp_avx512_jt<Traits>::small[filter.filter_width - 1];
		else
			m_func = resize_line16_h_fp_avx512_jt<Traits>::large[filter.filter_width % 4];
	}

	unsigned get_simultaneous_lines() const override { return 16; }

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		auto range = get_required_col_range(left, right);

		try {
			checked_size_t size = (static_cast<checked_size_t>(range.second) - floor_n(range.first, 16) + 16) * sizeof(pixel_type) * 16;
			return size.get();
		} catch (const std::overflow_error &) {
			error::throw_<error::OutOfMemory>();
		}
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		auto range = get_required_col_range(left, right);

		alignas(64) const pixel_type *src_ptr[16];
		alignas(64) pixel_type *dst_ptr[16];
		pixel_type *transpose_buf = static_cast<pixel_type *>(tmp);
		unsigned height = get_image_attributes().height;

		calculate_line_address(src_ptr + 0, src->data(), src->stride(), src->mask(), i + 0, height);
		calculate_line_address(src_ptr + 8, src->data(), src->stride(), src->mask(), i + std::min(8U, height - i - 1), height);

		transpose_line_16x16<Traits>(transpose_buf, src_ptr, floor_n(range.first, 16), ceil_n(range.second, 16));

		calculate_line_address(dst_ptr + 0, dst->data(), dst->stride(), dst->mask(), i + 0, height);
		calculate_line_address(dst_ptr + 8, dst->data(), dst->stride(), dst->mask(), i + std::min(8U, height - i - 1), height);

		m_func(m_filter.left.data(), m_filter.data.data(), m_filter.stride, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 16), left, right);
	}
};

class ResizeImplH_Permute_U16_AVX512 final : public graph::ImageFilterBase {
	typedef typename resize_line_h_perm_u16_avx512_jt::func_type func_type;

	struct PermuteContext {
		AlignedVector<unsigned> left;
		AlignedVector<uint16_t> permute;
		AlignedVector<int16_t> data;
		unsigned filter_rows;
		unsigned filter_width;
		unsigned input_width;
	};

	PermuteContext m_context;
	unsigned m_height;
	uint16_t m_pixel_max;
	bool m_is_sorted;

	func_type m_func;

	ResizeImplH_Permute_U16_AVX512(PermuteContext context, unsigned height, unsigned depth) :
		m_context(std::move(context)),
		m_height{ height },
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) },
		m_is_sorted{ std::is_sorted(m_context.left.begin(), m_context.left.end()) },
		m_func{ resize_line_h_perm_u16_avx512_jt::table[(m_context.filter_width - 1) / 2] }
	{}
public:
	static std::unique_ptr<graph::ImageFilter> create(const FilterContext &filter, unsigned height, unsigned depth)
	{
		// Transpose is faster for large filters.
		if (filter.filter_width > 16)
			return nullptr;

		PermuteContext context{};

		unsigned filter_width = ceil_n(filter.filter_width, 2);

		context.left.resize(ceil_n(filter.filter_rows, 16) / 16);
		context.permute.resize(ceil_n(filter.filter_rows, 16) * 2);
		context.data.resize(ceil_n(filter.filter_rows, 16) * filter_width);
		context.filter_rows = filter.filter_rows;
		context.filter_width = filter_width;
		context.input_width = filter.input_width;

		for (unsigned i = 0; i < filter.filter_rows; i += 16) {
			unsigned left_min = UINT_MAX;
			unsigned left_max = 0U;

			for (unsigned ii = i; ii < std::min(i + 16, context.filter_rows); ++ii) {
				left_min = std::min(left_min, filter.left[ii]);
				left_max = std::max(left_max, filter.left[ii]);
			}
			if (left_max - left_min >= 32)
				return nullptr;

			for (unsigned ii = i; ii < std::min(i + 16, context.filter_rows); ++ii) {
				context.permute[ii * 2 + 0] = filter.left[ii] - left_min;
				context.permute[ii * 2 + 1] = context.permute[ii * 2 + 0] + 1;
			}
			context.left[i / 16] = left_min;

			int16_t *data = context.data.data() + i * context.filter_width;
			for (unsigned k = 0; k < context.filter_width; k += 2) {
				for (unsigned ii = i; ii < std::min(i + 16, context.filter_rows); ++ii) {
					data[static_cast<size_t>(k / 2) * 32 + (ii - i) * 2 + 0] = filter.data_i16[ii * static_cast<ptrdiff_t>(filter.stride_i16) + k + 0];
					data[static_cast<size_t>(k / 2) * 32 + (ii - i) * 2 + 1] = filter.data_i16[ii * static_cast<ptrdiff_t>(filter.stride_i16) + k + 1];
				}
			}
		}

		std::unique_ptr<graph::ImageFilter> ret{ new ResizeImplH_Permute_U16_AVX512(std::move(context), height, depth) };
		return ret;
	}

	filter_flags get_flags() const override
	{
		filter_flags flags{};

		flags.same_row = true;
		flags.entire_row = !m_is_sorted;

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_context.filter_rows, m_height, PixelType::WORD };
	}

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override
	{
		if (m_is_sorted) {
			unsigned input_width = m_context.input_width;
			unsigned right_base = m_context.left[(right + 15) / 16 - 1];
			unsigned iter_width = m_context.filter_width + 32;

			return{ m_context.left[left / 16],  right_base + std::min(input_width - right_base, iter_width) };
		} else {
			return{ 0, m_context.input_width };
		}
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		const auto &src_buf = graph::static_buffer_cast<const uint16_t>(*src);
		const auto &dst_buf = graph::static_buffer_cast<uint16_t>(*dst);

		m_func(m_context.left.data(), m_context.permute.data(), m_context.data.data(), m_context.input_width, src_buf[i], dst_buf[i], left, right, m_pixel_max);
	}
};

template <class Traits>
class ResizeImplH_Permute_FP_AVX512 final : public graph::ImageFilterBase {
	typedef typename Traits::pixel_type pixel_type;
	typedef typename resize_line_h_perm_fp_avx512_jt<Traits>::func_type func_type;

	struct PermuteContext {
		AlignedVector<unsigned> left;
		AlignedVector<unsigned> permute;
		AlignedVector<float> data;
		unsigned filter_rows;
		unsigned filter_width;
		unsigned input_width;
	};

	PermuteContext m_context;
	unsigned m_height;
	bool m_is_sorted;

	func_type m_func;

	ResizeImplH_Permute_FP_AVX512(PermuteContext context, unsigned height) :
		m_context(std::move(context)),
		m_height{ height },
		m_is_sorted{ std::is_sorted(m_context.left.begin(), m_context.left.end()) },
		m_func{ resize_line_h_perm_fp_avx512_jt<Traits>::table[m_context.filter_width - 1] }
	{}
public:
	static std::unique_ptr<graph::ImageFilter> create(const FilterContext &filter, unsigned height)
	{
		// Transpose is faster for large filters.
		if (filter.filter_width > 16)
			return nullptr;

		PermuteContext context{};

		context.left.resize(ceil_n(filter.filter_rows, 16) / 16);
		context.permute.resize(ceil_n(filter.filter_rows, 16));
		context.data.resize(ceil_n(filter.filter_rows, 16) * filter.filter_width);
		context.filter_rows = filter.filter_rows;
		context.filter_width = filter.filter_width;
		context.input_width = filter.input_width;

		for (unsigned i = 0; i < filter.filter_rows; i += 16) {
			unsigned left_min = UINT_MAX;
			unsigned left_max = 0U;

			for (unsigned ii = i; ii < std::min(i + 16, context.filter_rows); ++ii) {
				left_min = std::min(left_min, filter.left[ii]);
				left_max = std::max(left_max, filter.left[ii]);
			}
			if (left_max - left_min >= 16)
				return nullptr;

			for (unsigned ii = i; ii < std::min(i + 16, context.filter_rows); ++ii) {
				context.permute[ii] = filter.left[ii] - left_min;
			}
			context.left[i / 16] = left_min;

			float *data = context.data.data() + i * context.filter_width;
			for (unsigned k = 0; k < context.filter_width; ++k) {
				for (unsigned ii = i; ii < std::min(i + 16, context.filter_rows); ++ii) {
					data[static_cast<size_t>(k) * 16 + (ii - i)] = filter.data[ii * static_cast<ptrdiff_t>(filter.stride) + k];
				}
			}
		}

		std::unique_ptr<graph::ImageFilter> ret{ new ResizeImplH_Permute_FP_AVX512(std::move(context), height) };
		return ret;
	}

	filter_flags get_flags() const override
	{
		filter_flags flags{};

		flags.same_row = true;
		flags.entire_row = !m_is_sorted;

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_context.filter_rows, m_height, Traits::type_constant };
	}

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override
	{
		if (m_is_sorted) {
			unsigned input_width = m_context.input_width;
			unsigned right_base = m_context.left[(right + 15) / 16 - 1];
			unsigned iter_width = m_context.filter_width + 16;

			return{ m_context.left[left / 16],  right_base + std::min(input_width - right_base, iter_width) };
		} else {
			return{ 0, m_context.input_width };
		}
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		const auto &src_buf = graph::static_buffer_cast<const pixel_type>(*src);
		const auto &dst_buf = graph::static_buffer_cast<pixel_type>(*dst);

		m_func(m_context.left.data(), m_context.permute.data(), m_context.data.data(), m_context.input_width, src_buf[i], dst_buf[i], left, right);
	}
};

class ResizeImplV_U16_AVX512 final : public ResizeImplV {
	uint16_t m_pixel_max;
public:
	ResizeImplV_U16_AVX512(const FilterContext &filter, unsigned width, unsigned depth) :
		ResizeImplV(filter, image_attributes{ width, filter.filter_rows, PixelType::WORD }),
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		checked_size_t size = 0;

		try {
			if (m_filter.filter_width > 8)
				size += (ceil_n(checked_size_t{ right }, 32) - floor_n(left, 32)) * sizeof(uint32_t);
		} catch (const std::overflow_error &) {
			error::throw_<error::OutOfMemory>();
		}

		return size.get();
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		const auto &dst_buf = graph::static_buffer_cast<uint16_t>(*dst);

		const int16_t *filter_data = m_filter.data_i16.data() + i * m_filter.stride_i16;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		alignas(64) const uint16_t *src_lines[8];
		uint16_t *dst_line = dst_buf[i];
		uint32_t *accum_buf = static_cast<uint32_t *>(tmp);

		unsigned top = m_filter.left[i];

		if (filter_width <= 8) {
			calculate_line_address(src_lines, src->data(), src->stride(), src->mask(), top + 0, src_height);
			resize_line_v_u16_avx512_jt_a[filter_width - 1](filter_data, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
		} else {
			unsigned k_end = ceil_n(filter_width, 8) - 8;

			calculate_line_address(src_lines, src->data(), src->stride(), src->mask(), top + 0, src_height);
			resize_line_v_u16_avx512<6, false, true>(filter_data + 0, src_lines, dst_line, accum_buf, left, right, m_pixel_max);

			for (unsigned k = 8; k < k_end; k += 8) {
				calculate_line_address(src_lines, src->data(), src->stride(), src->mask(), top + k, src_height);
				resize_line_v_u16_avx512<6, true, true>(filter_data + k, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
			}

			calculate_line_address(src_lines, src->data(), src->stride(), src->mask(), top + k_end, src_height);
			resize_line_v_u16_avx512_jt_b[filter_width - k_end - 1](filter_data + k_end, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
		}
	}
};

template <class Traits>
class ResizeImplV_FP_AVX512 final : public ResizeImplV {
	typedef typename Traits::pixel_type pixel_type;
public:
	ResizeImplV_FP_AVX512(const FilterContext &filter, unsigned width) :
		ResizeImplV(filter, image_attributes{ width, filter.filter_rows, Traits::type_constant })
	{}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		const auto &dst_buf = graph::static_buffer_cast<pixel_type>(*dst);

		const float *filter_data = m_filter.data.data() + i * m_filter.stride;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		alignas(64) const pixel_type *src_lines[8];
		pixel_type *dst_line = dst_buf[i];

		{
			unsigned taps_remain = std::min(filter_width - 0, 8U);
			unsigned top = m_filter.left[i] + 0;

			calculate_line_address(src_lines, src->data(), src->stride(), src->mask(), top, src_height);
			resize_line_v_fp_avx512_jt<Traits>::table_a[taps_remain - 1](filter_data + 0, src_lines, dst_line, left, right);
		}

		for (unsigned k = 8; k < filter_width; k += 8) {
			unsigned taps_remain = std::min(filter_width - k, 8U);
			unsigned top = m_filter.left[i] + k;

			calculate_line_address(src_lines, src->data(), src->stride(), src->mask(), top, src_height);
			resize_line_v_fp_avx512_jt<Traits>::table_b[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
		}
	}
};

} // namespace


std::unique_ptr<graph::ImageFilter> create_resize_impl_h_avx512(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

#ifndef ZIMG_RESIZE_NO_PERMUTE
	if (type == PixelType::WORD)
		ret = ResizeImplH_Permute_U16_AVX512::create(context, height, depth);
	else if (type == PixelType::HALF)
		ret = ResizeImplH_Permute_FP_AVX512<f16_traits>::create(context, height);
	else if (type == PixelType::FLOAT)
		ret = ResizeImplH_Permute_FP_AVX512<f32_traits>::create(context, height);
#endif

	if (!ret) {
		if (type == PixelType::WORD)
			ret = ztd::make_unique<ResizeImplH_U16_AVX512>(context, height, depth);
		else if (type == PixelType::HALF)
			ret = ztd::make_unique<ResizeImplH_FP_AVX512<f16_traits>>(context, height);
		else if (type == PixelType::FLOAT)
			ret = ztd::make_unique<ResizeImplH_FP_AVX512<f32_traits>>(context, height);
	}

	return ret;
}

std::unique_ptr<graph::ImageFilter> create_resize_impl_v_avx512(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == PixelType::WORD)
		ret = ztd::make_unique<ResizeImplV_U16_AVX512>(context, width, depth);
	else if (type == PixelType::HALF)
		ret = ztd::make_unique<ResizeImplV_FP_AVX512<f16_traits>>(context, width);
	else if (type == PixelType::FLOAT)
		ret = ztd::make_unique<ResizeImplV_FP_AVX512<f32_traits>>(context, width);

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86_AVX512
