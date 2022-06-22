#pragma once

#ifndef ZIMG_RESIZE_X86_RESIZE_IMPL_AVX512_COMMON_H_
#define ZIMG_RESIZE_X86_RESIZE_IMPL_AVX512_COMMON_H_

#include <algorithm>
#include <climits>
#include <cstdint>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/make_array.h"
#include "common/except.h"
#include "graphengine/filter.h"
#include "resize/resize_impl.h"

#include "common/x86/sse2_util.h"
#include "common/x86/avx512_util.h"

namespace zimg {
namespace resize {
namespace {

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

template <int Taps>
inline FORCE_INLINE __m512i resize_line16_h_u16_avx512_xiter(unsigned j,
                                                             const unsigned * RESTRICT filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                                             const uint16_t * RESTRICT src, unsigned src_base, uint16_t limit)
{
	static_assert(Taps <= 8, "only up to 8 taps can be unrolled");
	static_assert(Taps >= -6, "only up to 6 taps in epilogue");
	static_assert(Taps % 2 == 0, "tap count must be even");
	constexpr int Tail = Taps > 0 ? Taps : -Taps;

	const __m512i i16_min = _mm512_set1_epi16(INT16_MIN);
	const __m512i lim = _mm512_set1_epi16(limit + INT16_MIN);

	const int16_t *filter_coeffs = filter_data + j * filter_stride;
	const uint16_t *src_p = src + (filter_left[j] - src_base) * 32;

	__m512i accum_lo = _mm512_setzero_si512();
	__m512i accum_hi = _mm512_setzero_si512();
	__m512i x0, x1, xl, xh, c, coeffs;

	unsigned k_end = Taps > 0 ? 0 : floor_n(filter_width + 1, 8);

	for (unsigned k = 0; k < k_end; k += 8) {
		coeffs = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)(filter_coeffs + k)));

		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_AAAA);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 0));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 32));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		accum_lo = mm512_dpwssd_epi32(accum_lo, c, xl);
		accum_hi = mm512_dpwssd_epi32(accum_hi, c, xh);

		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_BBBB);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 64));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 96));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		accum_lo = mm512_dpwssd_epi32(accum_lo, c, xl);
		accum_hi = mm512_dpwssd_epi32(accum_hi, c, xh);

		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_CCCC);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 128));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 160));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		accum_lo = mm512_dpwssd_epi32(accum_lo, c, xl);
		accum_hi = mm512_dpwssd_epi32(accum_hi, c, xh);

		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_DDDD);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 192));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 224));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		accum_lo = mm512_dpwssd_epi32(accum_lo, c, xl);
		accum_hi = mm512_dpwssd_epi32(accum_hi, c, xh);

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
		accum_lo = mm512_dpwssd_epi32(accum_lo, c, xl);
		accum_hi = mm512_dpwssd_epi32(accum_hi, c, xh);
	}

	if (Tail >= 4) {
		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_BBBB);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 64));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 96));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		accum_lo = mm512_dpwssd_epi32(accum_lo, c, xl);
		accum_hi = mm512_dpwssd_epi32(accum_hi, c, xh);
	}

	if (Tail >= 6) {
		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_CCCC);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 128));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 160));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		accum_lo = mm512_dpwssd_epi32(accum_lo, c, xl);
		accum_hi = mm512_dpwssd_epi32(accum_hi, c, xh);
	}

	if (Tail >= 8) {
		c = _mm512_shuffle_epi32(coeffs, _MM_PERM_DDDD);
		x0 = _mm512_load_si512((const __m256i *)(src_p + 192));
		x1 = _mm512_load_si512((const __m256i *)(src_p + 224));
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);
		accum_lo = mm512_dpwssd_epi32(accum_lo, c, xl);
		accum_hi = mm512_dpwssd_epi32(accum_hi, c, xh);
	}

	accum_lo = export2_i30_u16(accum_lo, accum_hi);
	accum_lo = _mm512_min_epi16(accum_lo, lim);
	accum_lo = _mm512_sub_epi16(accum_lo, i16_min);
	return accum_lo;
}

template <int Taps>
void resize_line16_h_u16_avx512(const unsigned * RESTRICT filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                const uint16_t * RESTRICT src, uint16_t * const * /* RESTRICT */ dst, unsigned src_base, unsigned left, unsigned right, uint16_t limit)
{
	unsigned vec_left = ceil_n(left, 32);
	unsigned vec_right = floor_n(right, 32);

#define XITER resize_line16_h_u16_avx512_xiter<Taps>
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

constexpr auto resize_line16_h_u16_avx512_jt_small = make_array(
	resize_line16_h_u16_avx512<2>,
	resize_line16_h_u16_avx512<2>,
	resize_line16_h_u16_avx512<4>,
	resize_line16_h_u16_avx512<4>,
	resize_line16_h_u16_avx512<6>,
	resize_line16_h_u16_avx512<6>,
	resize_line16_h_u16_avx512<8>,
	resize_line16_h_u16_avx512<8>);

constexpr auto resize_line16_h_u16_avx512_jt_large = make_array(
	resize_line16_h_u16_avx512<0>,
	resize_line16_h_u16_avx512<-2>,
	resize_line16_h_u16_avx512<-2>,
	resize_line16_h_u16_avx512<-4>,
	resize_line16_h_u16_avx512<-4>,
	resize_line16_h_u16_avx512<-6>,
	resize_line16_h_u16_avx512<-6>,
	resize_line16_h_u16_avx512<0>);


template <unsigned Taps>
void resize_line_h_perm_u16_avx512(const unsigned * RESTRICT permute_left, const uint16_t * RESTRICT permute_mask, const int16_t * RESTRICT filter_data, unsigned input_width,
                                   const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, unsigned left, unsigned right, uint16_t limit)
{
	static_assert(Taps <= 16, "permuted resampler only supports up to 16 taps");

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

		const __m512i mask = _mm512_load_si512(permute_mask + static_cast<size_t>(j) * 2);
		const int16_t *data = filter_data + static_cast<size_t>(j) * Taps;

		__m512i accum0 = _mm512_setzero_si512();
		__m512i accum1 = _mm512_setzero_si512();
		__m512i x, x0, x8, x16, coeffs;

		if (Taps >= 2) {
			x0 = _mm512_loadu_si512(src + left + 0);
			x0 = _mm512_add_epi16(x0, i16_min);

			x = x0;
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 0 * 16);
			accum0 = mm512_dpwssd_epi32(accum0, coeffs, x);
		}
		if (Taps >= 4) {
			x8 = _mm512_loadu_si512(src + left + 8);
			x8 = _mm512_add_epi16(x8, i16_min);

			x = _mm512_alignr_epi8(x8, x0, 4);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 2 * 16);
			accum1 = mm512_dpwssd_epi32(accum1, coeffs, x);
		}
		if (Taps >= 6) {
			x = _mm512_alignr_epi8(x8, x0, 8);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 4 * 16);
			accum0 = mm512_dpwssd_epi32(accum0, coeffs, x);
		}
		if (Taps >= 8) {
			x = _mm512_alignr_epi8(x8, x0, 12);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 6 * 16);
			accum1 = mm512_dpwssd_epi32(accum1, coeffs, x);
		}
		if (Taps >= 10) {
			x = x8;
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 8 * 16);
			accum0 = mm512_dpwssd_epi32(accum0, coeffs, x);
		}
		if (Taps >= 12) {
			x16 = _mm512_loadu_si512(src + left + 16);
			x16 = _mm512_add_epi16(x16, i16_min);

			x = _mm512_alignr_epi8(x16, x8, 4);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 10 * 16);
			accum1 = mm512_dpwssd_epi32(accum1, coeffs, x);
		}
		if (Taps >= 14) {
			x = _mm512_alignr_epi8(x16, x8, 8);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 12 * 16);
			accum0 = mm512_dpwssd_epi32(accum0, coeffs, x);
		}
		if (Taps >= 16) {
			x = _mm512_alignr_epi8(x16, x8, 12);
			x = _mm512_permutexvar_epi16(mask, x);
			coeffs = _mm512_load_si512(data + 14 * 16);
			accum1 = mm512_dpwssd_epi32(accum1, coeffs, x);
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
		const int16_t *data = filter_data + j * Taps;

		__m512i accum = _mm512_setzero_si512();
		__m512i x, coeffs;

		for (unsigned k = 0; k < Taps; k += 2) {
			unsigned num_load = std::min(input_width - left - k, 32U);
			__mmask32 load_mask = 0xFFFFFFFFU >> (32 - num_load);

			x = _mm512_maskz_loadu_epi16(load_mask, src + left + k);
			x = _mm512_permutexvar_epi16(mask, x);
			x = _mm512_add_epi16(x, i16_min);
			coeffs = _mm512_load_si512(data + k * 16);
			accum = mm512_dpwssd_epi32(accum, coeffs, x);
		}

		__m256i out = export_i30_u16(accum);
		out = _mm256_min_epi16(out, _mm512_castsi512_si256(lim));
		out = _mm256_sub_epi16(out, _mm512_castsi512_si256(i16_min));
		_mm256_store_si256((__m256i *)(dst + j), out);
	}
}

constexpr auto resize_line_h_perm_u16_avx512_jt = make_array(
	resize_line_h_perm_u16_avx512<2>,
	resize_line_h_perm_u16_avx512<4>,
	resize_line_h_perm_u16_avx512<6>,
	resize_line_h_perm_u16_avx512<8>,
	resize_line_h_perm_u16_avx512<10>,
	resize_line_h_perm_u16_avx512<12>,
	resize_line_h_perm_u16_avx512<14>,
	resize_line_h_perm_u16_avx512<16>);


constexpr unsigned V_ACCUM_NONE = 0;
constexpr unsigned V_ACCUM_INITIAL = 1;
constexpr unsigned V_ACCUM_UPDATE = 2;
constexpr unsigned V_ACCUM_FINAL = 3;

template <unsigned Taps, unsigned AccumMode>
inline FORCE_INLINE __m512i resize_line_v_u16_avx512_xiter(unsigned j, unsigned accum_base,
                                                           const uint16_t *src_p0, const uint16_t *src_p1, const uint16_t *src_p2, const uint16_t *src_p3,
                                                           const uint16_t *src_p4, const uint16_t *src_p5, const uint16_t *src_p6, const uint16_t *src_p7,
                                                           uint32_t * RESTRICT accum_p, const __m512i &c01, const __m512i &c23, const __m512i &c45, const __m512i &c67, uint16_t limit)
{
	static_assert(Taps >= 2 && Taps <= 8, "must have between 2-8 taps");
	static_assert(Taps % 2 == 0, "tap count must be even");

	const __m512i i16_min = _mm512_set1_epi16(INT16_MIN);
	const __m512i lim = _mm512_set1_epi16(limit + INT16_MIN);

	__m512i accum_lo = _mm512_setzero_si512();
	__m512i accum_hi = _mm512_setzero_si512();
	__m512i x0, x1, xl, xh;

	if (Taps >= 2) {
		x0 = _mm512_load_si512(src_p0 + j);
		x1 = _mm512_load_si512(src_p1 + j);
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);

		if (AccumMode == V_ACCUM_UPDATE || AccumMode == V_ACCUM_FINAL) {
			accum_lo = mm512_dpwssd_epi32(_mm512_load_si512(accum_p + j - accum_base + 0), c01, xl);
			accum_hi = mm512_dpwssd_epi32(_mm512_load_si512(accum_p + j - accum_base + 16), c01, xh);
		} else {
			accum_lo = _mm512_madd_epi16(c01, xl);
			accum_hi = _mm512_madd_epi16(c01, xh);
		}
	}
	if (Taps >= 4) {
		x0 = _mm512_load_si512(src_p2 + j);
		x1 = _mm512_load_si512(src_p3 + j);
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);

		accum_lo = mm512_dpwssd_epi32(accum_lo, c23, xl);
		accum_hi = mm512_dpwssd_epi32(accum_hi, c23, xh);
	}
	if (Taps >= 6) {
		x0 = _mm512_load_si512(src_p4 + j);
		x1 = _mm512_load_si512(src_p5 + j);
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);

		accum_lo = mm512_dpwssd_epi32(accum_lo, c45, xl);
		accum_hi = mm512_dpwssd_epi32(accum_hi, c45, xh);
	}
	if (Taps >= 8) {
		x0 = _mm512_load_si512(src_p6 + j);
		x1 = _mm512_load_si512(src_p7 + j);
		x0 = _mm512_add_epi16(x0, i16_min);
		x1 = _mm512_add_epi16(x1, i16_min);

		xl = _mm512_unpacklo_epi16(x0, x1);
		xh = _mm512_unpackhi_epi16(x0, x1);

		accum_lo = mm512_dpwssd_epi32(accum_lo, c67, xl);
		accum_hi = mm512_dpwssd_epi32(accum_hi, c67, xh);
	}

	if (AccumMode == V_ACCUM_INITIAL || AccumMode == V_ACCUM_UPDATE) {
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

template <unsigned Taps, unsigned AccumMode>
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

#define XITER resize_line_v_u16_avx512_xiter<Taps, AccumMode>
#define XARGS accum_base, src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, accum, c01, c23, c45, c67, limit
	if (left != vec_left) {
		out = XITER(vec_left - 32, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			_mm512_mask_storeu_epi16(dst + vec_left - 32, mmask32_set_hi(vec_left - left), out);
	}

	for (unsigned j = vec_left; j < vec_right; j += 32) {
		out = XITER(j, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			_mm512_store_si512(dst + j, out);
	}

	if (right != vec_right) {
		out = XITER(vec_right, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			_mm512_mask_storeu_epi16(dst + vec_right, mmask32_set_lo(right - vec_right), out);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line_v_u16_avx512_jt_small = make_array(
	resize_line_v_u16_avx512<2, V_ACCUM_NONE>,
	resize_line_v_u16_avx512<2, V_ACCUM_NONE>,
	resize_line_v_u16_avx512<4, V_ACCUM_NONE>,
	resize_line_v_u16_avx512<4, V_ACCUM_NONE>,
	resize_line_v_u16_avx512<6, V_ACCUM_NONE>,
	resize_line_v_u16_avx512<6, V_ACCUM_NONE>,
	resize_line_v_u16_avx512<8, V_ACCUM_NONE>,
	resize_line_v_u16_avx512<8, V_ACCUM_NONE>);

constexpr auto resize_line_v_u16_avx512_initial = resize_line_v_u16_avx512<8, V_ACCUM_INITIAL>;
constexpr auto resize_line_v_u16_avx512_update = resize_line_v_u16_avx512<8, V_ACCUM_UPDATE>;

constexpr auto resize_line_v_u16_avx512_jt_final = make_array(
	resize_line_v_u16_avx512<2, V_ACCUM_FINAL>,
	resize_line_v_u16_avx512<2, V_ACCUM_FINAL>,
	resize_line_v_u16_avx512<4, V_ACCUM_FINAL>,
	resize_line_v_u16_avx512<4, V_ACCUM_FINAL>,
	resize_line_v_u16_avx512<6, V_ACCUM_FINAL>,
	resize_line_v_u16_avx512<6, V_ACCUM_FINAL>,
	resize_line_v_u16_avx512<8, V_ACCUM_FINAL>,
	resize_line_v_u16_avx512<8, V_ACCUM_FINAL>);


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
	decltype(resize_line16_h_u16_avx512_jt_small)::value_type m_func;
	uint16_t m_pixel_max;
public:
	ResizeImplH_U16_AVX512(const FilterContext &filter, unsigned height, unsigned depth) try :
		ResizeImplH(filter, height, PixelType::WORD),
		m_func{},
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{
		m_desc.step = 32;
		m_desc.scratchpad_size = (ceil_n(checked_size_t{ filter.input_width }, 32) * sizeof(uint16_t) * 32).get();

		if (filter.filter_width > 8)
			m_func = resize_line16_h_u16_avx512_jt_large[filter.filter_width % 8];
		else
			m_func = resize_line16_h_u16_avx512_jt_small[filter.filter_width - 1];
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		auto range = get_col_deps(left, right);

		alignas(64) const uint16_t *src_ptr[32];
		alignas(64) uint16_t *dst_ptr[32];
		uint16_t *transpose_buf = static_cast<uint16_t *>(tmp);
		unsigned height = m_desc.format.height;

		calculate_line_address(src_ptr + 0, in->ptr, in->stride, in->mask, i + 0, height);
		calculate_line_address(src_ptr + 8, in->ptr, in->stride, in->mask, i + std::min(8U, height - i - 1), height);
		calculate_line_address(src_ptr + 16, in->ptr, in->stride, in->mask, i + std::min(16U, height - i - 1), height);
		calculate_line_address(src_ptr + 24, in->ptr, in->stride, in->mask, i + std::min(24U, height - i - 1), height);

		transpose_line_32x32_epi16(transpose_buf, src_ptr, floor_n(range.first, 32), ceil_n(range.second, 32));

		calculate_line_address(dst_ptr + 0, out->ptr, out->stride, out->mask, i + 0, height);
		calculate_line_address(dst_ptr + 8, out->ptr, out->stride, out->mask, i + std::min(8U, height - i - 1), height);
		calculate_line_address(dst_ptr + 16, out->ptr, out->stride, out->mask, i + std::min(16U, height - i - 1), height);
		calculate_line_address(dst_ptr + 24, out->ptr, out->stride, out->mask, i + std::min(24U, height - i - 1), height);

		m_func(m_filter.left.data(), m_filter.data_i16.data(), m_filter.stride_i16, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 32), left, right, m_pixel_max);
	}
};


class ResizeImplH_Permute_U16_AVX512 final : public graphengine::Filter {
	typedef decltype(resize_line_h_perm_u16_avx512_jt)::value_type func_type;

	struct PermuteContext {
		AlignedVector<unsigned> left;
		AlignedVector<uint16_t> permute;
		AlignedVector<int16_t> data;
		unsigned filter_rows;
		unsigned filter_width;
		unsigned input_width;
	};

	graphengine::FilterDescriptor m_desc;
	PermuteContext m_context;
	uint16_t m_pixel_max;
	func_type m_func;

	ResizeImplH_Permute_U16_AVX512(PermuteContext context, unsigned height, unsigned depth) :
		m_desc{},
		m_context(std::move(context)),
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) },
		m_func{ resize_line_h_perm_u16_avx512_jt[(m_context.filter_width - 1) / 2] }
	{
		m_desc.format = { context.filter_rows, height, pixel_size(PixelType::WORD) };
		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.step = 1;
		m_desc.alignment_mask = 15;
		m_desc.flags.entire_row = !std::is_sorted(m_context.left.begin(), m_context.left.end());
	}
public:
	static std::unique_ptr<graphengine::Filter> create(const FilterContext &filter, unsigned height, unsigned depth)
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

		std::unique_ptr<graphengine::Filter> ret{ new ResizeImplH_Permute_U16_AVX512(std::move(context), height, depth) };
		return ret;
	}

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	pair_unsigned get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override
	{
		if (m_desc.flags.entire_row)
			return{ 0, m_context.input_width };

		unsigned input_width = m_context.input_width;
		unsigned right_base = m_context.left[(right + 15) / 16 - 1];
		unsigned iter_width = m_context.filter_width + 32;
		return{ m_context.left[left / 16],  right_base + std::min(input_width - right_base, iter_width) };
	}

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		m_func(m_context.left.data(), m_context.permute.data(), m_context.data.data(), m_context.input_width, in->get_line<uint16_t>(i), out->get_line<uint16_t>(i), left, right, m_pixel_max);
	}
};


class ResizeImplV_U16_AVX512 final : public ResizeImplV {
	uint16_t m_pixel_max;
public:
	ResizeImplV_U16_AVX512(const FilterContext &filter, unsigned width, unsigned depth) try :
		ResizeImplV(filter, width, PixelType::WORD),
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{
		if (m_filter.filter_width > 8)
			m_desc.scratchpad_size = (ceil_n(checked_size_t{ width }, 32) * sizeof(uint32_t)).get();
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		const int16_t *filter_data = m_filter.data_i16.data() + i * m_filter.stride_i16;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		alignas(64) const uint16_t *src_lines[8];
		uint16_t *dst_line = out->get_line<uint16_t>(i);
		uint32_t *accum_buf = static_cast<uint32_t *>(tmp);

		unsigned top = m_filter.left[i];

		if (filter_width <= 8) {
			calculate_line_address(src_lines, in->ptr, in->stride, in->mask, top + 0, src_height);
			resize_line_v_u16_avx512_jt_small[filter_width - 1](filter_data, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
		} else {
			unsigned k_end = ceil_n(filter_width, 8) - 8;

			calculate_line_address(src_lines, in->ptr, in->stride, in->mask, top + 0, src_height);
			resize_line_v_u16_avx512_initial(filter_data + 0, src_lines, dst_line, accum_buf, left, right, m_pixel_max);

			for (unsigned k = 8; k < k_end; k += 8) {
				calculate_line_address(src_lines, in->ptr, in->stride, in->mask, top + k, src_height);
				resize_line_v_u16_avx512_update(filter_data + k, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
			}

			calculate_line_address(src_lines, in->ptr, in->stride, in->mask, top + k_end, src_height);
			resize_line_v_u16_avx512_jt_final[filter_width - k_end - 1](filter_data + k_end, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
		}
	}
};

} // namespace
} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_X86_RESIZE_IMPL_AVX512_COMMON_H_
