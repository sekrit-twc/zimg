#ifdef ZIMG_X86

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <emmintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"

#define HAVE_CPU_SSE2
  #include "common/x86util.h"
#undef HAVE_CPU_SSE2

#include "common/make_unique.h"
#include "common/pixel.h"
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {
namespace resize {

namespace {

inline FORCE_INLINE void mm_store_left_epi16(uint16_t *dst, __m128i x, unsigned count)
{
	mm_store_left_si128((__m128i *)dst, x, count * 2);
}

inline FORCE_INLINE void mm_store_right_epi16(uint16_t *dst, __m128i x, unsigned count)
{
	mm_store_right_si128((__m128i *)dst, x, count * 2);
}

void transpose_line_8x8_epi16(uint16_t *dst, const uint16_t *src_p0, const uint16_t *src_p1, const uint16_t *src_p2, const uint16_t *src_p3,
                              const uint16_t *src_p4, const uint16_t *src_p5, const uint16_t *src_p6, const uint16_t *src_p7,
                              unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; j += 8) {
		__m128i x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = _mm_load_si128((const __m128i *)(src_p0 + j));
		x1 = _mm_load_si128((const __m128i *)(src_p1 + j));
		x2 = _mm_load_si128((const __m128i *)(src_p2 + j));
		x3 = _mm_load_si128((const __m128i *)(src_p3 + j));
		x4 = _mm_load_si128((const __m128i *)(src_p4 + j));
		x5 = _mm_load_si128((const __m128i *)(src_p5 + j));
		x6 = _mm_load_si128((const __m128i *)(src_p6 + j));
		x7 = _mm_load_si128((const __m128i *)(src_p7 + j));

		mm_transpose8_epi16(x0, x1, x2, x3, x4, x5, x6, x7);

		_mm_store_si128((__m128i *)(dst + 0), x0);
		_mm_store_si128((__m128i *)(dst + 8), x1);
		_mm_store_si128((__m128i *)(dst + 16), x2);
		_mm_store_si128((__m128i *)(dst + 24), x3);
		_mm_store_si128((__m128i *)(dst + 32), x4);
		_mm_store_si128((__m128i *)(dst + 40), x5);
		_mm_store_si128((__m128i *)(dst + 48), x6);
		_mm_store_si128((__m128i *)(dst + 56), x7);

		dst += 64;
	}
}

inline FORCE_INLINE __m128i export_i30_u16(__m128i lo, __m128i hi, uint16_t limit)
{
	const __m128i round = _mm_set1_epi32(1 << 13);

	lo = _mm_add_epi32(lo, round);
	hi = _mm_add_epi32(hi, round);

	lo = _mm_srai_epi32(lo, 14);
	hi = _mm_srai_epi32(hi, 14);

	lo = _mm_packs_epi32(lo, hi);

	return lo;
}


template <bool DoLoop, unsigned Tail>
inline FORCE_INLINE __m128i resize_line8_h_u16_sse2_xiter(unsigned j,
                                                          const unsigned *filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                                          const uint16_t * RESTRICT src_ptr, unsigned src_base, uint16_t limit)
{
	const __m128i i16_min = _mm_set1_epi16(INT16_MIN);
	const __m128i lim = _mm_set1_epi16(limit + INT16_MIN);

	const int16_t *filter_coeffs = filter_data + j * filter_stride;
	const uint16_t *src_p = src_ptr + (filter_left[j] - src_base) * 8;

	__m128i accum_lo = _mm_setzero_si128();
	__m128i accum_hi = _mm_setzero_si128();
	__m128i x0, x1, xl, xh, c, coeffs;

	unsigned k_end = DoLoop ? floor_n(filter_width + 1, 8) : 0;

	for (unsigned k = 0; k < k_end; k += 8) {
		coeffs = _mm_load_si128((const __m128i *)(filter_coeffs + k));

		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x0 = _mm_load_si128((const __m128i *)(src_p + (k + 0) * 8));
		x1 = _mm_load_si128((const __m128i *)(src_p + (k + 1) * 8));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);

		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x0 = _mm_load_si128((const __m128i *)(src_p + (k + 2) * 8));
		x1 = _mm_load_si128((const __m128i *)(src_p + (k + 3) * 8));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);

		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x0 = _mm_load_si128((const __m128i *)(src_p + (k + 4) * 8));
		x1 = _mm_load_si128((const __m128i *)(src_p + (k + 5) * 8));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);

		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x0 = _mm_load_si128((const __m128i *)(src_p + (k + 6) * 8));
		x1 = _mm_load_si128((const __m128i *)(src_p + (k + 7) * 8));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}

	if (Tail >= 2) {
		coeffs = _mm_load_si128((const __m128i *)(filter_coeffs + k_end));

		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x0 = _mm_load_si128((const __m128i *)(src_p + (k_end + 0) * 8));
		x1 = _mm_load_si128((const __m128i *)(src_p + (k_end + 1) * 8));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}

	if (Tail >= 4) {
		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x0 = _mm_load_si128((const __m128i *)(src_p + (k_end + 2) * 8));
		x1 = _mm_load_si128((const __m128i *)(src_p + (k_end + 3) * 8));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}

	if (Tail >= 6) {
		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x0 = _mm_load_si128((const __m128i *)(src_p + (k_end + 4) * 8));
		x1 = _mm_load_si128((const __m128i *)(src_p + (k_end + 5) * 8));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}

	if (Tail >= 8) {
		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x0 = _mm_load_si128((const __m128i *)(src_p + (k_end + 6) * 8));
		x1 = _mm_load_si128((const __m128i *)(src_p + (k_end + 7) * 8));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}

	accum_lo = export_i30_u16(accum_lo, accum_hi, limit);
	accum_lo = _mm_min_epi16(accum_lo, lim);
	accum_lo = _mm_sub_epi16(accum_lo, i16_min);
	return accum_lo;
}

template <bool DoLoop, unsigned Tail>
void resize_line8_h_u16_sse2(const unsigned *filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                             const uint16_t * RESTRICT src_ptr, uint16_t * const *dst_ptr, unsigned src_base, unsigned left, unsigned right, uint16_t limit)
{
	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	uint16_t * RESTRICT dst_p0 = dst_ptr[0];
	uint16_t * RESTRICT dst_p1 = dst_ptr[1];
	uint16_t * RESTRICT dst_p2 = dst_ptr[2];
	uint16_t * RESTRICT dst_p3 = dst_ptr[3];
	uint16_t * RESTRICT dst_p4 = dst_ptr[4];
	uint16_t * RESTRICT dst_p5 = dst_ptr[5];
	uint16_t * RESTRICT dst_p6 = dst_ptr[6];
	uint16_t * RESTRICT dst_p7 = dst_ptr[7];

#define XITER resize_line8_h_u16_sse2_xiter<DoLoop, Tail>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src_ptr, src_base, limit
	for (unsigned j = left; j < vec_left; ++j) {
		__m128i x = XITER(j, XARGS);
		mm_scatter_epi16(dst_p0 + j, dst_p1 + j, dst_p2 + j, dst_p3 + j, dst_p4 + j, dst_p5 + j, dst_p6 + j, dst_p7 + j, x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m128i x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = XITER(j + 0, XARGS);
		x1 = XITER(j + 1, XARGS);
		x2 = XITER(j + 2, XARGS);
		x3 = XITER(j + 3, XARGS);
		x4 = XITER(j + 4, XARGS);
		x5 = XITER(j + 5, XARGS);
		x6 = XITER(j + 6, XARGS);
		x7 = XITER(j + 7, XARGS);

		mm_transpose8_epi16(x0, x1, x2, x3, x4, x5, x6, x7);

		_mm_store_si128((__m128i *)(dst_p0 + j), x0);
		_mm_store_si128((__m128i *)(dst_p1 + j), x1);
		_mm_store_si128((__m128i *)(dst_p2 + j), x2);
		_mm_store_si128((__m128i *)(dst_p3 + j), x3);
		_mm_store_si128((__m128i *)(dst_p4 + j), x4);
		_mm_store_si128((__m128i *)(dst_p5 + j), x5);
		_mm_store_si128((__m128i *)(dst_p6 + j), x6);
		_mm_store_si128((__m128i *)(dst_p7 + j), x7);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		__m128i x = XITER(j, XARGS);
		mm_scatter_epi16(dst_p0 + j, dst_p1 + j, dst_p2 + j, dst_p3 + j, dst_p4 + j, dst_p5 + j, dst_p6 + j, dst_p7 + j, x);
	}
#undef XITER
#undef XARGS
}

const decltype(&resize_line8_h_u16_sse2<false, 0>) resize_line8_h_u16_sse2_jt_small[] = {
	resize_line8_h_u16_sse2<false, 2>,
	resize_line8_h_u16_sse2<false, 2>,
	resize_line8_h_u16_sse2<false, 4>,
	resize_line8_h_u16_sse2<false, 4>,
	resize_line8_h_u16_sse2<false, 6>,
	resize_line8_h_u16_sse2<false, 6>,
	resize_line8_h_u16_sse2<false, 8>,
	resize_line8_h_u16_sse2<false, 8>
};

const decltype(&resize_line8_h_u16_sse2<false, 0>) resize_line8_h_u16_sse2_jt_large[] = {
	resize_line8_h_u16_sse2<true, 0>,
	resize_line8_h_u16_sse2<true, 2>,
	resize_line8_h_u16_sse2<true, 2>,
	resize_line8_h_u16_sse2<true, 4>,
	resize_line8_h_u16_sse2<true, 4>,
	resize_line8_h_u16_sse2<true, 6>,
	resize_line8_h_u16_sse2<true, 6>,
	resize_line8_h_u16_sse2<true, 0>,
};


template <unsigned N, bool ReadAccum, bool WriteToAccum>
inline FORCE_INLINE __m128i resize_line_v_u16_sse2_xiter(unsigned j, unsigned accum_base,
                                                         const uint16_t * RESTRICT src_p0, const uint16_t * RESTRICT src_p1, const uint16_t * RESTRICT src_p2, const uint16_t * RESTRICT src_p3,
                                                         const uint16_t * RESTRICT src_p4, const uint16_t * RESTRICT src_p5, const uint16_t * RESTRICT src_p6, const uint16_t * RESTRICT src_p7,
                                                         const uint32_t *accum_p, const __m128i &c01, const __m128i &c23, const __m128i &c45, const __m128i &c67, uint16_t limit)
{
	const __m128i i16_min = _mm_set1_epi16(INT16_MIN);
	const __m128i lim = _mm_set1_epi16(limit + INT16_MIN);

	__m128i accum_lo = _mm_setzero_si128();
	__m128i accum_hi = _mm_setzero_si128();
	__m128i x0, x1, xl, xh;

	if (N >= 0) {
		x0 = _mm_load_si128((const __m128i *)(src_p0 + j));
		x1 = _mm_load_si128((const __m128i *)(src_p1 + j));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c01, xl);
		xh = _mm_madd_epi16(c01, xh);

		if (ReadAccum) {
			accum_lo = _mm_add_epi32(_mm_load_si128((const __m128i *)(accum_p + j - accum_base + 0)), xl);
			accum_hi = _mm_add_epi32(_mm_load_si128((const __m128i *)(accum_p + j - accum_base + 4)), xh);
		} else {
			accum_lo = xl;
			accum_hi = xh;
		}
	}
	if (N >= 2) {
		x0 = _mm_load_si128((const __m128i *)(src_p2 + j));
		x1 = _mm_load_si128((const __m128i *)(src_p3 + j));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c23, xl);
		xh = _mm_madd_epi16(c23, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}
	if (N >= 4) {
		x0 = _mm_load_si128((const __m128i *)(src_p4 + j));
		x1 = _mm_load_si128((const __m128i *)(src_p5 + j));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c45, xl);
		xh = _mm_madd_epi16(c45, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}
	if (N >= 6) {
		x0 = _mm_load_si128((const __m128i *)(src_p6 + j));
		x1 = _mm_load_si128((const __m128i *)(src_p7 + j));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c67, xl);
		xh = _mm_madd_epi16(c67, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}

	if (WriteToAccum) {
		_mm_store_si128((__m128i *)(accum_p + j - accum_base + 0), accum_lo);
		_mm_store_si128((__m128i *)(accum_p + j - accum_base + 4), accum_hi);
		return _mm_setzero_si128();
	} else {
		accum_lo = export_i30_u16(accum_lo, accum_hi, limit);
		accum_lo = _mm_min_epi16(accum_lo, lim);
		accum_lo = _mm_sub_epi16(accum_lo, i16_min);

		return accum_lo;
	}
}

template <unsigned N, bool ReadAccum, bool WriteToAccum>
void resize_line_v_u16_sse2(const int16_t *filter_data, const uint16_t * const *src_lines, uint16_t *dst, uint32_t *accum, unsigned left, unsigned right, uint16_t limit)
{
	const uint16_t * RESTRICT src_p0 = src_lines[0];
	const uint16_t * RESTRICT src_p1 = src_lines[1];
	const uint16_t * RESTRICT src_p2 = src_lines[2];
	const uint16_t * RESTRICT src_p3 = src_lines[3];
	const uint16_t * RESTRICT src_p4 = src_lines[4];
	const uint16_t * RESTRICT src_p5 = src_lines[5];
	const uint16_t * RESTRICT src_p6 = src_lines[6];
	const uint16_t * RESTRICT src_p7 = src_lines[7];
	uint16_t * RESTRICT dst_p = dst;
	uint32_t * RESTRICT accum_p = accum;

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);
	unsigned accum_base = floor_n(left, 8);

	const __m128i c01 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[0]), _mm_set1_epi16(filter_data[1]));
	const __m128i c23 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[2]), _mm_set1_epi16(filter_data[3]));
	const __m128i c45 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[4]), _mm_set1_epi16(filter_data[5]));
	const __m128i c67 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[6]), _mm_set1_epi16(filter_data[7]));

	__m128i out;

#define XITER resize_line_v_u16_sse2_xiter<N, ReadAccum, WriteToAccum>
#define XARGS accum_base, src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, accum_p, c01, c23, c45, c67, limit
	if (left != vec_left) {
		out = XITER(vec_left - 8, XARGS);

		if (!WriteToAccum)
			mm_store_left_epi16(dst_p + vec_left - 8, out, vec_left - left);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		out = XITER(j, XARGS);

		if (!WriteToAccum)
			_mm_store_si128((__m128i *)(dst_p + j), out);
	}

	if (right != vec_right) {
		out = XITER(vec_right, XARGS);

		if (!WriteToAccum)
			mm_store_right_epi16(dst_p + vec_right, out, right - vec_right);
	}
#undef XITER
#undef XARGS
}

const decltype(&resize_line_v_u16_sse2<0, false, false>) resize_line_v_u16_sse2_jt_a[] = {
	resize_line_v_u16_sse2<0, false, false>,
	resize_line_v_u16_sse2<0, false, false>,
	resize_line_v_u16_sse2<2, false, false>,
	resize_line_v_u16_sse2<2, false, false>,
	resize_line_v_u16_sse2<4, false, false>,
	resize_line_v_u16_sse2<4, false, false>,
	resize_line_v_u16_sse2<6, false, false>,
	resize_line_v_u16_sse2<6, false, false>,
};

const decltype(&resize_line_v_u16_sse2<0, false, false>) resize_line_v_u16_sse2_jt_b[] = {
	resize_line_v_u16_sse2<0, true, false>,
	resize_line_v_u16_sse2<0, true, false>,
	resize_line_v_u16_sse2<2, true, false>,
	resize_line_v_u16_sse2<2, true, false>,
	resize_line_v_u16_sse2<4, true, false>,
	resize_line_v_u16_sse2<4, true, false>,
	resize_line_v_u16_sse2<6, true, false>,
	resize_line_v_u16_sse2<6, true, false>,
};


class ResizeImplH_U16_SSE2 final : public ResizeImplH {
	decltype(&resize_line8_h_u16_sse2<false, 0>) m_func;
	uint16_t m_pixel_max;
public:
	ResizeImplH_U16_SSE2(const FilterContext &filter, unsigned height, unsigned depth) :
		ResizeImplH(filter, image_attributes{ filter.filter_rows, height, PixelType::WORD }),
		m_func{},
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{
		if (filter.filter_width > 8)
			m_func = resize_line8_h_u16_sse2_jt_large[filter.filter_width % 8];
		else
			m_func = resize_line8_h_u16_sse2_jt_small[filter.filter_width - 1];
	}

	unsigned get_simultaneous_lines() const override { return 8; }

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		auto range = get_required_col_range(left, right);

		try {
			checked_size_t size = (static_cast<checked_size_t>(range.second) - floor_n(range.first, 8) + 8) * sizeof(uint16_t) * 8;
			return size.get();
		} catch (const std::overflow_error &) {
			throw error::OutOfMemory{};
		}
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		auto src_buf = graph::static_buffer_cast<const uint16_t>(*src);
		auto dst_buf = graph::static_buffer_cast<uint16_t>(*dst);
		auto range = get_required_col_range(left, right);

		const uint16_t *src_ptr[8] = { 0 };
		uint16_t *dst_ptr[8] = { 0 };
		uint16_t *transpose_buf = static_cast<uint16_t *>(tmp);
		unsigned height = get_image_attributes().height;

		for (unsigned n = 0; n < 8; ++n) {
			src_ptr[n] = src_buf[std::min(i + n, height - 1)];
		}

		transpose_line_8x8_epi16(transpose_buf, src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[4], src_ptr[5], src_ptr[6], src_ptr[7],
		                         floor_n(range.first, 8), ceil_n(range.second, 8));

		for (unsigned n = 0; n < 8; ++n) {
			dst_ptr[n] = dst_buf[std::min(i + n, height - 1)];
		}

		m_func(m_filter.left.data(), m_filter.data_i16.data(), m_filter.stride_i16, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 8), left, right, m_pixel_max);
	}
};


class ResizeImplV_U16_SSE2 final : public ResizeImplV {
	uint16_t m_pixel_max;
public:
	ResizeImplV_U16_SSE2(const FilterContext &filter, unsigned width, unsigned depth) :
		ResizeImplV(filter, image_attributes{ width, filter.filter_rows, PixelType::WORD }),
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		checked_size_t size = 0;

		try {
			if (m_filter.filter_width > 4)
				size += (ceil_n(checked_size_t{ right }, 8) - floor_n(left, 8)) * sizeof(uint32_t);
		} catch (const std::overflow_error &) {
			throw error::OutOfMemory{};
		}

		return size.get();
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		auto src_buf = graph::static_buffer_cast<const uint16_t>(*src);
		auto dst_buf = graph::static_buffer_cast<uint16_t>(*dst);

		const int16_t *filter_data = m_filter.data_i16.data() + i * m_filter.stride_i16;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const uint16_t *src_lines[8] = { 0 };
		uint16_t *dst_line = dst_buf[i];
		uint32_t *accum_buf = static_cast<uint32_t *>(tmp);

		unsigned k_end = ceil_n(filter_width, 8) - 8;
		unsigned top = m_filter.left[i];

		for (unsigned k = 0; k < k_end; k += 8) {
			for (unsigned n = 0; n < 8; ++n) {
				src_lines[n] = src_buf[std::min(top + k + n, src_height - 1)];
			}

			if (k == 0)
				resize_line_v_u16_sse2<6, false, true>(filter_data + k, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
			else
				resize_line_v_u16_sse2<6, true, true>(filter_data + k, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
		}

		for (unsigned n = 0; n < 8; ++n) {
			src_lines[n] = src_buf[std::min(top + k_end + n, src_height - 1)];
		}

		if (k_end == 0)
			resize_line_v_u16_sse2_jt_a[filter_width - k_end - 1](filter_data + k_end, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
		else
			resize_line_v_u16_sse2_jt_b[filter_width - k_end - 1](filter_data + k_end, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
	}
};

} // namespace


std::unique_ptr<graph::ImageFilter> create_resize_impl_h_sse2(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == PixelType::WORD)
		ret = ztd::make_unique<ResizeImplH_U16_SSE2>(context, height, depth);

	return ret;
}

std::unique_ptr<graph::ImageFilter> create_resize_impl_v_sse2(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == PixelType::WORD)
		ret = ztd::make_unique<ResizeImplV_U16_SSE2>(context, width, depth);

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
