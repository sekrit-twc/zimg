#ifdef ZIMG_X86

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <emmintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/make_array.h"
#include "common/pixel.h"
#include "graphengine/filter.h"
#include "resize/resize_impl.h"
#include "resize_impl_x86.h"

#include "common/x86/sse2_util.h"

namespace zimg {
namespace resize {

namespace {

void transpose_line_8x8_epi16(uint16_t * RESTRICT dst, const uint16_t * const * RESTRICT src, unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; j += 8) {
		__m128i x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = _mm_load_si128((const __m128i *)(src[0] + j));
		x1 = _mm_load_si128((const __m128i *)(src[1] + j));
		x2 = _mm_load_si128((const __m128i *)(src[2] + j));
		x3 = _mm_load_si128((const __m128i *)(src[3] + j));
		x4 = _mm_load_si128((const __m128i *)(src[4] + j));
		x5 = _mm_load_si128((const __m128i *)(src[5] + j));
		x6 = _mm_load_si128((const __m128i *)(src[6] + j));
		x7 = _mm_load_si128((const __m128i *)(src[7] + j));

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

inline FORCE_INLINE __m128i export_i30_u16(__m128i lo, __m128i hi)
{
	const __m128i round = _mm_set1_epi32(1 << 13);

	lo = _mm_add_epi32(lo, round);
	hi = _mm_add_epi32(hi, round);

	lo = _mm_srai_epi32(lo, 14);
	hi = _mm_srai_epi32(hi, 14);

	lo = _mm_packs_epi32(lo, hi);

	return lo;
}


template <int Taps>
inline FORCE_INLINE __m128i resize_line8_h_u16_sse2_xiter(unsigned j,
                                                          const unsigned * RESTRICT filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                                          const uint16_t * RESTRICT src, unsigned src_base, uint16_t limit)
{
	static_assert(Taps <= 8, "only up to 8 taps can be unrolled");
	static_assert(Taps >= -6, "only up to 6 taps in epilogue");
	static_assert(Taps % 2 == 0, "tap count must be even");
	constexpr int Tail = Taps > 0 ? Taps : -Taps;

	const __m128i i16_min = _mm_set1_epi16(INT16_MIN);
	const __m128i lim = _mm_set1_epi16(limit + INT16_MIN);

	const int16_t *filter_coeffs = filter_data + j * filter_stride;
	const uint16_t *src_p = src + (filter_left[j] - src_base) * 8;

	__m128i accum_lo = _mm_setzero_si128();
	__m128i accum_hi = _mm_setzero_si128();
	__m128i x0, x1, xl, xh, c, coeffs;

	unsigned k_end = Taps > 0 ? 0 : floor_n(filter_width + 1, 8);

	for (unsigned k = 0; k < k_end; k += 8) {
		coeffs = _mm_load_si128((const __m128i *)(filter_coeffs + k));

		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x0 = _mm_load_si128((const __m128i *)(src_p + 0));
		x1 = _mm_load_si128((const __m128i *)(src_p + 8));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);

		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x0 = _mm_load_si128((const __m128i *)(src_p + 16));
		x1 = _mm_load_si128((const __m128i *)(src_p + 24));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);

		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x0 = _mm_load_si128((const __m128i *)(src_p + 32));
		x1 = _mm_load_si128((const __m128i *)(src_p + 40));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);

		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x0 = _mm_load_si128((const __m128i *)(src_p + 48));
		x1 = _mm_load_si128((const __m128i *)(src_p + 56));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);

		src_p += 64;
	}

	if (Tail >= 2) {
		coeffs = _mm_load_si128((const __m128i *)(filter_coeffs + k_end));

		c = _mm_shuffle_epi32(coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x0 = _mm_load_si128((const __m128i *)(src_p + 0));
		x1 = _mm_load_si128((const __m128i *)(src_p + 8));
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
		x0 = _mm_load_si128((const __m128i *)(src_p + 16));
		x1 = _mm_load_si128((const __m128i *)(src_p + 24));
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
		x0 = _mm_load_si128((const __m128i *)(src_p + 32));
		x1 = _mm_load_si128((const __m128i *)(src_p + 40));
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
		x0 = _mm_load_si128((const __m128i *)(src_p + 48));
		x1 = _mm_load_si128((const __m128i *)(src_p + 56));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c, xl);
		xh = _mm_madd_epi16(c, xh);

		accum_lo = _mm_add_epi32(accum_lo, xl);
		accum_hi = _mm_add_epi32(accum_hi, xh);
	}

	accum_lo = export_i30_u16(accum_lo, accum_hi);
	accum_lo = _mm_min_epi16(accum_lo, lim);
	accum_lo = _mm_sub_epi16(accum_lo, i16_min);
	return accum_lo;
}

template <int Taps>
void resize_line8_h_u16_sse2(const unsigned * RESTRICT filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                             const uint16_t * RESTRICT src, uint16_t * const * RESTRICT dst, unsigned src_base, unsigned left, unsigned right, uint16_t limit)
{
	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	uint16_t *dst_p0 = dst[0];
	uint16_t *dst_p1 = dst[1];
	uint16_t *dst_p2 = dst[2];
	uint16_t *dst_p3 = dst[3];
	uint16_t *dst_p4 = dst[4];
	uint16_t *dst_p5 = dst[5];
	uint16_t *dst_p6 = dst[6];
	uint16_t *dst_p7 = dst[7];

#define XITER resize_line8_h_u16_sse2_xiter<Taps>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src, src_base, limit
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

constexpr auto resize_line8_h_u16_sse2_jt_small = make_array(
	resize_line8_h_u16_sse2<2>,
	resize_line8_h_u16_sse2<2>,
	resize_line8_h_u16_sse2<4>,
	resize_line8_h_u16_sse2<4>,
	resize_line8_h_u16_sse2<6>,
	resize_line8_h_u16_sse2<6>,
	resize_line8_h_u16_sse2<8>,
	resize_line8_h_u16_sse2<8>);

constexpr auto resize_line8_h_u16_sse2_jt_large = make_array(
	resize_line8_h_u16_sse2<0>,
	resize_line8_h_u16_sse2<-2>,
	resize_line8_h_u16_sse2<-2>,
	resize_line8_h_u16_sse2<-4>,
	resize_line8_h_u16_sse2<-4>,
	resize_line8_h_u16_sse2<-6>,
	resize_line8_h_u16_sse2<-6>,
	resize_line8_h_u16_sse2<0>);


constexpr unsigned V_ACCUM_NONE = 0;
constexpr unsigned V_ACCUM_INITIAL = 1;
constexpr unsigned V_ACCUM_UPDATE = 2;
constexpr unsigned V_ACCUM_FINAL = 3;

template <unsigned Taps, unsigned AccumMode>
inline FORCE_INLINE __m128i resize_line_v_u16_sse2_xiter(unsigned j, unsigned accum_base,
                                                         const uint16_t *src_p0, const uint16_t *src_p1, const uint16_t *src_p2, const uint16_t *src_p3,
                                                         const uint16_t *src_p4, const uint16_t *src_p5, const uint16_t *src_p6, const uint16_t *src_p7,
                                                         uint32_t * RESTRICT accum_p, const __m128i &c01, const __m128i &c23, const __m128i &c45, const __m128i &c67, uint16_t limit)
{
	static_assert(Taps >= 2 && Taps <= 8, "must have between 2-8 taps");
	static_assert(Taps % 2 == 0, "tap count must be even");

	const __m128i i16_min = _mm_set1_epi16(INT16_MIN);
	const __m128i lim = _mm_set1_epi16(limit + INT16_MIN);

	__m128i accum_lo = _mm_setzero_si128();
	__m128i accum_hi = _mm_setzero_si128();
	__m128i x0, x1, xl, xh;

	if (Taps >= 2) {
		x0 = _mm_load_si128((const __m128i *)(src_p0 + j));
		x1 = _mm_load_si128((const __m128i *)(src_p1 + j));
		x0 = _mm_add_epi16(x0, i16_min);
		x1 = _mm_add_epi16(x1, i16_min);

		xl = _mm_unpacklo_epi16(x0, x1);
		xh = _mm_unpackhi_epi16(x0, x1);
		xl = _mm_madd_epi16(c01, xl);
		xh = _mm_madd_epi16(c01, xh);

		if (AccumMode == V_ACCUM_UPDATE || AccumMode == V_ACCUM_FINAL) {
			accum_lo = _mm_add_epi32(_mm_load_si128((const __m128i *)(accum_p + j - accum_base + 0)), xl);
			accum_hi = _mm_add_epi32(_mm_load_si128((const __m128i *)(accum_p + j - accum_base + 4)), xh);
		} else {
			accum_lo = xl;
			accum_hi = xh;
		}
	}
	if (Taps >= 4) {
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
	if (Taps >= 6) {
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
	if (Taps >= 8) {
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

	if (AccumMode == V_ACCUM_INITIAL || AccumMode == V_ACCUM_UPDATE) {
		_mm_store_si128((__m128i *)(accum_p + j - accum_base + 0), accum_lo);
		_mm_store_si128((__m128i *)(accum_p + j - accum_base + 4), accum_hi);
		return _mm_setzero_si128();
	} else {
		accum_lo = export_i30_u16(accum_lo, accum_hi);
		accum_lo = _mm_min_epi16(accum_lo, lim);
		accum_lo = _mm_sub_epi16(accum_lo, i16_min);

		return accum_lo;
	}
}

template <unsigned Taps, unsigned AccumMode>
void resize_line_v_u16_sse2(const int16_t * RESTRICT filter_data, const uint16_t * const * RESTRICT src, uint16_t * RESTRICT dst, uint32_t * RESTRICT accum, unsigned left, unsigned right, uint16_t limit)
{
	const uint16_t * RESTRICT src_p0 = src[0];
	const uint16_t * RESTRICT src_p1 = src[1];
	const uint16_t * RESTRICT src_p2 = src[2];
	const uint16_t * RESTRICT src_p3 = src[3];
	const uint16_t * RESTRICT src_p4 = src[4];
	const uint16_t * RESTRICT src_p5 = src[5];
	const uint16_t * RESTRICT src_p6 = src[6];
	const uint16_t * RESTRICT src_p7 = src[7];

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);
	unsigned accum_base = floor_n(left, 8);

	const __m128i c01 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[0]), _mm_set1_epi16(filter_data[1]));
	const __m128i c23 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[2]), _mm_set1_epi16(filter_data[3]));
	const __m128i c45 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[4]), _mm_set1_epi16(filter_data[5]));
	const __m128i c67 = _mm_unpacklo_epi16(_mm_set1_epi16(filter_data[6]), _mm_set1_epi16(filter_data[7]));

	__m128i out;

#define XITER resize_line_v_u16_sse2_xiter<Taps, AccumMode>
#define XARGS accum_base, src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, accum, c01, c23, c45, c67, limit
	if (left != vec_left) {
		out = XITER(vec_left - 8, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			mm_store_idxhi_epi16((__m128i *)(dst + vec_left - 8), out, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		out = XITER(j, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			_mm_store_si128((__m128i *)(dst + j), out);
	}

	if (right != vec_right) {
		out = XITER(vec_right, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			mm_store_idxlo_epi16((__m128i *)(dst + vec_right), out, right % 8);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line_v_u16_sse2_jt_small = make_array(
	resize_line_v_u16_sse2<2, V_ACCUM_NONE>,
	resize_line_v_u16_sse2<2, V_ACCUM_NONE>,
	resize_line_v_u16_sse2<4, V_ACCUM_NONE>,
	resize_line_v_u16_sse2<4, V_ACCUM_NONE>,
	resize_line_v_u16_sse2<6, V_ACCUM_NONE>,
	resize_line_v_u16_sse2<6, V_ACCUM_NONE>,
	resize_line_v_u16_sse2<8, V_ACCUM_NONE>,
	resize_line_v_u16_sse2<8, V_ACCUM_NONE>);

constexpr auto resize_line_v_u16_sse2_initial = resize_line_v_u16_sse2<8, V_ACCUM_INITIAL>;
constexpr auto resize_line_v_u16_sse2_update = resize_line_v_u16_sse2<8, V_ACCUM_UPDATE>;

constexpr auto resize_line_v_u16_sse2_jt_final = make_array(
	resize_line_v_u16_sse2<2, V_ACCUM_FINAL>,
	resize_line_v_u16_sse2<2, V_ACCUM_FINAL>,
	resize_line_v_u16_sse2<4, V_ACCUM_FINAL>,
	resize_line_v_u16_sse2<4, V_ACCUM_FINAL>,
	resize_line_v_u16_sse2<6, V_ACCUM_FINAL>,
	resize_line_v_u16_sse2<6, V_ACCUM_FINAL>,
	resize_line_v_u16_sse2<8, V_ACCUM_FINAL>,
	resize_line_v_u16_sse2<8, V_ACCUM_FINAL>);


class ResizeImplH_U16_SSE2 final : public ResizeImplH {
	decltype(resize_line8_h_u16_sse2_jt_small)::value_type m_func;
	uint16_t m_pixel_max;
public:
	ResizeImplH_U16_SSE2(const FilterContext &filter, unsigned height, unsigned depth) try :
		ResizeImplH(filter, height, PixelType::WORD),
		m_func{},
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{
		m_desc.step = 8;
		m_desc.scratchpad_size = (ceil_n(checked_size_t{ filter.input_width }, 8) * sizeof(uint16_t) * 8).get();

		if (filter.filter_width <= 8)
			m_func = resize_line8_h_u16_sse2_jt_small[filter.filter_width - 1];
		else
			m_func = resize_line8_h_u16_sse2_jt_large[filter.filter_width % 8];
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		auto range = get_col_deps(left, right);

		const uint16_t *src_ptr[8] = { 0 };
		uint16_t *dst_ptr[8] = { 0 };
		uint16_t *transpose_buf = static_cast<uint16_t *>(tmp);
		unsigned height = m_desc.format.height;

		for (unsigned n = 0; n < 8; ++n) {
			src_ptr[n] = in->get_line<uint16_t>(std::min(i + n, height - 1));
		}

		transpose_line_8x8_epi16(transpose_buf, src_ptr, floor_n(range.first, 8), ceil_n(range.second, 8));

		for (unsigned n = 0; n < 8; ++n) {
			dst_ptr[n] = out->get_line<uint16_t>(std::min(i + n, height - 1));
		}

		m_func(m_filter.left.data(), m_filter.data_i16.data(), m_filter.stride_i16, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 8), left, right, m_pixel_max);
	}
};


class ResizeImplV_U16_SSE2 final : public ResizeImplV {
	uint16_t m_pixel_max;
public:
	ResizeImplV_U16_SSE2(const FilterContext &filter, unsigned width, unsigned depth) try :
		ResizeImplV(filter, width, PixelType::WORD),
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{
		if (m_filter.filter_width > 8)
			m_desc.scratchpad_size = (ceil_n(checked_size_t{ width }, 8) * sizeof(uint32_t)).get();
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		const int16_t *filter_data = m_filter.data_i16.data() + i * m_filter.stride_i16;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const uint16_t *src_lines[8] = { 0 };
		uint16_t *dst_line = out->get_line<uint16_t>(i);
		uint32_t *accum_buf = static_cast<uint32_t *>(tmp);

		unsigned top = m_filter.left[i];

		auto gather_8_lines = [&](unsigned i)
		{
			for (unsigned n = 0; n < 8; ++n) {
				src_lines[n] = in->get_line<uint16_t>(std::min(i + n, src_height - 1));
			}
		};

#define XARGS src_lines, dst_line, accum_buf, left, right, m_pixel_max
		if (filter_width <= 8) {
			gather_8_lines(top);
			resize_line_v_u16_sse2_jt_small[filter_width - 1](filter_data, XARGS);
		} else {
			unsigned k_end = ceil_n(filter_width, 8) - 8;

			gather_8_lines(top);
			resize_line_v_u16_sse2_initial(filter_data + 0, XARGS);

			for (unsigned k = 8; k < k_end; k += 8) {
				gather_8_lines(top + k);
				resize_line_v_u16_sse2_update(filter_data + k, XARGS);
			}

			gather_8_lines(top + k_end);
			resize_line_v_u16_sse2_jt_final[filter_width % 8](filter_data + k_end, XARGS);
		}
#undef XARGS
	}
};

} // namespace


std::unique_ptr<graphengine::Filter> create_resize_impl_h_sse2(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

	if (type == PixelType::WORD)
		ret = std::make_unique<ResizeImplH_U16_SSE2>(context, height, depth);

	return ret;
}

std::unique_ptr<graphengine::Filter> create_resize_impl_v_sse2(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

	if (type == PixelType::WORD)
		ret = std::make_unique<ResizeImplV_U16_SSE2>(context, width, depth);

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
