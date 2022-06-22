#ifdef ZIMG_X86

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/make_array.h"
#include "common/pixel.h"
#include "common/x86/cpuinfo_x86.h"
#include "graphengine/filter.h"
#include "resize/filter.h"
#include "resize/resize_impl.h"
#include "resize_impl_x86.h"

#include "common/x86/sse_util.h"
#include "common/x86/sse2_util.h"
#include "common/x86/avx_util.h"
#include "common/x86/avx2_util.h"

namespace zimg {
namespace resize {

namespace {

struct f16_traits {
	typedef __m128i vec8_type;
	typedef uint16_t pixel_type;

	static constexpr PixelType type_constant = PixelType::HALF;

	static inline FORCE_INLINE vec8_type load8_raw(const pixel_type *ptr)
	{
		return _mm_loadu_si128((const __m128i *)ptr);
	}

	static inline FORCE_INLINE void store8_raw(pixel_type *ptr, vec8_type x)
	{
		_mm_store_si128((__m128i *)ptr, x);
	}

	static inline FORCE_INLINE __m256 load8(const pixel_type *ptr)
	{
		return _mm256_cvtph_ps(load8_raw(ptr));
	}

	static inline FORCE_INLINE void store8(pixel_type *ptr, __m256 x)
	{
		store8_raw(ptr, _mm256_cvtps_ph(x, 0));
	}

	static inline FORCE_INLINE void transpose8(vec8_type &x0, vec8_type &x1, vec8_type &x2, vec8_type &x3,
	                                           vec8_type &x4, vec8_type &x5, vec8_type &x6, vec8_type &x7)
	{
		mm_transpose8_epi16(x0, x1, x2, x3, x4, x5, x6, x7);
	}

	static inline FORCE_INLINE void scatter8(pixel_type *dst0, pixel_type *dst1, pixel_type *dst2, pixel_type *dst3,
	                                         pixel_type *dst4, pixel_type *dst5, pixel_type *dst6, pixel_type *dst7, __m256 x)
	{
		mm_scatter_epi16(dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7, _mm256_cvtps_ph(x, 0));
	}

	static inline FORCE_INLINE void store_idxlo(pixel_type *dst, __m256 x, unsigned idx)
	{
		mm_store_idxlo_epi16((__m128i *)dst, _mm256_cvtps_ph(x, 0), idx);
	}

	static inline FORCE_INLINE void store_idxhi(pixel_type *dst, __m256 x, unsigned idx)
	{
		mm_store_idxhi_epi16((__m128i *)dst, _mm256_cvtps_ph(x, 0), idx);
	}
};

struct f32_traits {
	typedef __m256 vec8_type;
	typedef float pixel_type;

	static constexpr PixelType type_constant = PixelType::FLOAT;

	static inline FORCE_INLINE vec8_type load8_raw(const pixel_type *ptr)
	{
		return _mm256_loadu_ps(ptr);
	}

	static inline FORCE_INLINE void store8_raw(pixel_type *ptr, vec8_type x)
	{
		_mm256_store_ps(ptr, x);
	}

	static inline FORCE_INLINE __m256 load8(const pixel_type *ptr)
	{
		return load8_raw(ptr);
	}

	static inline FORCE_INLINE void store8(pixel_type *ptr, __m256 x)
	{
		store8_raw(ptr, x);
	}

	static inline FORCE_INLINE void transpose8(vec8_type &x0, vec8_type &x1, vec8_type &x2, vec8_type &x3,
	                                           vec8_type &x4, vec8_type &x5, vec8_type &x6, vec8_type &x7)
	{
		mm256_transpose8_ps(x0, x1, x2, x3, x4, x5, x6, x7);
	}

	static inline FORCE_INLINE void scatter8(pixel_type *dst0, pixel_type *dst1, pixel_type *dst2, pixel_type *dst3,
	                                         pixel_type *dst4, pixel_type *dst5, pixel_type *dst6, pixel_type *dst7, __m256 x)
	{
		mm_scatter_ps(dst0, dst1, dst2, dst3, _mm256_castps256_ps128(x));
		mm_scatter_ps(dst4, dst5, dst6, dst7, _mm256_extractf128_ps(x, 1));
	}

	static inline FORCE_INLINE void store_idxlo(pixel_type *dst, __m256 x, unsigned idx)
	{
		mm256_store_idxlo_ps(dst, x, idx);
	}

	static inline FORCE_INLINE void store_idxhi(pixel_type *dst, __m256 x, unsigned idx)
	{
		mm256_store_idxhi_ps(dst, x, idx);
	}
};


inline FORCE_INLINE __m256i export_i30_u16(__m256i lo, __m256i hi)
{
	const __m256i round = _mm256_set1_epi32(1 << 13);

	lo = _mm256_add_epi32(lo, round);
	hi = _mm256_add_epi32(hi, round);

	lo = _mm256_srai_epi32(lo, 14);
	hi = _mm256_srai_epi32(hi, 14);

	lo = _mm256_packs_epi32(lo, hi);

	return lo;
}


template <class Traits, class T>
void transpose_line_8x8(T * RESTRICT dst, const T * const * RESTRICT src, unsigned left, unsigned right)
{
	typedef typename Traits::vec8_type vec8_type;

	for (unsigned j = left; j < right; j += 8) {
		vec8_type x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = Traits::load8_raw(src[0] + j);
		x1 = Traits::load8_raw(src[1] + j);
		x2 = Traits::load8_raw(src[2] + j);
		x3 = Traits::load8_raw(src[3] + j);
		x4 = Traits::load8_raw(src[4] + j);
		x5 = Traits::load8_raw(src[5] + j);
		x6 = Traits::load8_raw(src[6] + j);
		x7 = Traits::load8_raw(src[7] + j);

		Traits::transpose8(x0, x1, x2, x3, x4, x5, x6, x7);

		Traits::store8_raw(dst + 0, x0);
		Traits::store8_raw(dst + 8, x1);
		Traits::store8_raw(dst + 16, x2);
		Traits::store8_raw(dst + 24, x3);
		Traits::store8_raw(dst + 32, x4);
		Traits::store8_raw(dst + 40, x5);
		Traits::store8_raw(dst + 48, x6);
		Traits::store8_raw(dst + 56, x7);

		dst += 64;
	}
}

void transpose_line_16x16_epi16(uint16_t * RESTRICT dst, const uint16_t * const * RESTRICT src, unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; j += 16) {
		__m256i x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

		x0 = _mm256_load_si256((const __m256i *)(src[0] + j));
		x1 = _mm256_load_si256((const __m256i *)(src[1] + j));
		x2 = _mm256_load_si256((const __m256i *)(src[2] + j));
		x3 = _mm256_load_si256((const __m256i *)(src[3] + j));
		x4 = _mm256_load_si256((const __m256i *)(src[4] + j));
		x5 = _mm256_load_si256((const __m256i *)(src[5] + j));
		x6 = _mm256_load_si256((const __m256i *)(src[6] + j));
		x7 = _mm256_load_si256((const __m256i *)(src[7] + j));
		x8 = _mm256_load_si256((const __m256i *)(src[8] + j));
		x9 = _mm256_load_si256((const __m256i *)(src[9] + j));
		x10 = _mm256_load_si256((const __m256i *)(src[10] + j));
		x11 = _mm256_load_si256((const __m256i *)(src[11] + j));
		x12 = _mm256_load_si256((const __m256i *)(src[12] + j));
		x13 = _mm256_load_si256((const __m256i *)(src[13] + j));
		x14 = _mm256_load_si256((const __m256i *)(src[14] + j));
		x15 = _mm256_load_si256((const __m256i *)(src[15] + j));

		mm256_transpose16_epi16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);

		_mm256_store_si256((__m256i *)(dst + 0), x0);
		_mm256_store_si256((__m256i *)(dst + 16), x1);
		_mm256_store_si256((__m256i *)(dst + 32), x2);
		_mm256_store_si256((__m256i *)(dst + 48), x3);
		_mm256_store_si256((__m256i *)(dst + 64), x4);
		_mm256_store_si256((__m256i *)(dst + 80), x5);
		_mm256_store_si256((__m256i *)(dst + 96), x6);
		_mm256_store_si256((__m256i *)(dst + 112), x7);
		_mm256_store_si256((__m256i *)(dst + 128), x8);
		_mm256_store_si256((__m256i *)(dst + 144), x9);
		_mm256_store_si256((__m256i *)(dst + 160), x10);
		_mm256_store_si256((__m256i *)(dst + 176), x11);
		_mm256_store_si256((__m256i *)(dst + 192), x12);
		_mm256_store_si256((__m256i *)(dst + 208), x13);
		_mm256_store_si256((__m256i *)(dst + 224), x14);
		_mm256_store_si256((__m256i *)(dst + 240), x15);

		dst += 256;
	}
}


template <int Taps>
inline FORCE_INLINE __m256i resize_line8_h_u16_avx2_xiter(unsigned j,
                                                          const unsigned * RESTRICT filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                                          const uint16_t * RESTRICT src, unsigned src_base, uint16_t limit)
{
	static_assert(Taps <= 8, "only up to 8 taps can be unrolled");
	static_assert(Taps >= -6, "only up to 6 taps in epilogue");
	static_assert(Taps % 2 == 0, "tap count must be even");
	constexpr int Tail = Taps > 0 ? Taps : -Taps;

	const __m256i i16_min = _mm256_set1_epi16(INT16_MIN);
	const __m256i lim = _mm256_set1_epi16(limit + INT16_MIN);

	const int16_t *filter_coeffs = filter_data + j * filter_stride;
	const uint16_t *src_p = src + (filter_left[j] - src_base) * 16;

	__m256i accum_lo = _mm256_setzero_si256();
	__m256i accum_hi = _mm256_setzero_si256();
	__m256i x0, x1, xl, xh, c, coeffs;

	unsigned k_end = Taps > 0 ? 0 : floor_n(filter_width + 1, 8);

	for (unsigned k = 0; k < k_end; k += 8) {
		coeffs = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(filter_coeffs + k)));

		c = _mm256_shuffle_epi32(coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x0 = _mm256_load_si256((const __m256i *)(src_p + 0));
		x1 = _mm256_load_si256((const __m256i *)(src_p + 16));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c, xl);
		xh = _mm256_madd_epi16(c, xh);

		accum_lo = _mm256_add_epi32(accum_lo, xl);
		accum_hi = _mm256_add_epi32(accum_hi, xh);

		c = _mm256_shuffle_epi32(coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x0 = _mm256_load_si256((const __m256i *)(src_p + 32));
		x1 = _mm256_load_si256((const __m256i *)(src_p + 48));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c, xl);
		xh = _mm256_madd_epi16(c, xh);

		accum_lo = _mm256_add_epi32(accum_lo, xl);
		accum_hi = _mm256_add_epi32(accum_hi, xh);

		c = _mm256_shuffle_epi32(coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x0 = _mm256_load_si256((const __m256i *)(src_p + 64));
		x1 = _mm256_load_si256((const __m256i *)(src_p + 80));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c, xl);
		xh = _mm256_madd_epi16(c, xh);

		accum_lo = _mm256_add_epi32(accum_lo, xl);
		accum_hi = _mm256_add_epi32(accum_hi, xh);

		c = _mm256_shuffle_epi32(coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x0 = _mm256_load_si256((const __m256i *)(src_p + 96));
		x1 = _mm256_load_si256((const __m256i *)(src_p + 112));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c, xl);
		xh = _mm256_madd_epi16(c, xh);

		accum_lo = _mm256_add_epi32(accum_lo, xl);
		accum_hi = _mm256_add_epi32(accum_hi, xh);

		src_p += 128;
	}

	if (Tail >= 2) {
		coeffs = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(filter_coeffs + k_end)));

		c = _mm256_shuffle_epi32(coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x0 = _mm256_load_si256((const __m256i *)(src_p + 0));
		x1 = _mm256_load_si256((const __m256i *)(src_p + 16));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c, xl);
		xh = _mm256_madd_epi16(c, xh);

		accum_lo = _mm256_add_epi32(accum_lo, xl);
		accum_hi = _mm256_add_epi32(accum_hi, xh);
	}

	if (Tail >= 4) {
		c = _mm256_shuffle_epi32(coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x0 = _mm256_load_si256((const __m256i *)(src_p + 32));
		x1 = _mm256_load_si256((const __m256i *)(src_p + 48));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c, xl);
		xh = _mm256_madd_epi16(c, xh);

		accum_lo = _mm256_add_epi32(accum_lo, xl);
		accum_hi = _mm256_add_epi32(accum_hi, xh);
	}

	if (Tail >= 6) {
		c = _mm256_shuffle_epi32(coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x0 = _mm256_load_si256((const __m256i *)(src_p + 64));
		x1 = _mm256_load_si256((const __m256i *)(src_p + 80));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c, xl);
		xh = _mm256_madd_epi16(c, xh);

		accum_lo = _mm256_add_epi32(accum_lo, xl);
		accum_hi = _mm256_add_epi32(accum_hi, xh);
	}

	if (Tail >= 8) {
		c = _mm256_shuffle_epi32(coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x0 = _mm256_load_si256((const __m256i *)(src_p + 96));
		x1 = _mm256_load_si256((const __m256i *)(src_p + 112));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c, xl);
		xh = _mm256_madd_epi16(c, xh);

		accum_lo = _mm256_add_epi32(accum_lo, xl);
		accum_hi = _mm256_add_epi32(accum_hi, xh);
	}

	accum_lo = export_i30_u16(accum_lo, accum_hi);
	accum_lo = _mm256_min_epi16(accum_lo, lim);
	accum_lo = _mm256_sub_epi16(accum_lo, i16_min);
	return accum_lo;
}

template <int Taps>
void resize_line8_h_u16_avx2(const unsigned * RESTRICT filter_left, const int16_t * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                             const uint16_t * RESTRICT src, uint16_t * const * /* RESTRICT */ dst, unsigned src_base, unsigned left, unsigned right, uint16_t limit)
{
	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

#define XITER resize_line8_h_u16_avx2_xiter<Taps>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src, src_base, limit
	for (unsigned j = left; j < vec_left; ++j) {
		__m256i x = XITER(j, XARGS);

		mm_scatter_epi16(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, dst[4] + j, dst[5] + j, dst[6] + j, dst[7] + j, _mm256_castsi256_si128(x));
		mm_scatter_epi16(dst[8] + j, dst[9] + j, dst[10] + j, dst[11] + j, dst[12] + j, dst[13] + j, dst[14] + j, dst[15] + j, _mm256_extractf128_si256(x, 1));
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		uint16_t cache alignas(32)[16][16];
		__m256i x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

		for (unsigned jj = j; jj < j + 16; ++jj) {
			__m256i x = XITER(jj, XARGS);
			_mm256_store_si256((__m256i *)cache[jj - j], x);
		}

		x0 = _mm256_load_si256((const __m256i *)cache[0]);
		x1 = _mm256_load_si256((const __m256i *)cache[1]);
		x2 = _mm256_load_si256((const __m256i *)cache[2]);
		x3 = _mm256_load_si256((const __m256i *)cache[3]);
		x4 = _mm256_load_si256((const __m256i *)cache[4]);
		x5 = _mm256_load_si256((const __m256i *)cache[5]);
		x6 = _mm256_load_si256((const __m256i *)cache[6]);
		x7 = _mm256_load_si256((const __m256i *)cache[7]);
		x8 = _mm256_load_si256((const __m256i *)cache[8]);
		x9 = _mm256_load_si256((const __m256i *)cache[9]);
		x10 = _mm256_load_si256((const __m256i *)cache[10]);
		x11 = _mm256_load_si256((const __m256i *)cache[11]);
		x12 = _mm256_load_si256((const __m256i *)cache[12]);
		x13 = _mm256_load_si256((const __m256i *)cache[13]);
		x14 = _mm256_load_si256((const __m256i *)cache[14]);
		x15 = _mm256_load_si256((const __m256i *)cache[15]);

		mm256_transpose16_epi16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);

		_mm256_store_si256((__m256i *)(dst[0] + j), x0);
		_mm256_store_si256((__m256i *)(dst[1] + j), x1);
		_mm256_store_si256((__m256i *)(dst[2] + j), x2);
		_mm256_store_si256((__m256i *)(dst[3] + j), x3);
		_mm256_store_si256((__m256i *)(dst[4] + j), x4);
		_mm256_store_si256((__m256i *)(dst[5] + j), x5);
		_mm256_store_si256((__m256i *)(dst[6] + j), x6);
		_mm256_store_si256((__m256i *)(dst[7] + j), x7);
		_mm256_store_si256((__m256i *)(dst[8] + j), x8);
		_mm256_store_si256((__m256i *)(dst[9] + j), x9);
		_mm256_store_si256((__m256i *)(dst[10] + j), x10);
		_mm256_store_si256((__m256i *)(dst[11] + j), x11);
		_mm256_store_si256((__m256i *)(dst[12] + j), x12);
		_mm256_store_si256((__m256i *)(dst[13] + j), x13);
		_mm256_store_si256((__m256i *)(dst[14] + j), x14);
		_mm256_store_si256((__m256i *)(dst[15] + j), x15);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		__m256i x = XITER(j, XARGS);

		mm_scatter_epi16(dst[0] + j, dst[1] + j, dst[2] + j, dst[3] + j, dst[4] + j, dst[5] + j, dst[6] + j, dst[7] + j, _mm256_castsi256_si128(x));
		mm_scatter_epi16(dst[8] + j, dst[9] + j, dst[10] + j, dst[11] + j, dst[12] + j, dst[13] + j, dst[14] + j, dst[15] + j, _mm256_extractf128_si256(x, 1));
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line8_h_u16_avx2_jt_small = make_array(
	resize_line8_h_u16_avx2<2>,
	resize_line8_h_u16_avx2<2>,
	resize_line8_h_u16_avx2<4>,
	resize_line8_h_u16_avx2<4>,
	resize_line8_h_u16_avx2<6>,
	resize_line8_h_u16_avx2<6>,
	resize_line8_h_u16_avx2<8>,
	resize_line8_h_u16_avx2<8>);

constexpr auto resize_line8_h_u16_avx2_jt_large = make_array(
	resize_line8_h_u16_avx2<0>,
	resize_line8_h_u16_avx2<-2>,
	resize_line8_h_u16_avx2<-2>,
	resize_line8_h_u16_avx2<-4>,
	resize_line8_h_u16_avx2<-4>,
	resize_line8_h_u16_avx2<-6>,
	resize_line8_h_u16_avx2<-6>,
	resize_line8_h_u16_avx2<0>);


template <class Traits, int Taps>
inline FORCE_INLINE __m256 resize_line8_h_fp_avx2_xiter(unsigned j,
                                                        const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                                        const typename Traits::pixel_type * RESTRICT src, unsigned src_base)
{
	static_assert(Taps <= 8, "only up to 8 taps can be unrolled");
	static_assert(Taps >= -3, "only up to 3 taps in epilogue");
	constexpr int Tail = Taps >= 4 ? Taps - 4 : Taps > 0 ? Taps : -Taps;

	typedef typename Traits::pixel_type pixel_type;

	const float *filter_coeffs = filter_data + j * filter_stride;
	const pixel_type *src_p = src + (filter_left[j] - src_base) * 8;

	__m256 accum0 = _mm256_setzero_ps();
	__m256 accum1 = _mm256_setzero_ps();
	__m256 x, c, coeffs;

	unsigned k_end = Taps >= 4 ? 4 : Taps > 0 ? 0 : floor_n(filter_width, 4);

	for (unsigned k = 0; k < k_end; k += 4) {
		coeffs = _mm256_broadcast_ps((const __m128 *)(filter_coeffs + k));

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x = Traits::load8(src_p + 0);
		accum0 = _mm256_fmadd_ps(c, x, accum0);

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x = Traits::load8(src_p + 8);
		accum1 = _mm256_fmadd_ps(c, x, accum1);

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x = Traits::load8(src_p + 16);
		accum0 = _mm256_fmadd_ps(c, x, accum0);

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x = Traits::load8(src_p + 24);
		accum1 = _mm256_fmadd_ps(c, x, accum1);

		src_p += 32;
	}

	if (Tail >= 1) {
		coeffs = _mm256_broadcast_ps((const __m128 *)(filter_coeffs + k_end));

		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x = Traits::load8(src_p + 0);
		accum0 = _mm256_fmadd_ps(c, x, accum0);
	}
	if (Tail >= 2) {
		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x = Traits::load8(src_p + 8);
		accum1 = _mm256_fmadd_ps(c, x, accum1);
	}
	if (Tail >= 3) {
		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x = Traits::load8(src_p + 16);
		accum0 = _mm256_fmadd_ps(c, x, accum0);
	}
	if (Tail >= 4) {
		c = _mm256_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x = Traits::load8(src_p + 24);
		accum1 = _mm256_fmadd_ps(c, x, accum1);
	}

	if (Taps <= 0 || Taps >= 2)
		accum0 = _mm256_add_ps(accum0, accum1);

	return accum0;
}

template <class Traits, int Taps>
void resize_line8_h_fp_avx2(const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                            const typename Traits::pixel_type * RESTRICT src, typename Traits::pixel_type * const * RESTRICT dst, unsigned src_base, unsigned left, unsigned right)
{
	typedef typename Traits::pixel_type pixel_type;

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	pixel_type *dst_p0 = dst[0];
	pixel_type *dst_p1 = dst[1];
	pixel_type *dst_p2 = dst[2];
	pixel_type *dst_p3 = dst[3];
	pixel_type *dst_p4 = dst[4];
	pixel_type *dst_p5 = dst[5];
	pixel_type *dst_p6 = dst[6];
	pixel_type *dst_p7 = dst[7];

#define XITER resize_line8_h_fp_avx2_xiter<Traits, Taps>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src, src_base
	for (unsigned j = left; j < vec_left; ++j) {
		__m256 x = XITER(j, XARGS);
		Traits::scatter8(dst_p0 + j, dst_p1 + j, dst_p2 + j, dst_p3 + j, dst_p4 + j, dst_p5 + j, dst_p6 + j, dst_p7 + j, x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m256 x0, x1, x2, x3, x4, x5, x6, x7;

		x0 = XITER(j + 0, XARGS);
		x1 = XITER(j + 1, XARGS);
		x2 = XITER(j + 2, XARGS);
		x3 = XITER(j + 3, XARGS);
		x4 = XITER(j + 4, XARGS);
		x5 = XITER(j + 5, XARGS);
		x6 = XITER(j + 6, XARGS);
		x7 = XITER(j + 7, XARGS);

		mm256_transpose8_ps(x0, x1, x2, x3, x4, x5, x6, x7);

		Traits::store8(dst_p0 + j, x0);
		Traits::store8(dst_p1 + j, x1);
		Traits::store8(dst_p2 + j, x2);
		Traits::store8(dst_p3 + j, x3);
		Traits::store8(dst_p4 + j, x4);
		Traits::store8(dst_p5 + j, x5);
		Traits::store8(dst_p6 + j, x6);
		Traits::store8(dst_p7 + j, x7);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		__m256 x = XITER(j, XARGS);
		Traits::scatter8(dst_p0 + j, dst_p1 + j, dst_p2 + j, dst_p3 + j, dst_p4 + j, dst_p5 + j, dst_p6 + j, dst_p7 + j, x);
	}
#undef XITER
#undef XARGS
}

template <class Traits>
constexpr auto resize_line8_h_fp_avx2_jt_small = make_array(
	resize_line8_h_fp_avx2<Traits, 1>,
	resize_line8_h_fp_avx2<Traits, 2>,
	resize_line8_h_fp_avx2<Traits, 3>,
	resize_line8_h_fp_avx2<Traits, 4>,
	resize_line8_h_fp_avx2<Traits, 5>,
	resize_line8_h_fp_avx2<Traits, 6>,
	resize_line8_h_fp_avx2<Traits, 7>,
	resize_line8_h_fp_avx2<Traits, 8>);

template <class Traits>
constexpr auto resize_line8_h_fp_avx2_jt_large = make_array(
	resize_line8_h_fp_avx2<Traits, 0>,
	resize_line8_h_fp_avx2<Traits, -1>,
	resize_line8_h_fp_avx2<Traits, -2>,
	resize_line8_h_fp_avx2<Traits, -3>);


template <unsigned Taps>
void resize_line_h_perm_u16_avx2(const unsigned * RESTRICT permute_left, const unsigned * RESTRICT permute_mask, const int16_t * RESTRICT filter_data, unsigned input_width,
                                 const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, unsigned left, unsigned right, uint16_t limit)
{
	static_assert(Taps <= 10, "permuted resampler only supports up to 10 taps");

	const __m256i i16_min = _mm256_set1_epi16(INT16_MIN);
	const __m256i lim = _mm256_set1_epi16(limit + INT16_MIN);

	unsigned vec_right = floor_n(right, 8);
	unsigned fallback_idx = vec_right;

	for (unsigned j = floor_n(left, 8); j < vec_right; j += 8) {
		unsigned left = permute_left[j / 8];

		if (input_width - left < 24) {
			fallback_idx = j;
			break;
		}

		const __m256i mask = _mm256_load_si256((const __m256i *)(permute_mask + j));
		const int16_t *data = filter_data + static_cast<size_t>(j) * Taps;

		__m256i accum0 = _mm256_setzero_si256();
		__m256i accum1 = _mm256_setzero_si256();
		__m256i x, x0, x8, coeffs;

		if (Taps >= 2) {
			x0 = _mm256_loadu_si256((const __m256i *)(src + left + 0));
			x0 = _mm256_add_epi16(x0, i16_min);

			x = x0;
			x = _mm256_permutevar8x32_epi32(x, mask);
			coeffs = _mm256_load_si256((__m256i *)(data + 0 * 8));
			x = _mm256_madd_epi16(coeffs, x);
			accum0 = _mm256_add_epi32(accum0, x);
		}
		if (Taps >= 4) {
			x8 = _mm256_loadu_si256((const __m256i *)(src + left + 8));
			x8 = _mm256_add_epi16(x8, i16_min);

			x = _mm256_alignr_epi8(x8, x0, 4);
			x = _mm256_permutevar8x32_epi32(x, mask);
			coeffs = _mm256_load_si256((const __m256i *)(data + 2 * 8));
			x = _mm256_madd_epi16(coeffs, x);
			accum1 = _mm256_add_epi32(accum1, x);
		}
		if (Taps >= 6) {
			x = _mm256_alignr_epi8(x8, x0, 8);
			x = _mm256_permutevar8x32_epi32(x, mask);
			coeffs = _mm256_load_si256((const __m256i *)(data + 4 * 8));
			x = _mm256_madd_epi16(coeffs, x);
			accum0 = _mm256_add_epi32(accum0, x);
		}
		if (Taps >= 8) {
			x = _mm256_alignr_epi8(x8, x0, 12);
			x = _mm256_permutevar8x32_epi32(x, mask);
			coeffs = _mm256_load_si256((const __m256i *)(data + 6 * 8));
			x = _mm256_madd_epi16(coeffs, x);
			accum1 = _mm256_add_epi32(accum1, x);
		}
		if (Taps >= 10) {
			x = x8;
			x = _mm256_permutevar8x32_epi32(x, mask);
			coeffs = _mm256_load_si256((const __m256i *)(data + 8 * 8));
			x = _mm256_madd_epi16(coeffs, x);
			accum0 = _mm256_add_epi32(accum0, x);
		}

		accum0 = _mm256_add_epi32(accum0, accum1);
		accum0 = export_i30_u16(accum0, accum0);
		accum0 = _mm256_min_epi16(accum0, lim);
		accum0 = _mm256_sub_epi16(accum0, i16_min);
		accum0 = _mm256_permute4x64_epi64(accum0, _MM_SHUFFLE(3, 1, 2, 0));

		_mm_store_si128((__m128i *)(dst + j), _mm256_castsi256_si128(accum0));
	}
	for (unsigned j = fallback_idx; j < right; j += 8) {
		unsigned left = permute_left[j / 8];
		const int16_t *data = filter_data + static_cast<size_t>(j) * Taps;

		__m256i accum = _mm256_setzero_si256();
		__m256i x, coeffs;

		for (unsigned k = 0; k < Taps; k += 2) {
			alignas(32) uint16_t tmp[16];

			for (unsigned kk = 0; kk < 8; ++kk) {
				unsigned idx = left + permute_mask[j + kk] * 2 + k;

				tmp[kk * 2 + 0] = src[idx + 0];
				tmp[kk * 2 + 1] = src[std::min(idx + 1, input_width)];
			}

			x = _mm256_loadu_si256((const __m256i *)tmp);
			x = _mm256_add_epi16(x, i16_min);
			coeffs = _mm256_load_si256((const __m256i *)(data + k * 8));
			x = _mm256_madd_epi16(coeffs, x);
			accum = _mm256_add_epi32(accum, x);
		}

		accum = export_i30_u16(accum, accum);
		accum = _mm256_min_epi16(accum, lim);
		accum = _mm256_sub_epi16(accum, i16_min);
		accum = _mm256_permute4x64_epi64(accum, _MM_SHUFFLE(3, 1, 2, 0));

		_mm_store_si128((__m128i *)(dst + j), _mm256_castsi256_si128(accum));
	}
}

constexpr auto resize_line_h_perm_u16_avx2_jt = make_array(
	resize_line_h_perm_u16_avx2<2>,
	resize_line_h_perm_u16_avx2<4>,
	resize_line_h_perm_u16_avx2<6>,
	resize_line_h_perm_u16_avx2<8>,
	resize_line_h_perm_u16_avx2<10>);


template <class Traits, unsigned Taps>
void resize_line_h_perm_fp_avx2(const unsigned * RESTRICT permute_left, const unsigned * RESTRICT permute_mask, const float * RESTRICT filter_data, unsigned input_width,
                                const typename Traits::pixel_type * RESTRICT src, typename Traits::pixel_type * RESTRICT dst, unsigned left, unsigned right)
{
	static_assert(Taps <= 8, "permuted resampler only supports up to 8 taps");

	unsigned vec_right = floor_n(right, 8);
	unsigned fallback_idx = vec_right;

#define mm256_alignr_epi8_ps(a, b, imm) _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256((a)), _mm256_castps_si256((b)), (imm)))
	for (unsigned j = floor_n(left, 8); j < vec_right; j += 8) {
		unsigned left = permute_left[j / 8];

		if (input_width - left < (Taps >= 6 ? 16 : 12)) {
			fallback_idx = j;
			break;
		}

		const __m256i mask = _mm256_load_si256((const __m256i *)(permute_mask + j));
		const float *data = filter_data + static_cast<size_t>(j) * Taps;

		__m256 accum0 = _mm256_setzero_ps();
		__m256 accum1 = _mm256_setzero_ps();
		__m256 x, x0, x4, x8, coeffs;

		if (Taps >= 1) {
			x0 = Traits::load8(src + left + 0);

			x = x0;
			x = _mm256_permutevar8x32_ps(x, mask);
			coeffs = _mm256_load_ps(data + 0 * 8);
			accum0 = _mm256_fmadd_ps(coeffs, x, accum0);
		}
		if (Taps >= 2) {
			x4 = Traits::load8(src + left + 4);

			x = mm256_alignr_epi8_ps(x4, x0, 4);
			x = _mm256_permutevar8x32_ps(x, mask);
			coeffs = _mm256_load_ps(data + 1 * 8);
			accum1 = _mm256_fmadd_ps(coeffs, x, accum1);
		}
		if (Taps >= 3) {
			x = mm256_alignr_epi8_ps(x4, x0, 8);
			x = _mm256_permutevar8x32_ps(x, mask);
			coeffs = _mm256_load_ps(data + 2 * 8);
			accum0 = _mm256_fmadd_ps(coeffs, x, accum0);
		}
		if (Taps >= 4) {
			x = mm256_alignr_epi8_ps(x4, x0, 12);
			x = _mm256_permutevar8x32_ps(x, mask);
			coeffs = _mm256_load_ps(data + 3 * 8);
			accum1 = _mm256_fmadd_ps(coeffs, x, accum1);
		}
		if (Taps >= 5) {
			x = x4;
			x = _mm256_permutevar8x32_ps(x, mask);
			coeffs = _mm256_load_ps(data + 4 * 8);
			accum0 = _mm256_fmadd_ps(coeffs, x, accum0);
		}
		if (Taps >= 6) {
			x8 = Traits::load8(src + left + 8);

			x = mm256_alignr_epi8_ps(x8, x4, 4);
			x = _mm256_permutevar8x32_ps(x, mask);
			coeffs = _mm256_load_ps(data + 5 * 8);
			accum1 = _mm256_fmadd_ps(coeffs, x, accum1);
		}
		if (Taps >= 7) {
			x = mm256_alignr_epi8_ps(x8, x4, 8);
			x = _mm256_permutevar8x32_ps(x, mask);
			coeffs = _mm256_load_ps(data + 6 * 8);
			accum0 = _mm256_fmadd_ps(coeffs, x, accum0);
		}
		if (Taps >= 8) {
			x = mm256_alignr_epi8_ps(x8, x4, 12);
			x = _mm256_permutevar8x32_ps(x, mask);
			coeffs = _mm256_load_ps(data + 7 * 8);
			accum1 = _mm256_fmadd_ps(coeffs, x, accum1);
		}

		accum0 = _mm256_add_ps(accum0, accum1);
		Traits::store8(dst + j, accum0);
	}
#undef mm256_alignr_ps
	for (unsigned j = fallback_idx; j < right; j += 8) {
		unsigned left = permute_left[j / 8];
		const float *data = filter_data + static_cast<size_t>(j) * Taps;

		__m256 accum0 = _mm256_setzero_ps();
		__m256 accum1 = _mm256_setzero_ps();
		__m256 x, coeffs;

		for (unsigned k = 0; k < Taps; ++k) {
			alignas(32) typename Traits::pixel_type tmp[8];

			for (unsigned kk = 0; kk < 8; ++kk) {
				tmp[kk] = src[left + permute_mask[j + kk] + k];
			}

			x = Traits::load8(tmp);
			coeffs = _mm256_load_ps(data + k * 8);

			if (k % 2)
				accum1 = _mm256_fmadd_ps(coeffs, x, accum1);
			else
				accum0 = _mm256_fmadd_ps(coeffs, x, accum0);
		}
		accum0 = _mm256_add_ps(accum0, accum1);
		Traits::store8(dst + j, accum0);
	}
}

template <class Traits>
constexpr auto resize_line_h_perm_fp_avx2_jt = make_array(
	resize_line_h_perm_fp_avx2<Traits, 1>,
	resize_line_h_perm_fp_avx2<Traits, 2>,
	resize_line_h_perm_fp_avx2<Traits, 3>,
	resize_line_h_perm_fp_avx2<Traits, 4>,
	resize_line_h_perm_fp_avx2<Traits, 5>,
	resize_line_h_perm_fp_avx2<Traits, 6>,
	resize_line_h_perm_fp_avx2<Traits, 7>,
	resize_line_h_perm_fp_avx2<Traits, 8>);


constexpr unsigned V_ACCUM_NONE = 0;
constexpr unsigned V_ACCUM_INITIAL = 1;
constexpr unsigned V_ACCUM_UPDATE = 2;
constexpr unsigned V_ACCUM_FINAL = 3;

template <unsigned Taps, unsigned AccumMode>
inline FORCE_INLINE __m256i resize_line_v_u16_avx2_xiter(unsigned j, unsigned accum_base,
                                                         const uint16_t *src_p0, const uint16_t *src_p1, const uint16_t *src_p2, const uint16_t *src_p3,
                                                         const uint16_t *src_p4, const uint16_t *src_p5, const uint16_t *src_p6, const uint16_t *src_p7,
                                                         uint32_t * RESTRICT accum_p, const __m256i &c01, const __m256i &c23, const __m256i &c45, const __m256i &c67, uint16_t limit)
{
	static_assert(Taps >= 2 && Taps <= 8, "must have between 2-8 taps");
	static_assert(Taps % 2 == 0, "tap count must be even");

	const __m256i i16_min = _mm256_set1_epi16(INT16_MIN);
	const __m256i lim = _mm256_set1_epi16(limit + INT16_MIN);

	__m256i accum_lo = _mm256_setzero_si256();
	__m256i accum_hi = _mm256_setzero_si256();
	__m256i x0, x1, xl, xh;

	if (Taps >= 2) {
		x0 = _mm256_load_si256((const __m256i *)(src_p0 + j));
		x1 = _mm256_load_si256((const __m256i *)(src_p1 + j));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c01, xl);
		xh = _mm256_madd_epi16(c01, xh);

		if (AccumMode == V_ACCUM_UPDATE || AccumMode == V_ACCUM_FINAL) {
			accum_lo = _mm256_add_epi32(_mm256_load_si256((const __m256i *)(accum_p + j - accum_base + 0)), xl);
			accum_hi = _mm256_add_epi32(_mm256_load_si256((const __m256i *)(accum_p + j - accum_base + 8)), xh);
		} else {
			accum_lo = xl;
			accum_hi = xh;
		}
	}
	if (Taps >= 4) {
		x0 = _mm256_load_si256((const __m256i *)(src_p2 + j));
		x1 = _mm256_load_si256((const __m256i *)(src_p3 + j));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c23, xl);
		xh = _mm256_madd_epi16(c23, xh);

		accum_lo = _mm256_add_epi32(accum_lo, xl);
		accum_hi = _mm256_add_epi32(accum_hi, xh);
	}
	if (Taps >= 6) {
		x0 = _mm256_load_si256((const __m256i *)(src_p4 + j));
		x1 = _mm256_load_si256((const __m256i *)(src_p5 + j));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c45, xl);
		xh = _mm256_madd_epi16(c45, xh);

		accum_lo = _mm256_add_epi32(accum_lo, xl);
		accum_hi = _mm256_add_epi32(accum_hi, xh);
	}
	if (Taps >= 8) {
		x0 = _mm256_load_si256((const __m256i *)(src_p6 + j));
		x1 = _mm256_load_si256((const __m256i *)(src_p7 + j));
		x0 = _mm256_add_epi16(x0, i16_min);
		x1 = _mm256_add_epi16(x1, i16_min);

		xl = _mm256_unpacklo_epi16(x0, x1);
		xh = _mm256_unpackhi_epi16(x0, x1);
		xl = _mm256_madd_epi16(c67, xl);
		xh = _mm256_madd_epi16(c67, xh);

		accum_lo = _mm256_add_epi32(accum_lo, xl);
		accum_hi = _mm256_add_epi32(accum_hi, xh);
	}

	if (AccumMode == V_ACCUM_INITIAL || AccumMode == V_ACCUM_UPDATE) {
		_mm256_store_si256((__m256i *)(accum_p + j - accum_base + 0), accum_lo);
		_mm256_store_si256((__m256i *)(accum_p + j - accum_base + 8), accum_hi);
		return _mm256_setzero_si256();
	} else {
		accum_lo = export_i30_u16(accum_lo, accum_hi);
		accum_lo = _mm256_min_epi16(accum_lo, lim);
		accum_lo = _mm256_sub_epi16(accum_lo, i16_min);

		return accum_lo;
	}
}

template <unsigned Taps, unsigned AccumMode>
void resize_line_v_u16_avx2(const int16_t * RESTRICT filter_data, const uint16_t * const * RESTRICT src, uint16_t * RESTRICT dst, uint32_t * RESTRICT accum, unsigned left, unsigned right, uint16_t limit)
{
	const uint16_t *src_p0 = src[0];
	const uint16_t *src_p1 = src[1];
	const uint16_t *src_p2 = src[2];
	const uint16_t *src_p3 = src[3];
	const uint16_t *src_p4 = src[4];
	const uint16_t *src_p5 = src[5];
	const uint16_t *src_p6 = src[6];
	const uint16_t *src_p7 = src[7];

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);
	unsigned accum_base = floor_n(left, 16);

	const __m256i c01 = _mm256_unpacklo_epi16(_mm256_set1_epi16(filter_data[0]), _mm256_set1_epi16(filter_data[1]));
	const __m256i c23 = _mm256_unpacklo_epi16(_mm256_set1_epi16(filter_data[2]), _mm256_set1_epi16(filter_data[3]));
	const __m256i c45 = _mm256_unpacklo_epi16(_mm256_set1_epi16(filter_data[4]), _mm256_set1_epi16(filter_data[5]));
	const __m256i c67 = _mm256_unpacklo_epi16(_mm256_set1_epi16(filter_data[6]), _mm256_set1_epi16(filter_data[7]));

	__m256i out;

#define XITER resize_line_v_u16_avx2_xiter<Taps, AccumMode>
#define XARGS accum_base, src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, accum, c01, c23, c45, c67, limit
	if (left != vec_left) {
		out = XITER(vec_left - 16, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			mm256_store_idxhi_epi16((__m256i *)(dst + vec_left - 16), out, left % 16);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		out = XITER(j, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			_mm256_store_si256((__m256i *)(dst + j), out);
	}

	if (right != vec_right) {
		out = XITER(vec_right, XARGS);

		if (AccumMode == V_ACCUM_NONE || AccumMode == V_ACCUM_FINAL)
			mm256_store_idxlo_epi16((__m256i *)(dst + vec_right), out, right % 16);
	}
#undef XITER
#undef XARGS
}

constexpr auto resize_line_v_u16_avx2_jt_small = make_array(
	resize_line_v_u16_avx2<2, V_ACCUM_NONE>,
	resize_line_v_u16_avx2<2, V_ACCUM_NONE>,
	resize_line_v_u16_avx2<4, V_ACCUM_NONE>,
	resize_line_v_u16_avx2<4, V_ACCUM_NONE>,
	resize_line_v_u16_avx2<6, V_ACCUM_NONE>,
	resize_line_v_u16_avx2<6, V_ACCUM_NONE>,
	resize_line_v_u16_avx2<8, V_ACCUM_NONE>,
	resize_line_v_u16_avx2<8, V_ACCUM_NONE>);

constexpr auto resize_line_v_u16_avx2_initial = resize_line_v_u16_avx2<8, V_ACCUM_INITIAL>;
constexpr auto resize_line_v_u16_avx2_update = resize_line_v_u16_avx2<8, V_ACCUM_UPDATE>;

constexpr auto resize_line_v_u16_avx2_jt_final = make_array(
	resize_line_v_u16_avx2<2, V_ACCUM_FINAL>,
	resize_line_v_u16_avx2<2, V_ACCUM_FINAL>,
	resize_line_v_u16_avx2<4, V_ACCUM_FINAL>,
	resize_line_v_u16_avx2<4, V_ACCUM_FINAL>,
	resize_line_v_u16_avx2<6, V_ACCUM_FINAL>,
	resize_line_v_u16_avx2<6, V_ACCUM_FINAL>,
	resize_line_v_u16_avx2<8, V_ACCUM_FINAL>,
	resize_line_v_u16_avx2<8, V_ACCUM_FINAL>);


template <class Traits, unsigned Taps, bool Continue, class T = typename Traits::pixel_type>
inline FORCE_INLINE __m256 resize_line_v_fp_avx2_xiter(unsigned j,
                                                       const T *src_p0, const T *src_p1, const T *src_p2, const T *src_p3,
                                                       const T *src_p4, const T *src_p5, const T *src_p6, const T *src_p7, T * RESTRICT accum_p,
                                                       const __m256 &c0, const __m256 &c1, const __m256 &c2, const __m256 &c3,
                                                       const __m256 &c4, const __m256 &c5, const __m256 &c6, const __m256 &c7)
{
	static_assert(Taps >= 1 && Taps <= 8, "must have between 1-8 taps");

	typedef typename Traits::pixel_type pixel_type;
	static_assert(std::is_same<pixel_type, T>::value, "must not specify T");

	__m256 accum0 = _mm256_setzero_ps();
	__m256 accum1 = _mm256_setzero_ps();
	__m256 x;

	if (Taps >= 1) {
		x = Traits::load8(src_p0 + j);
		accum0 = Continue ? _mm256_fmadd_ps(c0, x, Traits::load8(accum_p + j)) : _mm256_mul_ps(c0, x);
	}
	if (Taps >= 2) {
		x = Traits::load8(src_p1 + j);
		accum1 = _mm256_mul_ps(c1, x);
	}
	if (Taps >= 3) {
		x = Traits::load8(src_p2 + j);
		accum0 = _mm256_fmadd_ps(c2, x, accum0);
	}
	if (Taps >= 4) {
		x = Traits::load8(src_p3 + j);
		accum1 = _mm256_fmadd_ps(c3, x, accum1);
	}
	if (Taps >= 5) {
		x = Traits::load8(src_p4 + j);
		accum0 = _mm256_fmadd_ps(c4, x, accum0);
	}
	if (Taps >= 6) {
		x = Traits::load8(src_p5 + j);
		accum1 = _mm256_fmadd_ps(c5, x, accum1);
	}
	if (Taps >= 7) {
		x = Traits::load8(src_p6 + j);
		accum0 = _mm256_fmadd_ps(c6, x, accum0);
	}
	if (Taps >= 8) {
		x = Traits::load8(src_p7 + j);
		accum1 = _mm256_fmadd_ps(c7, x, accum1);
	}

	accum0 = (Taps >= 2) ? _mm256_add_ps(accum0, accum1) : accum0;
	return accum0;
}

template <class Traits, unsigned Taps, bool Continue>
void resize_line_v_fp_avx2(const float * RESTRICT filter_data, const typename Traits::pixel_type * const * RESTRICT src, typename Traits::pixel_type * RESTRICT dst, unsigned left, unsigned right)
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

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const __m256 c0 = _mm256_broadcast_ss(filter_data + 0);
	const __m256 c1 = _mm256_broadcast_ss(filter_data + 1);
	const __m256 c2 = _mm256_broadcast_ss(filter_data + 2);
	const __m256 c3 = _mm256_broadcast_ss(filter_data + 3);
	const __m256 c4 = _mm256_broadcast_ss(filter_data + 4);
	const __m256 c5 = _mm256_broadcast_ss(filter_data + 5);
	const __m256 c6 = _mm256_broadcast_ss(filter_data + 6);
	const __m256 c7 = _mm256_broadcast_ss(filter_data + 7);

	__m256 accum;

#define XITER resize_line_v_fp_avx2_xiter<Traits, Taps, Continue>
#define XARGS src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, dst, c0, c1, c2, c3, c4, c5, c6, c7
	if (left != vec_left) {
		accum = XITER(vec_left - 8, XARGS);
		Traits::store_idxhi(dst + vec_left - 8, accum, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		accum = XITER(j, XARGS);
		Traits::store8(dst + j, accum);
	}

	if (right != vec_right) {
		accum = XITER(vec_right, XARGS);
		Traits::store_idxlo(dst + vec_right, accum, right % 8);
	}
#undef XITER
#undef XARGS
}

template <class Traits>
constexpr auto resize_line_v_fp_avx2_jt_init = make_array(
	resize_line_v_fp_avx2<Traits, 1, false>,
	resize_line_v_fp_avx2<Traits, 2, false>,
	resize_line_v_fp_avx2<Traits, 3, false>,
	resize_line_v_fp_avx2<Traits, 4, false>,
	resize_line_v_fp_avx2<Traits, 5, false>,
	resize_line_v_fp_avx2<Traits, 6, false>,
	resize_line_v_fp_avx2<Traits, 7, false>,
	resize_line_v_fp_avx2<Traits, 8, false>);

template <class Traits>
constexpr auto resize_line_v_fp_avx2_jt_cont = make_array(
	resize_line_v_fp_avx2<Traits, 1, true>,
	resize_line_v_fp_avx2<Traits, 2, true>,
	resize_line_v_fp_avx2<Traits, 3, true>,
	resize_line_v_fp_avx2<Traits, 4, true>,
	resize_line_v_fp_avx2<Traits, 5, true>,
	resize_line_v_fp_avx2<Traits, 6, true>,
	resize_line_v_fp_avx2<Traits, 7, true>,
	resize_line_v_fp_avx2<Traits, 8, true>);


class ResizeImplH_U16_AVX2 final : public ResizeImplH {
	decltype(resize_line8_h_u16_avx2_jt_small)::value_type m_func;
	uint16_t m_pixel_max;
public:
	ResizeImplH_U16_AVX2(const FilterContext &filter, unsigned height, unsigned depth) try :
		ResizeImplH(filter, height, PixelType::WORD),
		m_func{},
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{
		m_desc.step = 16;
		m_desc.scratchpad_size = (ceil_n(checked_size_t{ filter.input_width }, 16) * sizeof(uint16_t) * 16).get();

		if (filter.filter_width > 8)
			m_func = resize_line8_h_u16_avx2_jt_large[filter.filter_width % 8];
		else
			m_func = resize_line8_h_u16_avx2_jt_small[filter.filter_width - 1];
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		auto range = get_col_deps(left, right);

		const uint16_t *src_ptr[16] = { 0 };
		uint16_t *dst_ptr[16] = { 0 };
		uint16_t *transpose_buf = static_cast<uint16_t *>(tmp);
		unsigned height = m_desc.format.height;

		for (unsigned n = 0; n < 16; ++n) {
			src_ptr[n] = in->get_line<uint16_t>(std::min(i + n, height - 1));
		}

		transpose_line_16x16_epi16(transpose_buf, src_ptr, floor_n(range.first, 16), ceil_n(range.second, 16));

		for (unsigned n = 0; n < 16; ++n) {
			dst_ptr[n] = out->get_line<uint16_t>(std::min(i + n, height - 1));
		}

		m_func(m_filter.left.data(), m_filter.data_i16.data(), m_filter.stride_i16, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 16), left, right, m_pixel_max);
	}
};


template <class Traits>
class ResizeImplH_FP_AVX2 final : public ResizeImplH {
	typedef typename Traits::pixel_type pixel_type;
	typedef typename decltype(resize_line8_h_fp_avx2_jt_small<Traits>)::value_type func_type;

	func_type m_func;
public:
	ResizeImplH_FP_AVX2(const FilterContext &filter, unsigned height) try :
		ResizeImplH(filter, height, Traits::type_constant),
		m_func{}
	{
		m_desc.step = 8;
		m_desc.scratchpad_size = (ceil_n(checked_size_t{ filter.input_width }, 8) * sizeof(pixel_type) * 8).get();

		if (filter.filter_width <= 8)
			m_func = resize_line8_h_fp_avx2_jt_small<Traits>[filter.filter_width - 1];
		else
			m_func = resize_line8_h_fp_avx2_jt_large<Traits>[filter.filter_width % 4];
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		auto range = get_col_deps(left, right);

		const pixel_type *src_ptr[8] = { 0 };
		pixel_type *dst_ptr[8] = { 0 };
		pixel_type *transpose_buf = static_cast<pixel_type *>(tmp);
		unsigned height = m_desc.format.height;

		src_ptr[0] = in->get_line<pixel_type>(std::min(i + 0, height - 1));
		src_ptr[1] = in->get_line<pixel_type>(std::min(i + 1, height - 1));
		src_ptr[2] = in->get_line<pixel_type>(std::min(i + 2, height - 1));
		src_ptr[3] = in->get_line<pixel_type>(std::min(i + 3, height - 1));
		src_ptr[4] = in->get_line<pixel_type>(std::min(i + 4, height - 1));
		src_ptr[5] = in->get_line<pixel_type>(std::min(i + 5, height - 1));
		src_ptr[6] = in->get_line<pixel_type>(std::min(i + 6, height - 1));
		src_ptr[7] = in->get_line<pixel_type>(std::min(i + 7, height - 1));

		transpose_line_8x8<Traits>(transpose_buf, src_ptr, floor_n(range.first, 8), ceil_n(range.second, 8));

		dst_ptr[0] = out->get_line<pixel_type>(std::min(i + 0, height - 1));
		dst_ptr[1] = out->get_line<pixel_type>(std::min(i + 1, height - 1));
		dst_ptr[2] = out->get_line<pixel_type>(std::min(i + 2, height - 1));
		dst_ptr[3] = out->get_line<pixel_type>(std::min(i + 3, height - 1));
		dst_ptr[4] = out->get_line<pixel_type>(std::min(i + 4, height - 1));
		dst_ptr[5] = out->get_line<pixel_type>(std::min(i + 5, height - 1));
		dst_ptr[6] = out->get_line<pixel_type>(std::min(i + 6, height - 1));
		dst_ptr[7] = out->get_line<pixel_type>(std::min(i + 7, height - 1));

		m_func(m_filter.left.data(), m_filter.data.data(), m_filter.stride, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 8), left, right);
	}
};


class ResizeImplH_Permute_U16_AVX2 final : public graphengine::Filter {
	typedef decltype(resize_line_h_perm_u16_avx2_jt)::value_type func_type;

	struct PermuteContext {
		AlignedVector<unsigned> left;
		AlignedVector<unsigned> permute;
		AlignedVector<int16_t> data;
		unsigned filter_rows;
		unsigned filter_width;
		unsigned input_width;
	};

	graphengine::FilterDescriptor m_desc;
	PermuteContext m_context;
	uint16_t m_pixel_max;
	func_type m_func;

	ResizeImplH_Permute_U16_AVX2(PermuteContext context, unsigned height, unsigned depth) :
		m_desc{},
		m_context(std::move(context)),
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) },
		m_func{ resize_line_h_perm_u16_avx2_jt[(m_context.filter_width - 1) / 2] }
	{
		m_desc.format = { context.filter_rows, height, pixel_size(PixelType::WORD) };
		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.step = 1;
		m_desc.alignment_mask = 7;
		m_desc.flags.entire_row = !std::is_sorted(m_context.left.begin(), m_context.left.end());
	}
public:
	static std::unique_ptr<graphengine::Filter> create(const FilterContext &filter, unsigned height, unsigned depth)
	{
		// Transpose is faster for large filters.
		if (filter.filter_width > 8)
			return nullptr;

		PermuteContext context{};

		unsigned filter_width = ceil_n(filter.filter_width + 2, 2);

		context.left.resize(ceil_n(filter.filter_rows, 8) / 8);
		context.permute.resize(ceil_n(filter.filter_rows, 8));
		context.data.resize(ceil_n(filter.filter_rows, 8) * filter_width);
		context.filter_rows = filter.filter_rows;
		context.filter_width = filter_width;
		context.input_width = filter.input_width;

		for (unsigned i = 0; i < filter.filter_rows; i += 8) {
			unsigned left_min = UINT_MAX;
			unsigned left_max = 0U;

			for (unsigned ii = i; ii < std::min(i + 8, context.filter_rows); ++ii) {
				left_min = std::min(left_min, filter.left[ii]);
				left_max = std::max(left_max, filter.left[ii]);
			}
			if (floor_n(left_max - left_min, 2) >= 16)
				return nullptr;

			for (unsigned ii = i; ii < std::min(i + 8, context.filter_rows); ++ii) {
				context.permute[ii] = floor_n(filter.left[ii] - left_min, 2) / 2;
			}
			context.left[i / 8] = left_min;

			int16_t *data = context.data.data() + i * context.filter_width;
			for (unsigned k = 0; k < filter.filter_width; k += 2) {
				for (unsigned ii = i; ii < std::min(i + 8, context.filter_rows); ++ii) {
					unsigned offset = (filter.left[ii] - context.left[i / 8]) % 2;

					if (offset) {
						data[static_cast<size_t>(k / 2) * 16 + (ii - i) * 2 + 1] = filter.data_i16[ii * static_cast<ptrdiff_t>(filter.stride_i16) + k + 0];
						data[static_cast<size_t>(k / 2 + 1) * 16 + (ii - i) * 2] = filter.data_i16[ii * static_cast<ptrdiff_t>(filter.stride_i16) + k + 1];
					} else {
						data[static_cast<size_t>(k / 2) * 16 + (ii - i) * 2 + 0] = filter.data_i16[ii * static_cast<ptrdiff_t>(filter.stride_i16) + k + 0];
						data[static_cast<size_t>(k / 2) * 16 + (ii - i) * 2 + 1] = filter.data_i16[ii * static_cast<ptrdiff_t>(filter.stride_i16) + k + 1];
					}
				}
			}
		}

		std::unique_ptr<graphengine::Filter> ret{ new ResizeImplH_Permute_U16_AVX2(std::move(context), height, depth) };
		return ret;
	}

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	pair_unsigned get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override
	{
		if (m_desc.flags.entire_row)
			return{ 0, m_context.input_width };

		unsigned input_width = m_context.input_width;
		unsigned right_base = m_context.left[(right + 7) / 8 - 1];
		unsigned iter_width = m_context.filter_width + 16;
		return{ m_context.left[left / 8],  right_base + std::min(input_width - right_base, iter_width) };
	}

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		m_func(m_context.left.data(), m_context.permute.data(), m_context.data.data(), m_context.input_width, in->get_line<uint16_t>(i), out->get_line<uint16_t>(i), left, right, m_pixel_max);
	}
};


template <class Traits>
class ResizeImplH_Permute_FP_AVX2 final : public graphengine::Filter {
	typedef typename Traits::pixel_type pixel_type;
	typedef typename decltype(resize_line_h_perm_fp_avx2_jt<Traits>)::value_type func_type;

	struct PermuteContext {
		AlignedVector<unsigned> left;
		AlignedVector<unsigned> permute;
		AlignedVector<float> data;
		unsigned filter_rows;
		unsigned filter_width;
		unsigned input_width;
	};

	graphengine::FilterDescriptor m_desc;
	PermuteContext m_context;
	func_type m_func;

	ResizeImplH_Permute_FP_AVX2(PermuteContext context, unsigned height) :
		m_desc{},
		m_context(std::move(context)),
		m_func{ resize_line_h_perm_fp_avx2_jt<Traits>[m_context.filter_width - 1] }
	{
		m_desc.format = { context.filter_rows, height, pixel_size(Traits::type_constant) };
		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.step = 1;
		m_desc.alignment_mask = 7;
		m_desc.flags.entire_row = !std::is_sorted(m_context.left.begin(), m_context.left.end());
	}
public:
	static std::unique_ptr<graphengine::Filter> create(const FilterContext &filter, unsigned height)
	{
		// Transpose is faster for large filters.
		if (filter.filter_width > 8)
			return nullptr;

		PermuteContext context{};

		context.left.resize(ceil_n(filter.filter_rows, 8) / 8);
		context.permute.resize(ceil_n(filter.filter_rows, 8));
		context.data.resize(ceil_n(filter.filter_rows, 8) * filter.filter_width);
		context.filter_rows = filter.filter_rows;
		context.filter_width = filter.filter_width;
		context.input_width = filter.input_width;

		for (unsigned i = 0; i < filter.filter_rows; i += 8) {
			unsigned left_min = UINT_MAX;
			unsigned left_max = 0U;

			for (unsigned ii = i; ii < std::min(i + 8, context.filter_rows); ++ii) {
				left_min = std::min(left_min, filter.left[ii]);
				left_max = std::max(left_max, filter.left[ii]);
			}
			if (left_max - left_min >= 8)
				return nullptr;

			for (unsigned ii = i; ii < std::min(i + 8, context.filter_rows); ++ii) {
				context.permute[ii] = filter.left[ii] - left_min;
			}
			context.left[i / 8] = left_min;

			float *data = context.data.data() + i * context.filter_width;
			for (unsigned k = 0; k < context.filter_width; ++k) {
				for (unsigned ii = i; ii < std::min(i + 8, context.filter_rows); ++ii) {
					data[static_cast<size_t>(k) * 8 + (ii - i)] = filter.data[ii * static_cast<ptrdiff_t>(filter.stride) + k];
				}
			}
		}

		std::unique_ptr<graphengine::Filter> ret{ new ResizeImplH_Permute_FP_AVX2(std::move(context), height) };
		return ret;
	}

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	pair_unsigned get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override
	{
		if (m_desc.flags.entire_row)
			return{ 0, m_context.input_width };

		unsigned input_width = m_context.input_width;
		unsigned right_base = m_context.left[(right + 7) / 8 - 1];
		unsigned iter_width = m_context.filter_width + 8;
		return{ m_context.left[left / 8],  right_base + std::min(input_width - right_base, iter_width) };
	}

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		m_func(m_context.left.data(), m_context.permute.data(), m_context.data.data(), m_context.input_width, in->get_line<pixel_type>(i), out->get_line<pixel_type>(i), left, right);
	}
};


class ResizeImplV_U16_AVX2 final : public ResizeImplV {
	uint16_t m_pixel_max;
public:
	ResizeImplV_U16_AVX2(const FilterContext &filter, unsigned width, unsigned depth) try :
		ResizeImplV(filter, width, PixelType::WORD),
		m_pixel_max{ static_cast<uint16_t>((1UL << depth) - 1) }
	{
		if (m_filter.filter_width > 8)
			m_desc.scratchpad_size = (ceil_n(checked_size_t{ width }, 16) * sizeof(uint32_t)).get();
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

		auto calculate_line_address = [&](unsigned i)
		{
			for (unsigned n = 0; n < 8; ++n) {
				src_lines[n] = in->get_line<uint16_t>(std::min(i + n, src_height - 1));
			}
		};

		if (filter_width <= 8) {
			calculate_line_address(top);
			resize_line_v_u16_avx2_jt_small[filter_width - 1](filter_data, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
		} else {
			unsigned k_end = ceil_n(filter_width, 8) - 8;

			calculate_line_address(top);
			resize_line_v_u16_avx2_initial(filter_data + 0, src_lines, dst_line, accum_buf, left, right, m_pixel_max);

			for (unsigned k = 8; k < k_end; k += 8) {
				calculate_line_address(top + k);
				resize_line_v_u16_avx2_update(filter_data + k, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
			}

			calculate_line_address(top + k_end);
			resize_line_v_u16_avx2_jt_final[filter_width - k_end - 1](filter_data + k_end, src_lines, dst_line, accum_buf, left, right, m_pixel_max);
		}
	}
};


template <class Traits>
class ResizeImplV_FP_AVX2 final : public ResizeImplV {
	typedef typename Traits::pixel_type pixel_type;
public:
	ResizeImplV_FP_AVX2(const FilterContext &filter, unsigned width) :
		ResizeImplV(filter, width, Traits::type_constant)
	{}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		const float *filter_data = m_filter.data.data() + i * m_filter.stride;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const pixel_type *src_lines[8] = { 0 };
		pixel_type *dst_line = out->get_line<pixel_type>(i);

		{
			unsigned taps_remain = std::min(filter_width - 0, 8U);
			unsigned top = m_filter.left[i] + 0;

			src_lines[0] = in->get_line<pixel_type>(std::min(top + 0, src_height - 1));
			src_lines[1] = in->get_line<pixel_type>(std::min(top + 1, src_height - 1));
			src_lines[2] = in->get_line<pixel_type>(std::min(top + 2, src_height - 1));
			src_lines[3] = in->get_line<pixel_type>(std::min(top + 3, src_height - 1));
			src_lines[4] = in->get_line<pixel_type>(std::min(top + 4, src_height - 1));
			src_lines[5] = in->get_line<pixel_type>(std::min(top + 5, src_height - 1));
			src_lines[6] = in->get_line<pixel_type>(std::min(top + 6, src_height - 1));
			src_lines[7] = in->get_line<pixel_type>(std::min(top + 7, src_height - 1));

			resize_line_v_fp_avx2_jt_init<Traits>[taps_remain - 1](filter_data + 0, src_lines, dst_line, left, right);
		}

		for (unsigned k = 8; k < filter_width; k += 8) {
			unsigned taps_remain = std::min(filter_width - k, 8U);
			unsigned top = m_filter.left[i] + k;

			src_lines[0] = in->get_line<pixel_type>(std::min(top + 0, src_height - 1));
			src_lines[1] = in->get_line<pixel_type>(std::min(top + 1, src_height - 1));
			src_lines[2] = in->get_line<pixel_type>(std::min(top + 2, src_height - 1));
			src_lines[3] = in->get_line<pixel_type>(std::min(top + 3, src_height - 1));
			src_lines[4] = in->get_line<pixel_type>(std::min(top + 4, src_height - 1));
			src_lines[5] = in->get_line<pixel_type>(std::min(top + 5, src_height - 1));
			src_lines[6] = in->get_line<pixel_type>(std::min(top + 6, src_height - 1));
			src_lines[7] = in->get_line<pixel_type>(std::min(top + 7, src_height - 1));

			resize_line_v_fp_avx2_jt_cont<Traits>[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
		}
	}
};

} // namespace


std::unique_ptr<graphengine::Filter> create_resize_impl_h_avx2(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

#ifndef ZIMG_RESIZE_NO_PERMUTE
	if (cpu_has_slow_permute(query_x86_capabilities()))
		ret = nullptr;
	else if (type == PixelType::WORD)
		ret = ResizeImplH_Permute_U16_AVX2::create(context, height, depth);
	else if (type == PixelType::HALF)
		ret = ResizeImplH_Permute_FP_AVX2<f16_traits>::create(context, height);
	else if (type == PixelType::FLOAT)
		ret = ResizeImplH_Permute_FP_AVX2<f32_traits>::create(context, height);
#endif

	if (!ret) {
		if (type == PixelType::WORD)
			ret = std::make_unique<ResizeImplH_U16_AVX2>(context, height, depth);
		else if (type == PixelType::HALF)
			ret = std::make_unique<ResizeImplH_FP_AVX2<f16_traits>>(context, height);
		else if (type == PixelType::FLOAT)
			ret = std::make_unique<ResizeImplH_FP_AVX2<f32_traits>>(context, height);
	}

	return ret;
}

std::unique_ptr<graphengine::Filter> create_resize_impl_v_avx2(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

	if (type == PixelType::WORD)
		ret = std::make_unique<ResizeImplV_U16_AVX2>(context, width, depth);
	else if (type == PixelType::HALF)
		ret = std::make_unique<ResizeImplV_FP_AVX2<f16_traits>>(context, width);
	else if (type == PixelType::FLOAT)
		ret = std::make_unique<ResizeImplV_FP_AVX2<f32_traits>>(context, width);

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
