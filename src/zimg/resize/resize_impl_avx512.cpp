#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
  #undef ZIMG_X86_AVX512
#endif

#ifdef ZIMG_X86_AVX512

#include <algorithm>
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

#define HAVE_CPU_SSE
#define HAVE_CPU_SSE2
#define HAVE_CPU_AVX
#define HAVE_CPU_AVX2
  #include "common/x86util.h"
#undef HAVE_CPU_SSE
#undef HAVE_CPU_SSE2
#undef HAVE_CPU_AVX
#undef HAVE_CPU_AVX2

#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {
namespace resize {

namespace {

static inline FORCE_INLINE void mm512_transpose16_ps(__m512 &row0, __m512 &row1, __m512 &row2, __m512 &row3,
                                                     __m512 &row4, __m512 &row5, __m512 &row6, __m512 &row7,
                                                     __m512 &row8, __m512 &row9, __m512 &row10, __m512 &row11,
                                                     __m512 &row12, __m512 &row13, __m512 &row14, __m512 &row15)
{
	__m512 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
	__m512 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8, tt9, tt10, tt11, tt12, tt13, tt14, tt15;

	t0 = _mm512_unpacklo_ps(row0, row1);
	t1 = _mm512_unpackhi_ps(row0, row1);
	t2 = _mm512_unpacklo_ps(row2, row3);
	t3 = _mm512_unpackhi_ps(row2, row3);
	t4 = _mm512_unpacklo_ps(row4, row5);
	t5 = _mm512_unpackhi_ps(row4, row5);
	t6 = _mm512_unpacklo_ps(row6, row7);
	t7 = _mm512_unpackhi_ps(row6, row7);
	t8 = _mm512_unpacklo_ps(row8, row9);
	t9 = _mm512_unpackhi_ps(row8, row9);
	t10 = _mm512_unpacklo_ps(row10, row11);
	t11 = _mm512_unpackhi_ps(row10, row11);
	t12 = _mm512_unpacklo_ps(row12, row13);
	t13 = _mm512_unpackhi_ps(row12, row13);
	t14 = _mm512_unpacklo_ps(row14, row15);
	t15 = _mm512_unpackhi_ps(row14, row15);

	tt0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t2)));
	tt1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t2)));
	tt2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t1), _mm512_castps_pd(t3)));
	tt3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t1), _mm512_castps_pd(t3)));
	tt4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t4), _mm512_castps_pd(t6)));
	tt5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t4), _mm512_castps_pd(t6)));
	tt6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t5), _mm512_castps_pd(t7)));
	tt7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t5), _mm512_castps_pd(t7)));
	tt8 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t8), _mm512_castps_pd(t10)));
	tt9 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t8), _mm512_castps_pd(t10)));
	tt10 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t9), _mm512_castps_pd(t11)));
	tt11 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t9), _mm512_castps_pd(t11)));
	tt12 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t12), _mm512_castps_pd(t14)));
	tt13 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t12), _mm512_castps_pd(t14)));
	tt14 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t13), _mm512_castps_pd(t15)));
	tt15 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t13), _mm512_castps_pd(t15)));

	t0 = _mm512_shuffle_f32x4(tt0, tt4, 0x88);
	t1 = _mm512_shuffle_f32x4(tt1, tt5, 0x88);
	t2 = _mm512_shuffle_f32x4(tt2, tt6, 0x88);
	t3 = _mm512_shuffle_f32x4(tt3, tt7, 0x88);
	t4 = _mm512_shuffle_f32x4(tt0, tt4, 0xdd);
	t5 = _mm512_shuffle_f32x4(tt1, tt5, 0xdd);
	t6 = _mm512_shuffle_f32x4(tt2, tt6, 0xdd);
	t7 = _mm512_shuffle_f32x4(tt3, tt7, 0xdd);
	t8 = _mm512_shuffle_f32x4(tt8, tt12, 0x88);
	t9 = _mm512_shuffle_f32x4(tt9, tt13, 0x88);
	t10 = _mm512_shuffle_f32x4(tt10, tt14, 0x88);
	t11 = _mm512_shuffle_f32x4(tt11, tt15, 0x88);
	t12 = _mm512_shuffle_f32x4(tt8, tt12, 0xdd);
	t13 = _mm512_shuffle_f32x4(tt9, tt13, 0xdd);
	t14 = _mm512_shuffle_f32x4(tt10, tt14, 0xdd);
	t15 = _mm512_shuffle_f32x4(tt11, tt15, 0xdd);

	row0 = _mm512_shuffle_f32x4(t0, t8, 0x88);
	row1 = _mm512_shuffle_f32x4(t1, t9, 0x88);
	row2 = _mm512_shuffle_f32x4(t2, t10, 0x88);
	row3 = _mm512_shuffle_f32x4(t3, t11, 0x88);
	row4 = _mm512_shuffle_f32x4(t4, t12, 0x88);
	row5 = _mm512_shuffle_f32x4(t5, t13, 0x88);
	row6 = _mm512_shuffle_f32x4(t6, t14, 0x88);
	row7 = _mm512_shuffle_f32x4(t7, t15, 0x88);
	row8 = _mm512_shuffle_f32x4(t0, t8, 0xdd);
	row9 = _mm512_shuffle_f32x4(t1, t9, 0xdd);
	row10 = _mm512_shuffle_f32x4(t2, t10, 0xdd);
	row11 = _mm512_shuffle_f32x4(t3, t11, 0xdd);
	row12 = _mm512_shuffle_f32x4(t4, t12, 0xdd);
	row13 = _mm512_shuffle_f32x4(t5, t13, 0xdd);
	row14 = _mm512_shuffle_f32x4(t6, t14, 0xdd);
	row15 = _mm512_shuffle_f32x4(t7, t15, 0xdd);
}


struct f16_traits {
	typedef __m256i vec16_type;
	typedef uint16_t pixel_type;

	static constexpr PixelType type_constant = PixelType::HALF;

	static inline FORCE_INLINE vec16_type load16_raw(const pixel_type *ptr)
	{
		return _mm256_load_si256((const __m256i *)ptr);
	}

	static inline FORCE_INLINE void store16_raw(pixel_type *ptr, vec16_type x)
	{
		_mm256_store_si256((__m256i *)ptr, x);
	}

	static inline FORCE_INLINE __m512 load16(const pixel_type *ptr)
	{
		return _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)ptr));
	}

	static inline FORCE_INLINE void store16(pixel_type *ptr, __m512 x)
	{
		_mm256_store_si256((__m256i *)ptr, _mm512_cvtps_ph(x, 0));
	}

	static inline FORCE_INLINE void mask_store16(pixel_type *ptr, __mmask16 mask, __m512 x)
	{
		_mm256_mask_storeu_epi16((__m256i *)ptr, mask, _mm512_cvtps_ph(x, 0));
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
		return _mm512_load_ps(ptr);
	}

	static inline FORCE_INLINE void store16_raw(pixel_type *ptr, vec16_type x)
	{
		_mm512_store_ps(ptr, x);
	}

	static inline FORCE_INLINE __m512 load16(const pixel_type *ptr)
	{
		return _mm512_load_ps(ptr);
	}

	static inline FORCE_INLINE void store16(pixel_type *ptr, __m512 x)
	{
		_mm512_store_ps(ptr, x);
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
		mm_scatter_ps(dst0, dst1, dst2, dst3, _mm512_extractf32x4_ps(x, 0));
		mm_scatter_ps(dst4, dst5, dst6, dst7, _mm512_extractf32x4_ps(x, 1));
		mm_scatter_ps(dst8, dst9, dst10, dst11, _mm512_extractf32x4_ps(x, 2));
		mm_scatter_ps(dst12, dst13, dst14, dst15, _mm512_extractf32x4_ps(x, 3));
	}
};


template <class Traits, class T>
void transpose_line_16x16(T *dst, const T *src_p[16], unsigned left, unsigned right)
{
	typedef typename Traits::vec16_type vec16_type;

	const T *src_p0 = src_p[0];
	const T *src_p1 = src_p[1];
	const T *src_p2 = src_p[2];
	const T *src_p3 = src_p[3];
	const T *src_p4 = src_p[4];
	const T *src_p5 = src_p[5];
	const T *src_p6 = src_p[6];
	const T *src_p7 = src_p[7];
	const T *src_p8 = src_p[8];
	const T *src_p9 = src_p[9];
	const T *src_p10 = src_p[10];
	const T *src_p11 = src_p[11];
	const T *src_p12 = src_p[12];
	const T *src_p13 = src_p[13];
	const T *src_p14 = src_p[14];
	const T *src_p15 = src_p[15];

	for (unsigned j = left; j < right; j += 16) {
		vec16_type x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

		x0 = Traits::load16_raw(src_p0 + j);
		x1 = Traits::load16_raw(src_p1 + j);
		x2 = Traits::load16_raw(src_p2 + j);
		x3 = Traits::load16_raw(src_p3 + j);
		x4 = Traits::load16_raw(src_p4 + j);
		x5 = Traits::load16_raw(src_p5 + j);
		x6 = Traits::load16_raw(src_p6 + j);
		x7 = Traits::load16_raw(src_p7 + j);
		x8 = Traits::load16_raw(src_p8 + j);
		x9 = Traits::load16_raw(src_p9 + j);
		x10 = Traits::load16_raw(src_p10 + j);
		x11 = Traits::load16_raw(src_p11 + j);
		x12 = Traits::load16_raw(src_p12 + j);
		x13 = Traits::load16_raw(src_p13 + j);
		x14 = Traits::load16_raw(src_p14 + j);
		x15 = Traits::load16_raw(src_p15 + j);

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


template <class Traits, unsigned FWidth, unsigned Tail>
inline FORCE_INLINE __m512 resize_line16_h_fp_avx512_xiter(unsigned j,
                                                           const unsigned *filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                                           const typename Traits::pixel_type * RESTRICT src_ptr, unsigned src_base)
{
	typedef typename Traits::pixel_type pixel_type;

	const float *filter_coeffs = filter_data + j * filter_stride;
	const pixel_type *src_p = src_ptr + (filter_left[j] - src_base) * 16;

	__m512 accum0 = _mm512_setzero_ps();
	__m512 accum1 = _mm512_setzero_ps();
	__m512 x, c, coeffs;

	unsigned k_end = FWidth ? FWidth - Tail : floor_n(filter_width, 4);

	for (unsigned k = 0; k < k_end; k += 4) {
		coeffs = _mm512_broadcast_f32x4(_mm_load_ps(filter_coeffs + k));

		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x = Traits::load16(src_p + (k + 0) * 16);
		accum0 = _mm512_fmadd_ps(c, x, accum0);

		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x = Traits::load16(src_p + (k + 1) * 16);
		accum1 = _mm512_fmadd_ps(c, x, accum1);

		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x = Traits::load16(src_p + (k + 2) * 16);
		accum0 = _mm512_fmadd_ps(c, x, accum0);

		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x = Traits::load16(src_p + (k + 3) * 16);
		accum1 = _mm512_fmadd_ps(c, x, accum1);
	}

	if (Tail >= 1) {
		coeffs = _mm512_broadcast_f32x4(_mm_load_ps(filter_coeffs + k_end));

		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(0, 0, 0, 0));
		x = Traits::load16(src_p + (k_end + 0) * 16);
		accum0 = _mm512_fmadd_ps(c, x, accum0);
	}
	if (Tail >= 2) {
		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1, 1, 1, 1));
		x = Traits::load16(src_p + (k_end + 1) * 16);
		accum1 = _mm512_fmadd_ps(c, x, accum1);
	}
	if (Tail >= 3) {
		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(2, 2, 2, 2));
		x = Traits::load16(src_p + (k_end + 2) * 16);
		accum0 = _mm512_fmadd_ps(c, x, accum0);
	}
	if (Tail >= 4) {
		c = _mm512_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3, 3, 3, 3));
		x = Traits::load16(src_p + (k_end + 3) * 16);
		accum1 = _mm512_fmadd_ps(c, x, accum1);
	}

	if (!FWidth || FWidth >= 2)
		accum0 = _mm512_add_ps(accum0, accum1);

	return accum0;
}

template <class Traits, unsigned FWidth, unsigned Tail>
void resize_line16_h_fp_avx512(const unsigned *filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                               const typename Traits::pixel_type *src_ptr, typename Traits::pixel_type * const *dst_ptr, unsigned src_base, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

#define XITER resize_line16_h_fp_avx512_xiter<Traits, FWidth, Tail>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src_ptr, src_base
	for (unsigned j = left; j < vec_left; ++j) {
		__m512 x = XITER(j, XARGS);
		Traits::scatter16(dst_ptr[0] + j, dst_ptr[1] + j, dst_ptr[2] + j, dst_ptr[3] + j, dst_ptr[4] + j, dst_ptr[5] + j, dst_ptr[6] + j, dst_ptr[7] + j,
		                  dst_ptr[8] + j, dst_ptr[9] + j, dst_ptr[10] + j, dst_ptr[11] + j, dst_ptr[12] + j, dst_ptr[13] + j, dst_ptr[14] + j, dst_ptr[15] + j, x);
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

		Traits::store16(dst_ptr[0] + j, x0);
		Traits::store16(dst_ptr[1] + j, x1);
		Traits::store16(dst_ptr[2] + j, x2);
		Traits::store16(dst_ptr[3] + j, x3);
		Traits::store16(dst_ptr[4] + j, x4);
		Traits::store16(dst_ptr[5] + j, x5);
		Traits::store16(dst_ptr[6] + j, x6);
		Traits::store16(dst_ptr[7] + j, x7);
		Traits::store16(dst_ptr[8] + j, x8);
		Traits::store16(dst_ptr[9] + j, x9);
		Traits::store16(dst_ptr[10] + j, x10);
		Traits::store16(dst_ptr[11] + j, x11);
		Traits::store16(dst_ptr[12] + j, x12);
		Traits::store16(dst_ptr[13] + j, x13);
		Traits::store16(dst_ptr[14] + j, x14);
		Traits::store16(dst_ptr[15] + j, x15);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		__m512 x = XITER(j, XARGS);
		Traits::scatter16(dst_ptr[0] + j, dst_ptr[1] + j, dst_ptr[2] + j, dst_ptr[3] + j, dst_ptr[4] + j, dst_ptr[5] + j, dst_ptr[6] + j, dst_ptr[7] + j,
		                  dst_ptr[8] + j, dst_ptr[9] + j, dst_ptr[10] + j, dst_ptr[11] + j, dst_ptr[12] + j, dst_ptr[13] + j, dst_ptr[14] + j, dst_ptr[15] + j, x);
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


template <class Traits, unsigned N, bool UpdateAccum, class T = typename Traits::pixel_type>
inline FORCE_INLINE __m512 resize_line_v_fp_avx512_xiter(unsigned j,
                                                         const T * RESTRICT src_p0, const T * RESTRICT src_p1,
                                                         const T * RESTRICT src_p2, const T * RESTRICT src_p3,
                                                         const T * RESTRICT src_p4, const T * RESTRICT src_p5,
                                                         const T * RESTRICT src_p6, const T * RESTRICT src_p7, T * RESTRICT dst_p,
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
		accum0 = UpdateAccum ? _mm512_fmadd_ps(c0, x, Traits::load16(dst_p + j)) : _mm512_mul_ps(c0, x);
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
void resize_line_v_fp_avx512(const float *filter_data, const typename Traits::pixel_type * const *src_lines, typename Traits::pixel_type *dst, unsigned left, unsigned right)
{
	typedef typename Traits::pixel_type pixel_type;

	const pixel_type * RESTRICT src_p0 = src_lines[0];
	const pixel_type * RESTRICT src_p1 = src_lines[1];
	const pixel_type * RESTRICT src_p2 = src_lines[2];
	const pixel_type * RESTRICT src_p3 = src_lines[3];
	const pixel_type * RESTRICT src_p4 = src_lines[4];
	const pixel_type * RESTRICT src_p5 = src_lines[5];
	const pixel_type * RESTRICT src_p6 = src_lines[6];
	const pixel_type * RESTRICT src_p7 = src_lines[7];
	pixel_type * RESTRICT dst_p = dst;

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
#define XARGS src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, dst_p, c0, c1, c2, c3, c4, c5, c6, c7
	if (left != vec_left) {
		accum = XITER(vec_left - 16, XARGS);
		Traits::mask_store16(dst + vec_left - 16, 0xFFFFU << (16 - (vec_left - left)), accum);
	}
	for (unsigned j = vec_left; j < vec_right; j += 16) {
		accum = XITER(j, XARGS);
		Traits::mask_store16(dst + j, 0xFFFFU, accum);
	}
	if (right != vec_right) {
		accum = XITER(vec_right, XARGS);
		Traits::mask_store16(dst + vec_right, 0xFFFFU >> (16 - (right - vec_right)), accum);
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
#if defined(_M_X64) || defined(__X86_64__)
	_mm512_store_si512((__m512i *)dst, p);
#else
	_mm256_store_si256((__m256i *)dst, _mm512_cvtepi64_epi32(p));
#endif
}


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

		alignas(64) const pixel_type *src_ptr[16] = { 0 };
		alignas(64) pixel_type *dst_ptr[16] = { 0 };
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

	if (type == PixelType::HALF)
		ret = ztd::make_unique<ResizeImplH_FP_AVX512<f16_traits>>(context, height);
	else if (type == PixelType::FLOAT)
		ret = ztd::make_unique<ResizeImplH_FP_AVX512<f32_traits>>(context, height);

	return ret;
}

std::unique_ptr<graph::ImageFilter> create_resize_impl_v_avx512(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == PixelType::HALF)
		ret = ztd::make_unique<ResizeImplV_FP_AVX512<f16_traits>>(context, width);
	else if (type == PixelType::FLOAT)
		ret = ztd::make_unique<ResizeImplV_FP_AVX512<f32_traits>>(context, width);

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86_AVX512
