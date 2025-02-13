#ifdef ZIMG_X86

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
#include "common/make_array.h"
#include "common/pixel.h"
#include "common/unroll.h"
#include "graph/filter_base.h"
#include "resize/resize_impl.h"
#include "resize_impl_x86.h"

#include "common/x86/avx2_util.h"
#include "common/x86/avx512_util.h"

#define mm512_dpwssd_epi32(src, a, b) _mm512_add_epi32((src), _mm512_madd_epi16((a), (b)))
#include "resize_impl_avx512_common.h"

namespace zimg::resize {

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


template <class Traits, class T>
void transpose_line_16x16(T * RESTRICT dst, const T * const * RESTRICT src, unsigned left, unsigned right)
{
	typedef typename Traits::vec16_type vec16_type;

	for (unsigned j = left; j < right; j += 16) {
		vec16_type x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;

		x0  = Traits::load16_raw(src[0] + j);  x1  = Traits::load16_raw(src[1] + j);
		x2  = Traits::load16_raw(src[2] + j);  x3  = Traits::load16_raw(src[3] + j);
		x4  = Traits::load16_raw(src[4] + j);  x5  = Traits::load16_raw(src[5] + j);
		x6  = Traits::load16_raw(src[6] + j);  x7  = Traits::load16_raw(src[7] + j);
		x8  = Traits::load16_raw(src[8] + j);  x9  = Traits::load16_raw(src[9] + j);
		x10 = Traits::load16_raw(src[10] + j); x11 = Traits::load16_raw(src[11] + j);
		x12 = Traits::load16_raw(src[12] + j); x13 = Traits::load16_raw(src[13] + j);
		x14 = Traits::load16_raw(src[14] + j); x15 = Traits::load16_raw(src[15] + j);

		Traits::transpose16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);

		Traits::store16_raw(dst + 0,   x0);  Traits::store16_raw(dst + 16,  x1);
		Traits::store16_raw(dst + 32,  x2);  Traits::store16_raw(dst + 48,  x3);
		Traits::store16_raw(dst + 64,  x4);  Traits::store16_raw(dst + 80,  x5);
		Traits::store16_raw(dst + 96,  x6);  Traits::store16_raw(dst + 112, x7);
		Traits::store16_raw(dst + 128, x8);  Traits::store16_raw(dst + 144, x9);
		Traits::store16_raw(dst + 160, x10); Traits::store16_raw(dst + 176, x11);
		Traits::store16_raw(dst + 192, x12); Traits::store16_raw(dst + 208, x13);
		Traits::store16_raw(dst + 224, x14); Traits::store16_raw(dst + 240, x15);

		dst += 256;
	}
}


template <class Traits, int Taps>
inline FORCE_INLINE __m512 resize_line16_h_fp_avx512_xiter(unsigned j, const unsigned *filter_left, const float *filter_data, unsigned filter_stride, unsigned filter_width,
                                                           const typename Traits::pixel_type *src, unsigned src_base)
{
	static_assert(Taps <= 8, "only up to 8 taps can be unrolled");
	static_assert(Taps >= -3, "only up to 3 taps in epilogue");
	constexpr int Tail = Taps >= 4 ? Taps - 4 : Taps > 0 ? Taps : -Taps;

	typedef typename Traits::pixel_type pixel_type;

	const float *filter_coeffs = filter_data + j * filter_stride;
	const pixel_type *src_p = src + (filter_left[j] - src_base) * 16;

	__m512 accum0 = _mm512_setzero_ps();
	__m512 accum1 = _mm512_setzero_ps();
	__m512 coeffs;

	auto f = ZIMG_UNROLL_FUNC(kk)
	{
		__m512 &acc = kk % 2 ? accum1 : accum0;

		__m512 c = _mm512_shuffle_ps(coeffs, coeffs, static_cast<unsigned>(_MM_SHUFFLE(kk, kk, kk, kk)));
		__m512 x = Traits::load16(src_p + kk * 16);
		acc = _mm512_fmadd_ps(c, x, acc);
	};

	unsigned k_end = Taps >= 4 ? 4 : Taps > 0 ? 0 : floor_n(filter_width, 4);

	for (unsigned k = 0; k < k_end; k += 4) {
		coeffs = _mm512_broadcast_f32x4(_mm_load_ps(filter_coeffs + k));
		unroll<4>(f);
		src_p += 64;
	}

	if constexpr (Tail) {
		coeffs = _mm512_broadcast_f32x4(_mm_load_ps(filter_coeffs + k_end));
		unroll<Tail>(f);
	}

	if constexpr (Taps <= 0 || Taps >= 2)
		accum0 = _mm512_add_ps(accum0, accum1);

	return accum0;
}

template <class Traits, int Taps>
void resize_line16_h_fp_avx512(const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                               const typename Traits::pixel_type * RESTRICT src, typename Traits::pixel_type * const * /* RESTRICT */ dst, unsigned src_base, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

#define XITER resize_line16_h_fp_avx512_xiter<Traits, Taps>
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

		x0  = _mm512_load_ps(cache[0]);  x1  = _mm512_load_ps(cache[1]);
		x2  = _mm512_load_ps(cache[2]);  x3  = _mm512_load_ps(cache[3]);
		x4  = _mm512_load_ps(cache[4]);  x5  = _mm512_load_ps(cache[5]);
		x6  = _mm512_load_ps(cache[6]);  x7  = _mm512_load_ps(cache[7]);
		x8  = _mm512_load_ps(cache[8]);  x9  = _mm512_load_ps(cache[9]);
		x10 = _mm512_load_ps(cache[10]); x11 = _mm512_load_ps(cache[11]);
		x12 = _mm512_load_ps(cache[12]); x13 = _mm512_load_ps(cache[13]);
		x14 = _mm512_load_ps(cache[14]); x15 = _mm512_load_ps(cache[15]);

		mm512_transpose16_ps(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15);

		Traits::store16(dst[0] + j,  x0);  Traits::store16(dst[1] + j,  x1);
		Traits::store16(dst[2] + j,  x2);  Traits::store16(dst[3] + j,  x3);
		Traits::store16(dst[4] + j,  x4);  Traits::store16(dst[5] + j,  x5);
		Traits::store16(dst[6] + j,  x6);  Traits::store16(dst[7] + j,  x7);
		Traits::store16(dst[8] + j,  x8);  Traits::store16(dst[9] + j,  x9);
		Traits::store16(dst[10] + j, x10); Traits::store16(dst[11] + j, x11);
		Traits::store16(dst[12] + j, x12); Traits::store16(dst[13] + j, x13);
		Traits::store16(dst[14] + j, x14); Traits::store16(dst[15] + j, x15);
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
constexpr auto resize_line16_h_fp_avx512_jt_small = make_array(
	resize_line16_h_fp_avx512<Traits, 1>,
	resize_line16_h_fp_avx512<Traits, 2>,
	resize_line16_h_fp_avx512<Traits, 3>,
	resize_line16_h_fp_avx512<Traits, 4>,
	resize_line16_h_fp_avx512<Traits, 5>,
	resize_line16_h_fp_avx512<Traits, 6>,
	resize_line16_h_fp_avx512<Traits, 7>,
	resize_line16_h_fp_avx512<Traits, 8>);

template <class Traits>
constexpr auto resize_line16_h_fp_avx512_jt_large = make_array(
	resize_line16_h_fp_avx512<Traits, 0>,
	resize_line16_h_fp_avx512<Traits, -1>,
	resize_line16_h_fp_avx512<Traits, -2>,
	resize_line16_h_fp_avx512<Traits, -3>);


template <class Traits, unsigned Taps>
void resize_line_h_perm_fp_avx512(const unsigned * RESTRICT permute_left, const unsigned * RESTRICT permute_mask, const float * RESTRICT filter_data, unsigned input_width,
                                  const typename Traits::pixel_type * RESTRICT src, typename Traits::pixel_type * RESTRICT dst, unsigned left, unsigned right)
{
	static_assert(Taps <= 16, "permuted resampler only supports up to 16 taps");
	constexpr unsigned runahead = Taps >= 14 ? 32 : Taps >= 10 ? 28 : Taps >= 6 ? 24 : 20;

	unsigned vec_right = floor_n(right, 16);
	unsigned fallback_idx = vec_right;

#define mm512_alignr_epi8_ps(a, b, imm) _mm512_castsi512_ps(_mm512_alignr_epi8(_mm512_castps_si512((a)), _mm512_castps_si512((b)), (imm)))
	for (unsigned j = floor_n(left, 16); j < vec_right; j += 16) {

		unsigned left = permute_left[j / 16];

		if (input_width - left < runahead) {
			fallback_idx = j;
			break;
		}

		const __m512i mask = _mm512_load_si512(permute_mask + j);
		const float *data = filter_data + static_cast<size_t>(j) * Taps;

		__m512 accum0 = _mm512_setzero_ps();
		__m512 accum1 = _mm512_setzero_ps();

		__m512 x0 = Traits::load16(src + left + 0);
		__m512 x4 = Traits::load16(src + left + 4);
		__m512 x8 = Taps >= 6 ? Traits::load16(src + left + 8) : _mm512_setzero_ps();
		__m512 x12 = Taps >= 10 ? Traits::load16(src + left + 12) : _mm512_setzero_ps();
		__m512 x16 = Taps >= 14 ? Traits::load16(src + left + 16) : _mm512_setzero_ps();

		unroll<Taps>(ZIMG_UNROLL_FUNC(k)
		{
			__m512 &acc = k % 2 ? accum1 : accum0;

			__m512 coeffs = _mm512_load_ps(data + k * 16);
			__m512 x;

			if constexpr (k >= 12)
				x = k % 4 ? mm512_alignr_epi8_ps(x16, x12, (k % 4) * 4) : x12;
			else if constexpr (k >= 8)
				x = k % 4 ? mm512_alignr_epi8_ps(x12, x8, (k % 4) * 4) : x8;
			else if constexpr (k >= 4)
				x = k % 4 ? mm512_alignr_epi8_ps(x8, x4, (k % 4) * 4) : x4;
			else
				x = k % 4 ? mm512_alignr_epi8_ps(x4, x0, (k % 4) * 4) : x0;

			x = _mm512_permutexvar_ps(mask, x);
			acc = _mm512_fmadd_ps(coeffs, x, acc);
		});

		accum0 = _mm512_add_ps(accum0, accum1);
		Traits::store16(dst + j, accum0);
	}
#undef mm512_alignr_epi8_ps
	for (unsigned j = fallback_idx; j < right; j += 16) {
		unsigned left = permute_left[j / 16];

		const __m512i mask = _mm512_load_si512(permute_mask + j);
		const float *data = filter_data + j * Taps;

		__m512 accum0 = _mm512_setzero_ps();
		__m512 accum1 = _mm512_setzero_ps();
		__m512 x, coeffs;

		for (unsigned k = 0; k < Taps; ++k) {
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
constexpr auto resize_line_h_perm_fp_avx512_jt = make_array(
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
	resize_line_h_perm_fp_avx512<Traits, 16>);


template <class Traits, unsigned Taps, bool Continue, class T = typename Traits::pixel_type>
inline FORCE_INLINE __m512 resize_line_v_fp_avx512_xiter(unsigned j, const T * const srcp[8], const T *accum_p, const __m512 c[8])
{
	typedef typename Traits::pixel_type pixel_type;
	static_assert(std::is_same_v<pixel_type, T>, "must not specify T");

	__m512 accum0 = _mm512_setzero_ps();
	__m512 accum1 = _mm512_setzero_ps();

	unroll<Taps>(ZIMG_UNROLL_FUNC(k)
	{
		__m512 &acc = k % 2 ? accum1 : accum0;
		__m512 x = Traits::load16(srcp[k] + j);

		if constexpr (k == 0 && Continue)
			acc = _mm512_fmadd_ps(c[k], x, Traits::load16(accum_p + j));
		else if constexpr (k == 0 || k == 1)
			acc = _mm512_mul_ps(c[k], x);
		else
			acc = _mm512_fmadd_ps(c[k], x, acc);
	});

	if constexpr (Taps >= 2) accum0 = _mm512_add_ps(accum0, accum1);
	return accum0;
}

template <class Traits, unsigned Taps, bool Continue>
void resize_line_v_fp_avx512(const float * RESTRICT filter_data, const typename Traits::pixel_type * const * RESTRICT src, typename Traits::pixel_type * RESTRICT dst,
                             unsigned left, unsigned right)
{
	typedef typename Traits::pixel_type pixel_type;

	const pixel_type *srcp[8] = {src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7]};
	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m512 c[8] = {
		_mm512_broadcastss_ps(_mm_load_ss(filter_data + 0)),
		_mm512_broadcastss_ps(_mm_load_ss(filter_data + 1)),
		_mm512_broadcastss_ps(_mm_load_ss(filter_data + 2)),
		_mm512_broadcastss_ps(_mm_load_ss(filter_data + 3)),
		_mm512_broadcastss_ps(_mm_load_ss(filter_data + 4)),
		_mm512_broadcastss_ps(_mm_load_ss(filter_data + 5)),
		_mm512_broadcastss_ps(_mm_load_ss(filter_data + 6)),
		_mm512_broadcastss_ps(_mm_load_ss(filter_data + 7)),
	};

#define XITER resize_line_v_fp_avx512_xiter<Traits, Taps, Continue>
#define XARGS srcp, dst, c
	if (left != vec_left) {
		__m512 accum = XITER(vec_left - 16, XARGS);
		Traits::mask_store16(dst + vec_left - 16, mmask16_set_hi(vec_left - left), accum);
	}
	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m512 accum = XITER(j, XARGS);
		Traits::mask_store16(dst + j, 0xFFFFU, accum);
	}
	if (right != vec_right) {
		__m512 accum = XITER(vec_right, XARGS);
		Traits::mask_store16(dst + vec_right, mmask16_set_lo(right - vec_right), accum);
	}
#undef XITER
#undef XARGS
}

template <class Traits>
constexpr auto resize_line_v_fp_avx512_jt_init = make_array(
	resize_line_v_fp_avx512<Traits, 1, false>,
	resize_line_v_fp_avx512<Traits, 2, false>,
	resize_line_v_fp_avx512<Traits, 3, false>,
	resize_line_v_fp_avx512<Traits, 4, false>,
	resize_line_v_fp_avx512<Traits, 5, false>,
	resize_line_v_fp_avx512<Traits, 6, false>,
	resize_line_v_fp_avx512<Traits, 7, false>,
	resize_line_v_fp_avx512<Traits, 8, false>);

template <class Traits>
constexpr auto resize_line_v_fp_avx512_jt_cont = make_array(
	resize_line_v_fp_avx512<Traits, 1, true>,
	resize_line_v_fp_avx512<Traits, 2, true>,
	resize_line_v_fp_avx512<Traits, 3, true>,
	resize_line_v_fp_avx512<Traits, 4, true>,
	resize_line_v_fp_avx512<Traits, 5, true>,
	resize_line_v_fp_avx512<Traits, 6, true>,
	resize_line_v_fp_avx512<Traits, 7, true>,
	resize_line_v_fp_avx512<Traits, 8, true>);


template <class Traits>
class ResizeImplH_FP_AVX512 : public ResizeImplH {
	typedef typename Traits::pixel_type pixel_type;
	typedef typename decltype(resize_line16_h_fp_avx512_jt_small<Traits>)::value_type func_type;

	func_type m_func;
public:
	ResizeImplH_FP_AVX512(const FilterContext &filter, unsigned height) try :
		ResizeImplH(filter, height, Traits::type_constant),
		m_func{}
	{
		m_desc.step = 16;
		m_desc.scratchpad_size = (ceil_n(checked_size_t{ filter.input_width }, 16) * sizeof(pixel_type) * 16).get();

		if (filter.filter_width <= 8)
			m_func = resize_line16_h_fp_avx512_jt_small<Traits>[filter.filter_width - 1];
		else
			m_func = resize_line16_h_fp_avx512_jt_large<Traits>[filter.filter_width % 4];
	} catch (const std::overflow_error &) {
		error::throw_<error::OutOfMemory>();
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		auto range = get_col_deps(left, right);

		alignas(64) const pixel_type *src_ptr[16];
		alignas(64) pixel_type *dst_ptr[16];
		pixel_type *transpose_buf = static_cast<pixel_type *>(tmp);
		unsigned height = m_desc.format.height;

		calculate_line_address(src_ptr + 0, in->ptr, in->stride, in->mask, i + 0, height);
		calculate_line_address(src_ptr + 8, in->ptr, in->stride, in->mask, i + std::min(8U, height - i - 1), height);

		transpose_line_16x16<Traits>(transpose_buf, src_ptr, floor_n(range.first, 16), ceil_n(range.second, 16));

		calculate_line_address(dst_ptr + 0, out->ptr, out->stride, out->mask, i + 0, height);
		calculate_line_address(dst_ptr + 8, out->ptr, out->stride, out->mask, i + std::min(8U, height - i - 1), height);

		m_func(m_filter.left.data(), m_filter.data.data(), m_filter.stride, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 16), left, right);
	}
};


template <class Traits>
class ResizeImplH_Permute_FP_AVX512 : public graph::FilterBase {
	typedef typename Traits::pixel_type pixel_type;
	typedef typename decltype(resize_line_h_perm_fp_avx512_jt<Traits>)::value_type func_type;

	struct PermuteContext {
		AlignedVector<unsigned> left;
		AlignedVector<unsigned> permute;
		AlignedVector<float> data;
		unsigned filter_rows;
		unsigned filter_width;
		unsigned input_width;
	};

	PermuteContext m_context;
	func_type m_func;

	ResizeImplH_Permute_FP_AVX512(PermuteContext context, unsigned height) :
		m_context(std::move(context)),
		m_func{ resize_line_h_perm_fp_avx512_jt<Traits>[m_context.filter_width - 1] }
	{
		m_desc.format = { context.filter_rows, height, pixel_size(Traits::type_constant) };
		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.step = 1;
		m_desc.alignment_mask = 15;
		m_desc.flags.entire_row = !std::is_sorted(m_context.left.begin(), m_context.left.end());
	}
public:
	static std::unique_ptr<graphengine::Filter> create(const FilterContext &filter, unsigned height)
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

		std::unique_ptr<graphengine::Filter> ret{ new ResizeImplH_Permute_FP_AVX512(std::move(context), height) };
		return ret;
	}

	pair_unsigned get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override
	{
		if (m_desc.flags.entire_row)
			return{ 0, m_context.input_width };

		unsigned input_width = m_context.input_width;
		unsigned right_base = m_context.left[(right + 15) / 16 - 1];
		unsigned iter_width = m_context.filter_width + 16;
		return{ m_context.left[left / 16],  right_base + std::min(input_width - right_base, iter_width) };
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		m_func(m_context.left.data(), m_context.permute.data(), m_context.data.data(), m_context.input_width, in->get_line<pixel_type>(i), out->get_line<pixel_type>(i), left, right);
	}
};


template <class Traits>
class ResizeImplV_FP_AVX512 : public ResizeImplV {
	typedef typename Traits::pixel_type pixel_type;
public:
	ResizeImplV_FP_AVX512(const FilterContext &filter, unsigned width) :
		ResizeImplV(filter, width, Traits::type_constant)
	{}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		const float *filter_data = m_filter.data.data() + i * m_filter.stride;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		alignas(64) const pixel_type *src_lines[8];
		pixel_type *dst_line = out->get_line<pixel_type>(i);

		{
			unsigned taps_remain = std::min(filter_width - 0, 8U);
			unsigned top = m_filter.left[i] + 0;

			calculate_line_address(src_lines, in->ptr, in->stride, in->mask, top, src_height);
			resize_line_v_fp_avx512_jt_init<Traits>[taps_remain - 1](filter_data + 0, src_lines, dst_line, left, right);
		}

		for (unsigned k = 8; k < filter_width; k += 8) {
			unsigned taps_remain = std::min(filter_width - k, 8U);
			unsigned top = m_filter.left[i] + k;

			calculate_line_address(src_lines, in->ptr, in->stride, in->mask, top, src_height);
			resize_line_v_fp_avx512_jt_cont<Traits>[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
		}
	}
};

} // namespace


std::unique_ptr<graphengine::Filter> create_resize_impl_h_avx512(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

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
			ret = std::make_unique<ResizeImplH_U16_AVX512>(context, height, depth);
		else if (type == PixelType::HALF)
			ret = std::make_unique<ResizeImplH_FP_AVX512<f16_traits>>(context, height);
		else if (type == PixelType::FLOAT)
			ret = std::make_unique<ResizeImplH_FP_AVX512<f32_traits>>(context, height);
	}

	return ret;
}

std::unique_ptr<graphengine::Filter> create_resize_impl_v_avx512(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

	if (type == PixelType::WORD)
		ret = std::make_unique<ResizeImplV_U16_AVX512>(context, width, depth);
	else if (type == PixelType::HALF)
		ret = std::make_unique<ResizeImplV_FP_AVX512<f16_traits>>(context, width);
	else if (type == PixelType::FLOAT)
		ret = std::make_unique<ResizeImplV_FP_AVX512<f32_traits>>(context, width);

	return ret;
}

} // namespace zimg::resize

#endif // ZIMG_X86
