#ifdef ZIMG_X86

#include <emmintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "f16c_x86.h"

#include "common/x86/sse_util.h"
#include "common/x86/sse2_util.h"

namespace zimg {
namespace depth {

namespace {

inline FORCE_INLINE __m128i mm_blendv_ps(__m128i a, __m128i b, __m128i mask)
{
	a = _mm_and_si128(mask, a);
	b = _mm_andnot_si128(mask, b);

	return _mm_or_si128(a, b);
}

inline FORCE_INLINE __m128i mm_min_epi32(__m128i a, __m128i b)
{
	__m128i mask = _mm_cmplt_epi32(a, b);
	return mm_blendv_ps(a, b, mask);
}

inline FORCE_INLINE __m128 mm_cvtph_ps(__m128i x)
{
	__m128 magic = _mm_castsi128_ps(_mm_set1_epi32(113UL << 23));
	__m128i shift_exp = _mm_set1_epi32(0x7C00UL << 13);
	__m128i sign_mask = _mm_set1_epi32(0x8000U);
	__m128i mant_mask = _mm_set1_epi32(0x7FFF);
	__m128i exp_adjust = _mm_set1_epi32((127UL - 15UL) << 23);
	__m128i exp_adjust_nan = _mm_set1_epi32((127UL - 16UL) << 23);
	__m128i exp_adjust_denorm = _mm_set1_epi32(1UL << 23);
	__m128i zero = _mm_set1_epi16(0);

	__m128i exp, ret, ret_nan, ret_denorm, sign, mask0, mask1;

	x = _mm_unpacklo_epi16(x, zero);

	ret = _mm_and_si128(x, mant_mask);
	ret = _mm_slli_epi32(ret, 13);
	exp = _mm_and_si128(shift_exp, ret);
	ret = _mm_add_epi32(ret, exp_adjust);

	mask0 = _mm_cmpeq_epi32(exp, shift_exp);
	mask1 = _mm_cmpeq_epi32(exp, zero);

	ret_nan = _mm_add_epi32(ret, exp_adjust_nan);
	ret_denorm = _mm_add_epi32(ret, exp_adjust_denorm);
	ret_denorm = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(ret_denorm), magic));

	sign = _mm_and_si128(x, sign_mask);
	sign = _mm_slli_epi32(sign, 16);

	ret = mm_blendv_ps(ret_nan, ret, mask0);
	ret = mm_blendv_ps(ret_denorm, ret, mask1);

	ret = _mm_or_si128(ret, sign);
	return _mm_castsi128_ps(ret);
}

inline FORCE_INLINE __m128i mm_cvtps_ph(__m128 x)
{
	__m128 magic = _mm_castsi128_ps(_mm_set1_epi32(15UL << 23));
	__m128i inf = _mm_set1_epi32(255UL << 23);
	__m128i f16inf = _mm_set1_epi32(31UL << 23);
	__m128i sign_mask = _mm_set1_epi32(0x80000000UL);
	__m128i round_mask = _mm_set1_epi32(~0x0FFFU);

	__m128i ret_0x7E00 = _mm_set1_epi32(0x7E00);
	__m128i ret_0x7C00 = _mm_set1_epi32(0x7C00);

	__m128i f, sign, ge_inf, eq_inf;

	f = _mm_castps_si128(x);
	sign = _mm_and_si128(f, sign_mask);
	f = _mm_xor_si128(f, sign);

	ge_inf = _mm_cmpgt_epi32(f, inf);
	eq_inf = _mm_cmpeq_epi32(f, inf);

	f = _mm_and_si128(f, round_mask);
	f = _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(f), magic));
	f = _mm_sub_epi32(f, round_mask);

	f = mm_min_epi32(f, f16inf);
	f = _mm_srli_epi32(f, 13);

	f = mm_blendv_ps(ret_0x7E00, f, ge_inf);
	f = mm_blendv_ps(ret_0x7C00, f, eq_inf);

	sign = _mm_srli_epi32(sign, 16);
	f = _mm_or_si128(f, sign);

	f = mm_packus_epi32(f, _mm_setzero_si128());
	return f;
}

} // namespace


void f16c_half_to_float_sse2(const void *src, void *dst, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	__m128i f16_val;
	__m128 f32_val;

	if (left != vec_left) {
		f16_val = _mm_loadl_epi64((const __m128i *)(src_p + vec_left - 4));
		f32_val = mm_cvtph_ps(f16_val);
		mm_store_idxhi_ps(dst_p + vec_left - 4, f32_val, left % 4);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		f16_val = _mm_loadl_epi64((const __m128i *)(src_p + j));
		f32_val = mm_cvtph_ps(f16_val);
		_mm_store_ps(dst_p + j, f32_val);
	}

	if (right != vec_right) {
		f16_val = _mm_loadl_epi64((const __m128i *)(src_p + vec_right));
		f32_val = mm_cvtph_ps(f16_val);
		mm_store_idxlo_ps(dst_p + vec_right, f32_val, right % 4);
	}
}

void f16c_float_to_half_sse2(const void *src, void *dst, unsigned left, unsigned right)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	__m128 f32_val;
	__m128i f16_val;
	alignas(16) uint64_t f16qw;

	if (left != vec_left) {
		f32_val = _mm_load_ps(src_p + vec_left - 4);
		f16_val = mm_cvtps_ph(f32_val);
		_mm_storel_epi64((__m128i *)&f16qw, f16_val);

		for (unsigned j = 0; j < vec_left - left; ++j) {
			dst_p[vec_left - j - 1] = static_cast<uint16_t>(f16qw >> (48 - 16 * j));
		}
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		f32_val = _mm_load_ps(src_p + j);
		f16_val = mm_cvtps_ph(f32_val);
		_mm_storel_epi64((__m128i *)(dst_p + j), f16_val);
	}

	if (right != vec_right) {
		f32_val = _mm_load_ps(src_p + vec_right);
		f16_val = mm_cvtps_ph(f32_val);
		_mm_storel_epi64((__m128i *)&f16qw, f16_val);

		for (unsigned j = 0; j < right - vec_right; ++j) {
			dst_p[vec_right + j] = static_cast<uint16_t>(f16qw >> (16 * j));
		}
	}
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
