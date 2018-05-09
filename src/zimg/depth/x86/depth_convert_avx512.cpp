#ifdef ZIMG_X86_AVX512

#include <cstdint>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "depth_convert_x86.h"

#include "common/x86/avx512_util.h"

namespace zimg {
namespace depth {

namespace {

struct LoadU8 {
	typedef uint8_t src_type;

	static __m512 load16(const uint8_t *ptr)
	{
		return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)ptr)));
	}
};

struct LoadU16 {
	typedef uint16_t src_type;

	static __m512 load16(const uint16_t *ptr)
	{
		return _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i *)ptr)));
	}
};

struct StoreF16 {
	typedef uint16_t dst_type;

	static void mask_store16(uint16_t *ptr, __mmask16 mask, __m512 x)
	{
		_mm256_mask_storeu_epi16(ptr, mask, _mm512_cvtps_ph(x, 0));
	}
};

struct StoreF32 {
	typedef float dst_type;

	static void mask_store16(float *ptr, __mmask16 mask, __m512 x)
	{
		_mm512_mask_store_ps(ptr, mask, x);
	}
};

template <class Load, class Store>
inline FORCE_INLINE void depth_convert_avx512_impl(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const typename Load::src_type *src_p = static_cast<const typename Load::src_type *>(src);
	typename Store::dst_type *dst_p = static_cast<typename Store::dst_type *>(dst);

	unsigned vec_left = floor_n(left, 16);
	unsigned vec_right = ceil_n(right, 16);

	const __m512 scale_ps = _mm512_set1_ps(scale);
	const __m512 offset_ps = _mm512_set1_ps(offset);

	if (left != vec_left) {
		__m512 x = Load::load16(src_p + vec_left - 16);
		x = _mm512_fmadd_ps(scale_ps, x, offset_ps);

		Store::mask_store16(dst_p + vec_left - 16, mmask16_set_hi(vec_left - left), x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m512 x = Load::load16(src_p + j);
		x = _mm512_fmadd_ps(scale_ps, x, offset_ps);

		Store::mask_store16(dst_p + j, 0xFFFFU, x);
	}

	if (right != vec_right) {
		__m512 x = Load::load16(src_p + vec_right);
		x = _mm512_fmadd_ps(scale_ps, x, offset_ps);

		Store::mask_store16(dst_p + vec_right, mmask16_set_lo(right - vec_right), x);
	}
}

} // namespace


void depth_convert_b2h_avx512(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	depth_convert_avx512_impl<LoadU8, StoreF16>(src, dst, scale, offset, left, right);
}

void depth_convert_b2f_avx512(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	depth_convert_avx512_impl<LoadU8, StoreF32>(src, dst, scale, offset, left, right);
}

void depth_convert_w2h_avx512(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	depth_convert_avx512_impl<LoadU16, StoreF16>(src, dst, scale, offset, left, right);
}

void depth_convert_w2f_avx512(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	depth_convert_avx512_impl<LoadU16, StoreF32>(src, dst, scale, offset, left, right);
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86_AVX512
