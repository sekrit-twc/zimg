#ifdef ZIMG_X86

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "depth_convert_x86.h"

#include "common/x86/sse2_util.h"
#include "common/x86/avx_util.h"
#include "common/x86/avx2_util.h"

namespace zimg {
namespace depth {

namespace {

struct LoadU8 {
	typedef uint8_t src_type;

	static __m256 load8(const uint8_t *ptr)
	{
		return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)ptr)));
	}
};

struct LoadU16 {
	typedef uint16_t src_type;

	static __m256 load8(const uint16_t *ptr)
	{
		return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_load_si128((const __m128i *)ptr)));
	}
};

struct StoreF16 {
	typedef uint16_t dst_type;

	static void store8(uint16_t *ptr, __m256 x)
	{
		_mm_store_si128((__m128i *)ptr, _mm256_cvtps_ph(x, 0));
	}

	static void store8_idxlo(uint16_t *ptr, __m256 x, unsigned idx)
	{
		mm_store_idxlo_epi16((__m128i *)ptr, _mm256_cvtps_ph(x, 0), idx);
	}

	static void store8_idxhi(uint16_t *ptr, __m256 x, unsigned idx)
	{
		mm_store_idxhi_epi16((__m128i *)ptr, _mm256_cvtps_ph(x, 0), idx);
	}
};

struct StoreF32 {
	typedef float dst_type;

	static void store8(float *ptr, __m256 x) { _mm256_store_ps(ptr, x); }

	static void store8_idxlo(float *ptr, __m256 x, unsigned idx) { mm256_store_idxlo_ps(ptr, x, idx); }

	static void store8_idxhi(float *ptr, __m256 x, unsigned idx) { mm256_store_idxhi_ps(ptr, x, idx); }
};

template <class Load, class Store>
inline FORCE_INLINE void depth_convert_avx2_impl(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	const typename Load::src_type *src_p = static_cast<const typename Load::src_type *>(src);
	typename Store::dst_type *dst_p = static_cast<typename Store::dst_type *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const __m256 scale_ps = _mm256_set1_ps(scale);
	const __m256 offset_ps = _mm256_set1_ps(offset);

	if (left != vec_left) {
		__m256 x = Load::load8(src_p + vec_left - 8);
		x = _mm256_fmadd_ps(scale_ps, x, offset_ps);

		Store::store8_idxhi(dst_p + vec_left - 8, x, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m256 x = Load::load8(src_p + j);
		x = _mm256_fmadd_ps(scale_ps, x, offset_ps);
		Store::store8(dst_p + j, x);
	}

	if (right != vec_right) {
		__m256 x = Load::load8(src_p + vec_right);
		x = _mm256_fmadd_ps(scale_ps, x, offset_ps);

		Store::store8_idxlo(dst_p + vec_right, x, right % 8);
	}
}

} // namespace


void depth_convert_b2h_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	depth_convert_avx2_impl<LoadU8, StoreF16>(src, dst, scale, offset, left, right);
}

void depth_convert_b2f_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	depth_convert_avx2_impl<LoadU8, StoreF32>(src, dst, scale, offset, left, right);
}

void depth_convert_w2h_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	depth_convert_avx2_impl<LoadU16, StoreF16>(src, dst, scale, offset, left, right);
}

void depth_convert_w2f_avx2(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)
{
	depth_convert_avx2_impl<LoadU16, StoreF32>(src, dst, scale, offset, left, right);
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
