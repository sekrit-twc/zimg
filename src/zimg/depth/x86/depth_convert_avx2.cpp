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

	static inline FORCE_INLINE  __m256 load8(const uint8_t *ptr)
	{
		return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)ptr)));
	}

	static inline FORCE_INLINE __m256i load16i(const uint8_t *ptr)
	{
		return _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i *)ptr));
	}
};

struct LoadU16 {
	typedef uint16_t src_type;

	static inline FORCE_INLINE __m256 load8(const uint16_t *ptr)
	{
		return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_load_si128((const __m128i *)ptr)));
	}

	static inline FORCE_INLINE __m256i load16i(const uint16_t *ptr)
	{
		return _mm256_load_si256((const __m256i *)ptr);
	}
};

struct StoreU8 {
	typedef uint8_t dst_type;

	static inline FORCE_INLINE void store16i(uint8_t *ptr, __m256i x)
	{
		x = _mm256_permute4x64_epi64(_mm256_packus_epi16(x, x), _MM_SHUFFLE(3, 1, 2, 0));
		_mm_store_si128((__m128i *)ptr, _mm256_castsi256_si128(x));
	}

	static inline FORCE_INLINE void store16i_idxlo(uint8_t *ptr, __m256i x, unsigned idx)
	{
		x = _mm256_permute4x64_epi64(_mm256_packus_epi16(x, x), _MM_SHUFFLE(3, 1, 2, 0));
		mm_store_idxlo_epi8((__m128i *)ptr, _mm256_castsi256_si128(x), idx);
	}

	static inline FORCE_INLINE void store16i_idxhi(uint8_t *ptr, __m256i x, unsigned idx)
	{
		x = _mm256_permute4x64_epi64(_mm256_packus_epi16(x, x), _MM_SHUFFLE(3, 1, 2, 0));
		mm_store_idxhi_epi8((__m128i *)ptr, _mm256_castsi256_si128(x), idx);
	}
};

struct StoreU16 {
	typedef uint16_t dst_type;

	static inline FORCE_INLINE void store16i(uint16_t *ptr, __m256i x)
	{
		_mm256_store_si256((__m256i *)ptr, x);
	}

	static inline FORCE_INLINE void store16i_idxlo(uint16_t *ptr, __m256i x, unsigned idx)
	{
		mm256_store_idxlo_epi16((__m256i *)ptr, x, idx);
	}

	static inline FORCE_INLINE void store16i_idxhi(uint16_t *ptr, __m256i x, unsigned idx)
	{
		mm256_store_idxhi_epi16((__m256i *)ptr, x, idx);
	}
};

struct StoreF16 {
	typedef uint16_t dst_type;

	static inline FORCE_INLINE void store8(uint16_t *ptr, __m256 x)
	{
		_mm_store_si128((__m128i *)ptr, _mm256_cvtps_ph(x, 0));
	}

	static inline FORCE_INLINE void store8_idxlo(uint16_t *ptr, __m256 x, unsigned idx)
	{
		mm_store_idxlo_epi16((__m128i *)ptr, _mm256_cvtps_ph(x, 0), idx);
	}

	static inline FORCE_INLINE void store8_idxhi(uint16_t *ptr, __m256 x, unsigned idx)
	{
		mm_store_idxhi_epi16((__m128i *)ptr, _mm256_cvtps_ph(x, 0), idx);
	}
};

struct StoreF32 {
	typedef float dst_type;

	static inline FORCE_INLINE void store8(float *ptr, __m256 x)
	{
		_mm256_store_ps(ptr, x);
	}

	static inline FORCE_INLINE void store8_idxlo(float *ptr, __m256 x, unsigned idx)
	{
		mm256_store_idxlo_ps(ptr, x, idx);
	}

	static inline FORCE_INLINE void store8_idxhi(float *ptr, __m256 x, unsigned idx)
	{
		mm256_store_idxhi_ps(ptr, x, idx);
	}
};

template <class Load, class Store>
inline FORCE_INLINE void left_shift_avx2_impl(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	const typename Load::src_type *src_p = static_cast<const typename Load::src_type *>(src);
	typename Store::dst_type *dst_p = static_cast<typename Store::dst_type *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	__m128i count = _mm_set1_epi64x(shift);

	if (left != vec_left) {
		__m256i x = Load::load16i(src_p + vec_left - 16);
		x = _mm256_sll_epi16(x, count);

		Store::store16i_idxhi(dst_p + vec_left - 16, x, left % 16);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m256i x = Load::load16i(src_p + j);
		x = _mm256_sll_epi16(x, count);
		Store::store16i(dst_p + j, x);
	}

	if (right != vec_right) {
		__m256i x = Load::load16i(src_p + vec_right);
		x = _mm256_sll_epi16(x, count);

		Store::store16i_idxlo(dst_p + vec_right, x, right % 16);
	}
}

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


void left_shift_b2b_avx2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	left_shift_avx2_impl<LoadU8, StoreU8>(src, dst, shift, left, right);
}

void left_shift_b2w_avx2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	left_shift_avx2_impl<LoadU8, StoreU16>(src, dst, shift, left, right);
}

void left_shift_w2b_avx2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	left_shift_avx2_impl<LoadU16, StoreU8>(src, dst, shift, left, right);
}

void left_shift_w2w_avx2(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
{
	left_shift_avx2_impl<LoadU16, StoreU16>(src, dst, shift, left, right);
}

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
