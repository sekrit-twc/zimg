#ifdef ZIMG_X86_AVX512

#include <cstdint>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "dither_x86.h"

#include "common/x86/avx512_util.h"

namespace zimg {
namespace depth {

namespace {

struct LoadU8 {
	typedef uint8_t type;

	static inline FORCE_INLINE __m512 load16(const uint8_t *ptr)
	{
		return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i *)ptr)));
	}
};

struct LoadU16 {
	typedef uint16_t type;

	static inline FORCE_INLINE __m512 load16(const uint16_t *ptr)
	{
		return _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i *)ptr)));

	}
};

struct LoadF16 {
	typedef uint16_t type;

	static inline FORCE_INLINE __m512 load16(const uint16_t *ptr)
	{
		return _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)ptr));
	}
};

struct LoadF32 {
	typedef float type;

	static inline FORCE_INLINE __m512 load16(const float *ptr)
	{
		return _mm512_load_ps(ptr);
	}
};

struct StoreU8 {
	typedef uint8_t type;

	static inline FORCE_INLINE void mask_store16(uint8_t *ptr, __mmask16 mask, __m512i x)
	{
		_mm_mask_storeu_epi8(ptr, mask, _mm512_cvtusepi32_epi8(x));
	}
};

struct StoreU16 {
	typedef uint16_t type;

	static inline FORCE_INLINE void mask_store16(uint16_t *ptr, __mmask16 mask, __m512i x)
	{
		_mm256_mask_storeu_epi16(ptr, mask, _mm512_cvtusepi32_epi16(x));
	}
};


inline FORCE_INLINE __m512i ordered_dither_avx512_xiter(__m512 x, unsigned j, const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                        const __m512 &scale, const __m512 &offset, const __m512i &out_max)
{
	__m512 dith = _mm512_load_ps(dither + ((dither_offset + j) & dither_mask));
	__m512i out;

	x = _mm512_fmadd_ps(scale, x, offset);
	x = _mm512_add_ps(x, dith);
	out = _mm512_cvtps_epi32(x);
	out = _mm512_min_epi32(out, out_max);
	out = _mm512_max_epi32(out, _mm512_setzero_si512());

	return out;
}

template <class Load, class Store>
inline FORCE_INLINE void ordered_dither_avx512_impl(const float *dither, unsigned dither_offset, unsigned dither_mask,
                                                    const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const typename Load::type *src_p = static_cast<const typename Load::type *>(src);
	typename Store::type *dst_p = static_cast<typename Store::type *>(dst);

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	const __m512 scale_ps = _mm512_set1_ps(scale);
	const __m512 offset_ps = _mm512_set1_ps(offset);
	const __m512i out_max = _mm512_set1_epi32((1 << bits) - 1);

#define XARGS dither, dither_offset, dither_mask, scale_ps, offset_ps, out_max
	if (left != vec_left) {
		__m512 x = Load::load16(src_p + vec_left - 16);
		__m512i out = ordered_dither_avx512_xiter(x, vec_left - 16, XARGS);

		Store::mask_store16(dst_p + vec_left - 16, mmask16_set_hi(vec_left - left), out);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m512 x = Load::load16(src_p + j);
		__m512i out = ordered_dither_avx512_xiter(x, j, XARGS);

		Store::mask_store16(dst_p + j, 0xFFFFU, out);
	}

	if (right != vec_right) {
		__m512 x = Load::load16(src_p + vec_right);
		__m512i out = ordered_dither_avx512_xiter(x, vec_right, XARGS);

		Store::mask_store16(dst_p + vec_right, mmask16_set_lo(right - vec_right), out);
	}
#undef XARGS
}

} // namespace


void ordered_dither_b2b_avx512(const float *dither, unsigned dither_offset, unsigned dither_mask,
                               const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx512_impl<LoadU8, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_b2w_avx512(const float *dither, unsigned dither_offset, unsigned dither_mask,
                               const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx512_impl<LoadU8, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_w2b_avx512(const float *dither, unsigned dither_offset, unsigned dither_mask,
                               const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx512_impl<LoadU16, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_w2w_avx512(const float *dither, unsigned dither_offset, unsigned dither_mask,
                               const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx512_impl<LoadU16, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_h2b_avx512(const float *dither, unsigned dither_offset, unsigned dither_mask,
                               const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx512_impl<LoadF16, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_h2w_avx512(const float *dither, unsigned dither_offset, unsigned dither_mask,
                               const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx512_impl<LoadF16, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_f2b_avx512(const float *dither, unsigned dither_offset, unsigned dither_mask,
                               const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx512_impl<LoadF32, StoreU8>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

void ordered_dither_f2w_avx512(const float *dither, unsigned dither_offset, unsigned dither_mask,
                               const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	ordered_dither_avx512_impl<LoadF32, StoreU16>(dither, dither_offset, dither_mask, src, dst, scale, offset, bits, left, right);
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86_AVX512
