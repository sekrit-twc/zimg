#ifdef ZIMG_X86

#include "common/ccdep.h"

#include <immintrin.h>
#include "common/align.h"
#include "f16c_x86.h"

#include "common/x86/sse2_util.h"
#include "common/x86/avx_util.h"

namespace zimg {
namespace depth {

void f16c_half_to_float_ivb(const void *src, void *dst, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	if (left != vec_left) {
		__m256 x = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(src_p + vec_left - 8)));
		mm256_store_idxhi_ps(dst_p + vec_left - 8, x, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m256 x = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(src_p + j)));
		_mm256_store_ps(dst_p + j, x);
	}

	if (right != vec_right) {
		__m256 x = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(src_p + vec_right)));
		mm256_store_idxlo_ps(dst_p + vec_right, x, right % 8);
	}
}

void f16c_float_to_half_ivb(const void *src, void *dst, unsigned left, unsigned right)
{
	const float *src_p = static_cast<const float *>(src);
	uint16_t *dst_p = static_cast<uint16_t *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	if (left != vec_left) {
		__m128i x = _mm256_cvtps_ph(_mm256_load_ps(src_p + vec_left - 8), 0);
		mm_store_idxhi_epi16((__m128i *)(dst_p + vec_left - 8), x, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m128i x = _mm256_cvtps_ph(_mm256_load_ps(src_p + j), 0);
		_mm_store_si128((__m128i *)(dst_p + j), x);
	}

	if (right != vec_right) {
		__m128i x = _mm256_cvtps_ph(_mm256_load_ps(src_p + vec_right), 0);
		mm_store_idxlo_epi16((__m128i *)(dst_p + vec_right), x, right % 8);
	}
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
