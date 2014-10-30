#ifdef ZIMG_X86

#include <immintrin.h>
#include "Common/align.h"
#include "operation.h"
#include "operation_impl_x86.h"

namespace zimg {;
namespace colorspace {;

namespace {;

class PixelAdapterF16C : public PixelAdapter {
public:
	void f16_to_f32(const uint16_t *src, float *dst, int width) const override
	{
		for (int i = 0; i < mod(width, 8); i += 8) {
			__m128i f16 = _mm_load_si128((const __m128i *)(src + i));
			__m256 f32 = _mm256_cvtph_ps(f16);
			_mm256_store_ps(dst + i, f32);
		}
		for (int i = mod(width, 8); i < width; ++i) {
			__m128i f16 = _mm_set1_epi16(src[i]);
			__m128 f32 = _mm_cvtph_ps(f16);
			_mm_store_ss(dst + i, f32);
		}
	}

	void f16_from_f32(const float *src, uint16_t *dst, int width) const override
	{
		for (int i = 0; i < mod(width, 8); i += 8) {
			__m256 f32 = _mm256_load_ps(src + i);
			__m128i f16 = _mm256_cvtps_ph(f32, 0);
			_mm_store_si128((__m128i *)(dst + i), f16);
		}
		for (int i = mod(width, 8); i < width; ++i) {
			__m128 f32 = _mm_load_ss(src + i);
			__m128i f16 = _mm_cvtps_ph(f32, 0);
			dst[i] = _mm_extract_epi16(f16, 0);
		}
	}
};

} // namespace


PixelAdapter *create_pixel_adapter_f16c()
{
	return new PixelAdapterF16C{};
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86
