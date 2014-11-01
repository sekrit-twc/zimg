#ifdef ZIMG_X86

#include <climits>
#include <immintrin.h>
#include "Common/align.h"
#include "operation.h"
#include "operation_impl.h"
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

class LookupTableOperationF6C : public Operation {
	float m_lut[1L << 16];
public:
	template <class Proc>
	explicit LookupTableOperationF6C(Proc proc)
	{
		for (uint32_t i = 0; i < UINT16_MAX + 1; ++i) {
			float x;

			_mm_store_ss(&x, _mm_cvtph_ps(_mm_set1_epi16(i)));
			m_lut[i] = proc(x);
		}
	}

	void process(float * const *ptr, int width) const override
	{
		for (int p = 0; p < 3; ++p) {
			for (int i = 0; i < mod(width, 8); i += 8) {
				uint16_t half[8];

				_mm_storeu_si128((__m128i *)half, _mm256_cvtps_ph(_mm256_load_ps(ptr[p] + i), 0));

				for (int k = 0; k < 8; ++k) {
					ptr[p][i + k] = m_lut[half[k]];
				}
			}
			for (int i = mod(width, 8); i < width; ++i) {
				uint16_t half = _mm_extract_epi16(_mm_cvtps_ph(_mm_load_ss(ptr[p] + i), 0), 0);
				ptr[p][i] = m_lut[half];
			}
		}
	}
};

} // namespace


PixelAdapter *create_pixel_adapter_f16c()
{
	return new PixelAdapterF16C{};
}

Operation *create_rec709_gamma_operation_f16c()
{
	return new LookupTableOperationF6C{ rec_709_gamma };
}

Operation *create_rec709_inverse_gamma_operation_f16c()
{
	return new LookupTableOperationF6C{ rec_709_inverse_gamma };
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86
