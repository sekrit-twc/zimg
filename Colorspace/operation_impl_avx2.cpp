#ifdef ZIMG_X86

#include <immintrin.h>
#include "Common/align.h"
#include "Common/osdep.h"
#include "matrix3.h"
#include "operation.h"
#include "operation_impl.h"
#include "operation_impl_x86.h"

namespace zimg {;
namespace colorspace {;

namespace {;

inline FORCE_INLINE uint16_t float_to_half(float x)
{
	__m128 f32 = _mm_set_ps1(x);
	__m128i f16 = _mm_cvtps_ph(f32, 0);
	return _mm_extract_epi16(f16, 0);
}

inline FORCE_INLINE float half_to_float(uint16_t x)
{
	__m128i f16 = _mm_set1_epi16(x);
	__m128 f32 = _mm_cvtph_ps(f16);
	return _mm_cvtss_f32(f32);
}

class PixelAdapterAVX2 : public PixelAdapter {
public:
	void f16_to_f32(const uint16_t *src, float *dst, int width) const override
	{
		for (int i = 0; i < mod(width, 8); i += 8) {
			__m128i f16 = _mm_load_si128((const __m128i *)&src[i]);
			__m256 f32 = _mm256_cvtph_ps(f16);
			_mm256_store_ps(&dst[i], f32);
		}
		for (int i = mod(width, 8); i < width; ++i) {
			dst[i] = half_to_float(src[i]);
		}
	}

	void f16_from_f32(const float *src, uint16_t *dst, int width) const override
	{
		for (int i = 0; i < mod(width, 8); i += 8) {
			__m256 f32 = _mm256_load_ps(&src[i]);
			__m128i f16 = _mm256_cvtps_ph(f32, 0);
			_mm_store_si128((__m128i *)&dst[i], f16);
		}
		for (int i = mod(width, 8); i < width; ++i) {
			dst[i] = float_to_half(src[i]);
		}
	}
};

class LookupTableOperationAVX2 : public Operation {
	float m_lut[1L << 16];
public:
	template <class Proc>
	explicit LookupTableOperationAVX2(Proc proc)
	{
		for (uint32_t i = 0; i < UINT16_MAX + 1; ++i) {
			float x = half_to_float(i);
			m_lut[i] = proc(x);
		}
	}

	void process(float * const *ptr, int width) const override
	{
		__m128i zero = _mm_set1_epi16(0);

		for (int p = 0; p < 3; ++p) {
			for (int i = 0; i < mod(width, 8); i += 8) {
				__m256 f32 = _mm256_load_ps(&ptr[p][i]);
				__m128i half = _mm256_cvtps_ph(f32, 0);
				__m128i lo = _mm_unpacklo_epi16(half, zero);
				__m128i hi = _mm_unpackhi_epi16(half, zero);

				__m256i idx = _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
				__m256 result = _mm256_i32gather_ps(m_lut, idx, 4);

				_mm256_store_ps(&ptr[p][i], result);
			}
			for (int i = mod(width, 8); i < width; ++i) {
				uint16_t half = float_to_half(ptr[p][i]);
				ptr[p][i] = m_lut[half];
			}
		}
	}
};

class MatrixOperationAVX2 : public MatrixOperationImpl {
public:
	explicit MatrixOperationAVX2(const Matrix3x3 &m) : MatrixOperationImpl(m)
	{}

	void process(float * const *ptr, int width) const override
	{
		__m256 c00 = _mm256_set1_ps(m_matrix[0][0]);
		__m256 c01 = _mm256_set1_ps(m_matrix[0][1]);
		__m256 c02 = _mm256_set1_ps(m_matrix[0][2]);
		__m256 c10 = _mm256_set1_ps(m_matrix[1][0]);
		__m256 c11 = _mm256_set1_ps(m_matrix[1][1]);
		__m256 c12 = _mm256_set1_ps(m_matrix[1][2]);
		__m256 c20 = _mm256_set1_ps(m_matrix[2][0]);
		__m256 c21 = _mm256_set1_ps(m_matrix[2][1]);
		__m256 c22 = _mm256_set1_ps(m_matrix[2][2]);

		for (int i = 0; i < mod(width, 8); i += 8) {
			__m256 a = _mm256_load_ps(&ptr[0][i]);
			__m256 b = _mm256_load_ps(&ptr[1][i]);
			__m256 c = _mm256_load_ps(&ptr[2][i]);

			__m256 x = _mm256_mul_ps(c00, a);
			__m256 y = _mm256_mul_ps(c10, a);
			__m256 z = _mm256_mul_ps(c20, a);

			x = _mm256_fmadd_ps(c01, b, x);
			x = _mm256_fmadd_ps(c02, c, x);

			y = _mm256_fmadd_ps(c11, b, y);
			y = _mm256_fmadd_ps(c12, c, y);

			z = _mm256_fmadd_ps(c21, b, z);
			z = _mm256_fmadd_ps(c22, c, z);

			_mm256_store_ps(&ptr[0][i], x);
			_mm256_store_ps(&ptr[1][i], y);
			_mm256_store_ps(&ptr[2][i], z);
		}
		for (int i = mod(width, 8); i < width; ++i) {
			float a, b, c;
			float x, y, z;

			a = ptr[0][i];
			b = ptr[1][i];
			c = ptr[2][i];

			x = m_matrix[0][0] * a + m_matrix[0][1] * b + m_matrix[0][2] * c;
			y = m_matrix[1][0] * a + m_matrix[1][1] * b + m_matrix[1][2] * c;
			z = m_matrix[2][0] * a + m_matrix[2][1] * b + m_matrix[2][2] * c;

			ptr[0][i] = x;
			ptr[1][i] = y;
			ptr[2][i] = z;
		}
	}
};

} // namespace


PixelAdapter *create_pixel_adapter_avx2()
{
	return new PixelAdapterAVX2{};
}

Operation *create_rec709_gamma_operation_avx2()
{
	return new LookupTableOperationAVX2{ rec_709_gamma };
}

Operation *create_rec709_inverse_gamma_operation_avx2()
{
	return new LookupTableOperationAVX2{ rec_709_inverse_gamma };
}

Operation *create_matrix_operation_avx2(const Matrix3x3 &m)
{
	return new MatrixOperationAVX2{ m };
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_x86
