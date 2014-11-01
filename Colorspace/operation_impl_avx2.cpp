#ifdef ZIMG_X86

#include <immintrin.h>
#include "Common/align.h"
#include "Common/osdep.h"
#include "matrix3.h"
#include "operation.h"
#include "operation_impl_x86.h"

namespace zimg {;
namespace colorspace {;

namespace {;

class MatrixOperationAVX2 : public Operation {
	float m_matrix[3][3];
public:
	explicit MatrixOperationAVX2(const Matrix3x3 &matrix)
	{
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				m_matrix[i][j] = (float)matrix[i][j];
			}
		}
	}

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
			__m256 a = _mm256_load_ps(ptr[0] + i);
			__m256 b = _mm256_load_ps(ptr[1] + i);
			__m256 c = _mm256_load_ps(ptr[2] + i);

			__m256 x = _mm256_mul_ps(c00, a);
			__m256 y = _mm256_mul_ps(c10, a);
			__m256 z = _mm256_mul_ps(c20, a);

			x = _mm256_fmadd_ps(c01, b, x);
			x = _mm256_fmadd_ps(c02, c, x);

			y = _mm256_fmadd_ps(c11, b, y);
			y = _mm256_fmadd_ps(c12, c, y);

			z = _mm256_fmadd_ps(c21, b, z);
			z = _mm256_fmadd_ps(c22, c, z);

			_mm256_store_ps(ptr[0] + i, x);
			_mm256_store_ps(ptr[1] + i, y);
			_mm256_store_ps(ptr[2] + i, z);
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


Operation *create_matrix_operation_avx2(const Matrix3x3 &m)
{
	return new MatrixOperationAVX2{ m };
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_x86
