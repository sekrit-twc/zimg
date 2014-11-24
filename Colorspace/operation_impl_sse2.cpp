#ifdef ZIMG_X86

#include <emmintrin.h>
#include "Common/align.h"
#include "matrix3.h"
#include "operation.h"
#include "operation_impl.h"
#include "operation_impl_x86.h"

namespace zimg {;
namespace colorspace {;

namespace {;

class MatrixOperationSSE2 : public MatrixOperationImpl {
public:
	explicit MatrixOperationSSE2(const Matrix3x3 &m) : MatrixOperationImpl(m)
	{}

	void process(float * const *ptr, int width) const override
	{
		__m128 c00 = _mm_set_ps1(m_matrix[0][0]);
		__m128 c01 = _mm_set_ps1(m_matrix[0][1]);
		__m128 c02 = _mm_set_ps1(m_matrix[0][2]);
		__m128 c10 = _mm_set_ps1(m_matrix[1][0]);
		__m128 c11 = _mm_set_ps1(m_matrix[1][1]);
		__m128 c12 = _mm_set_ps1(m_matrix[1][2]);
		__m128 c20 = _mm_set_ps1(m_matrix[2][0]);
		__m128 c21 = _mm_set_ps1(m_matrix[2][1]);
		__m128 c22 = _mm_set_ps1(m_matrix[2][2]);

		for (int i = 0; i < mod(width, 4); i += 4) {
			__m128 tmp0, tmp1;

			__m128 a = _mm_load_ps(&ptr[0][i]);
			__m128 b = _mm_load_ps(&ptr[1][i]);
			__m128 c = _mm_load_ps(&ptr[2][i]);

			__m128 x = _mm_mul_ps(c00, a);
			__m128 y = _mm_mul_ps(c10, a);
			__m128 z = _mm_mul_ps(c20, a);

			tmp0 = _mm_mul_ps(c01, b);
			tmp1 = _mm_mul_ps(c02, c);
			x = _mm_add_ps(x, tmp0);
			x = _mm_add_ps(x, tmp1);

			tmp0 = _mm_mul_ps(c11, b);
			tmp1 = _mm_mul_ps(c12, c);
			y = _mm_add_ps(y, tmp0);
			y = _mm_add_ps(y, tmp1);

			tmp0 = _mm_mul_ps(c21, b);
			tmp1 = _mm_mul_ps(c22, c);
			z = _mm_add_ps(z, tmp0);
			z = _mm_add_ps(z, tmp1);

			_mm_store_ps(&ptr[0][i], x);
			_mm_store_ps(&ptr[1][i], y);
			_mm_store_ps(&ptr[2][i], z);
		}
		for (int i = mod(width, 4); i < width; ++i) {
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


Operation *create_matrix_operation_sse2(const Matrix3x3 &m)
{
	return new MatrixOperationSSE2{ m };
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_x86
