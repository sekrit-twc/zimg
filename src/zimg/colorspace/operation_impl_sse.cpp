#ifdef ZIMG_X86

#include <xmmintrin.h>
#include "common/align.h"
#include "common/ccdep.h"

#define HAVE_CPU_SSE
  #include "common/x86util.h"
#undef HAVE_CPU_SSE

#include "common/make_unique.h"
#include "operation_impl.h"
#include "operation_impl_x86.h"

namespace zimg {
namespace colorspace {

namespace {

inline FORCE_INLINE void matrix_filter_line_sse_xiter(unsigned j, const float * RESTRICT const * RESTRICT src, __m128 &out0, __m128 &out1, __m128 &out2,
                                                      const __m128 &c00, const __m128 &c01, const __m128 &c02,
                                                      const __m128 &c10, const __m128 &c11, const __m128 &c12,
                                                      const __m128 &c20, const __m128 &c21, const __m128 &c22)
{
	__m128 a = _mm_load_ps(src[0] + j);
	__m128 b = _mm_load_ps(src[1] + j);
	__m128 c = _mm_load_ps(src[2] + j);
	__m128 x, y, z;

	x = _mm_mul_ps(c00, a);
	y = _mm_mul_ps(c01, b);
	z = _mm_mul_ps(c02, c);
	x = _mm_add_ps(x, y);
	x = _mm_add_ps(x, z);
	out0 = x;

	x = _mm_mul_ps(c10, a);
	y = _mm_mul_ps(c11, b);
	z = _mm_mul_ps(c12, c);
	x = _mm_add_ps(x, y);
	x = _mm_add_ps(x, z);
	out1 = x;

	x = _mm_mul_ps(c20, a);
	y = _mm_mul_ps(c21, b);
	z = _mm_mul_ps(c22, c);
	x = _mm_add_ps(x, y);
	x = _mm_add_ps(x, z);
	out2 = x;
}

void matrix_filter_line_sse(const float *matrix, const float * const * RESTRICT src, float * const * RESTRICT dst, unsigned left, unsigned right)
{
	const __m128 c00 = _mm_set_ps1(matrix[0]);
	const __m128 c01 = _mm_set_ps1(matrix[1]);
	const __m128 c02 = _mm_set_ps1(matrix[2]);
	const __m128 c10 = _mm_set_ps1(matrix[3]);
	const __m128 c11 = _mm_set_ps1(matrix[4]);
	const __m128 c12 = _mm_set_ps1(matrix[5]);
	const __m128 c20 = _mm_set_ps1(matrix[6]);
	const __m128 c21 = _mm_set_ps1(matrix[7]);
	const __m128 c22 = _mm_set_ps1(matrix[8]);
	__m128 out0, out1, out2;

	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

#define XITER matrix_filter_line_sse_xiter
#define XARGS src, out0, out1, out2, c00, c01, c02, c10, c11, c12, c20, c21, c22
	if (left != vec_left) {
		XITER(vec_left - 4, XARGS);

		mm_store_idxhi_ps(dst[0] + vec_left - 4, out0, left % 4);
		mm_store_idxhi_ps(dst[1] + vec_left - 4, out1, left % 4);
		mm_store_idxhi_ps(dst[2] + vec_left - 4, out2, left % 4);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		XITER(j, XARGS);

		_mm_store_ps(dst[0] + j, out0);
		_mm_store_ps(dst[1] + j, out1);
		_mm_store_ps(dst[2] + j, out2);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		mm_store_idxlo_ps(dst[0] + vec_right, out0, right % 4);
		mm_store_idxlo_ps(dst[1] + vec_right, out1, right % 4);
		mm_store_idxlo_ps(dst[2] + vec_right, out2, right % 4);
	}
#undef XITER
#undef XARGS
}


class MatrixOperationSSE final : public MatrixOperationImpl {
public:
	explicit MatrixOperationSSE(const Matrix3x3 &m) :
		MatrixOperationImpl(m)
	{
	}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		matrix_filter_line_sse(static_cast<const float *>(&m_matrix[0][0]), src, dst, left, right);
	}
};

} // namespace


std::unique_ptr<Operation> create_matrix_operation_sse(const Matrix3x3 &m)
{
	return ztd::make_unique<MatrixOperationSSE>(m);
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86
