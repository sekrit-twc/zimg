#ifdef ZIMG_X86

#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "colorspace/operation_impl.h"
#include "operation_impl_x86.h"

#include "common/x86/avx_util.h"

namespace zimg {
namespace colorspace {

namespace {

inline FORCE_INLINE void matrix_filter_line_avx_xiter(unsigned j, const float *src0, const float *src1, const float *src2,
                                                      const __m256 &c00, const __m256 &c01, const __m256 &c02,
                                                      const __m256 &c10, const __m256 &c11, const __m256 &c12,
                                                      const __m256 &c20, const __m256 &c21, const __m256 &c22,
                                                      __m256 &out0, __m256 &out1, __m256 &out2)
{
	__m256 a = _mm256_load_ps(src0 + j);
	__m256 b = _mm256_load_ps(src1 + j);
	__m256 c = _mm256_load_ps(src2 + j);
	__m256 x, y, z;

	x = _mm256_mul_ps(c00, a);
	y = _mm256_mul_ps(c01, b);
	z = _mm256_mul_ps(c02, c);
	x = _mm256_add_ps(x, y);
	x = _mm256_add_ps(x, z);
	out0 = x;

	x = _mm256_mul_ps(c10, a);
	y = _mm256_mul_ps(c11, b);
	z = _mm256_mul_ps(c12, c);
	x = _mm256_add_ps(x, y);
	x = _mm256_add_ps(x, z);
	out1 = x;

	x = _mm256_mul_ps(c20, a);
	y = _mm256_mul_ps(c21, b);
	z = _mm256_mul_ps(c22, c);
	x = _mm256_add_ps(x, y);
	x = _mm256_add_ps(x, z);
	out2 = x;
}

void matrix_filter_line_avx(const float *matrix, const float * const * RESTRICT src, float * const * RESTRICT dst, unsigned left, unsigned right)
{
	const float *src0 = src[0];
	const float *src1 = src[1];
	const float *src2 = src[2];
	float *dst0 = dst[0];
	float *dst1 = dst[1];
	float *dst2 = dst[2];

	const __m256 c00 = _mm256_broadcast_ss(matrix + 0);
	const __m256 c01 = _mm256_broadcast_ss(matrix + 1);
	const __m256 c02 = _mm256_broadcast_ss(matrix + 2);
	const __m256 c10 = _mm256_broadcast_ss(matrix + 3);
	const __m256 c11 = _mm256_broadcast_ss(matrix + 4);
	const __m256 c12 = _mm256_broadcast_ss(matrix + 5);
	const __m256 c20 = _mm256_broadcast_ss(matrix + 6);
	const __m256 c21 = _mm256_broadcast_ss(matrix + 7);
	const __m256 c22 = _mm256_broadcast_ss(matrix + 8);
	__m256 out0, out1, out2;

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

#define XITER matrix_filter_line_avx_xiter
#define XARGS src0, src1, src2, c00, c01, c02, c10, c11, c12, c20, c21, c22, out0, out1, out2
	if (left != vec_left) {
		XITER(vec_left - 8, XARGS);

		mm256_store_idxhi_ps(dst0 + vec_left - 8, out0, left % 8);
		mm256_store_idxhi_ps(dst1 + vec_left - 8, out1, left % 8);
		mm256_store_idxhi_ps(dst2 + vec_left - 8, out2, left % 8);
	}

	for (unsigned j = vec_left; j < vec_right; j += 8) {
		XITER(j, XARGS);

		_mm256_store_ps(dst0 + j, out0);
		_mm256_store_ps(dst1 + j, out1);
		_mm256_store_ps(dst2 + j, out2);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		mm256_store_idxlo_ps(dst0 + vec_right, out0, right % 8);
		mm256_store_idxlo_ps(dst1 + vec_right, out1, right % 8);
		mm256_store_idxlo_ps(dst2 + vec_right, out2, right % 8);
	}
#undef XITER
#undef XARGS
}


class MatrixOperationAVX final : public MatrixOperationImpl {
public:
	explicit MatrixOperationAVX(const Matrix3x3 &m) :
		MatrixOperationImpl(m)
	{
	}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		matrix_filter_line_avx(static_cast<const float *>(&m_matrix[0][0]), src, dst, left, right);
	}
};

} // namespace


std::unique_ptr<Operation> create_matrix_operation_avx(const Matrix3x3 &m)
{
	return std::make_unique<MatrixOperationAVX>(m);
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86
