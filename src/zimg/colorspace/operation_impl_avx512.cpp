#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
  #undef ZIMG_X86_AVX512
#endif

#ifdef ZIMG_X86_AVX512

#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/make_unique.h"
#include "operation_impl.h"
#include "operation_impl_x86.h"

namespace zimg {
namespace colorspace {

namespace {

inline FORCE_INLINE void matrix_filter_line_avx512_xiter(unsigned j, const float * RESTRICT const * RESTRICT src, __m512 &out0, __m512 &out1, __m512 &out2,
                                                         const __m512 &c00, const __m512 &c01, const __m512 &c02,
                                                         const __m512 &c10, const __m512 &c11, const __m512 &c12,
                                                         const __m512 &c20, const __m512 &c21, const __m512 &c22)
{
	__m512 a = _mm512_load_ps(src[0] + j);
	__m512 b = _mm512_load_ps(src[1] + j);
	__m512 c = _mm512_load_ps(src[2] + j);
	__m512 x, y, z;

	x = _mm512_mul_ps(c00, a);
	x = _mm512_fmadd_ps(c01, b, x);
	x = _mm512_fmadd_ps(c02, c, x);
	out0 = x;

	y = _mm512_mul_ps(c10, a);
	y = _mm512_fmadd_ps(c11, b, y);
	y = _mm512_fmadd_ps(c12, c, y);
	out1 = y;

	z = _mm512_mul_ps(c20, a);
	z = _mm512_fmadd_ps(c21, b, z);
	z = _mm512_fmadd_ps(c22, c, z);
	out2 = z;
}

void matrix_filter_line_avx512(const float *matrix, const float * const * RESTRICT src, float * const * RESTRICT dst, unsigned left, unsigned right)
{
	const __m512 c00 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 0));
	const __m512 c01 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 1));
	const __m512 c02 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 2));
	const __m512 c10 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 3));
	const __m512 c11 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 4));
	const __m512 c12 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 5));
	const __m512 c20 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 6));
	const __m512 c21 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 7));
	const __m512 c22 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 8));
	__m512 out0, out1, out2;

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

#define XITER matrix_filter_line_avx512_xiter
#define XARGS src, out0, out1, out2, c00, c01, c02, c10, c11, c12, c20, c21, c22
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);
		__mmask16 mask = 0xFFFFU << (16 - (vec_left - left));

		_mm512_mask_store_ps(dst[0] + vec_left - 16, mask, out0);
		_mm512_mask_store_ps(dst[1] + vec_left - 16, mask, out1);
		_mm512_mask_store_ps(dst[2] + vec_left - 16, mask, out2);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);

		_mm512_store_ps(dst[0] + j, out0);
		_mm512_store_ps(dst[1] + j, out1);
		_mm512_store_ps(dst[2] + j, out2);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);
		__mmask16 mask = 0xFFFFU >> (16 - (right - vec_right));

		_mm512_mask_store_ps(dst[0] + vec_right, mask, out0);
		_mm512_mask_store_ps(dst[1] + vec_right, mask, out1);
		_mm512_mask_store_ps(dst[2] + vec_right, mask, out2);
	}
#undef XITER
#undef XARGS
}


class MatrixOperationAVX512 final : public MatrixOperationImpl {
public:
	explicit MatrixOperationAVX512(const Matrix3x3 &m) :
		MatrixOperationImpl(m)
	{
	}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		matrix_filter_line_avx512(static_cast<const float *>(&m_matrix[0][0]), src, dst, left, right);
	}
};

} // namespace


std::unique_ptr<Operation> create_matrix_operation_avx512(const Matrix3x3 &m)
{
	return ztd::make_unique<MatrixOperationAVX512>(m);
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86_AVX512
