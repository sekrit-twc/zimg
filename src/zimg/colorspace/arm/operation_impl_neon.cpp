#ifdef ZIMG_ARM

#include <cstdint>
#include <arm_neon.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/make_unique.h"
#include "colorspace/operation_impl.h"
#include "operation_impl_arm.h"

#include "common/arm/neon_util.h"

namespace zimg {
namespace colorspace {

namespace {

inline FORCE_INLINE void matrix_filter_line_neon_xiter(unsigned j, const float *src0, const float *src1, const float *src2,
	                                                   const float32x4_t &c00, const float32x4_t &c01, const float32x4_t &c02,
	                                                   const float32x4_t &c10, const float32x4_t &c11, const float32x4_t &c12,
	                                                   const float32x4_t &c20, const float32x4_t &c21, const float32x4_t &c22,
                                                       float32x4_t &out0, float32x4_t &out1, float32x4_t &out2)
{
	float32x4_t a = vld1q_f32(src0 + j);
	float32x4_t b = vld1q_f32(src1 + j);
	float32x4_t c = vld1q_f32(src2 + j);
	float32x4_t x, y, z;

	x = vmulq_f32(c00, a);
	x = vfmaq_f32(x, c01, b);
	x = vfmaq_f32(x, c02, c);
	out0 = x;

	y = vmulq_f32(c10, a);
	y = vfmaq_f32(y, c11, b);
	y = vfmaq_f32(y, c12, c);
	out1 = y;

	z = vmulq_f32(c20, a);
	z = vfmaq_f32(z, c21, b);
	z = vfmaq_f32(z, c22, c);
	out2 = z;
}

void matrix_filter_line_neon(const float *matrix, const float * const * RESTRICT src, float * const * RESTRICT dst, unsigned left, unsigned right)
{
	const float *src0 = src[0];
	const float *src1 = src[1];
	const float *src2 = src[2];
	float *dst0 = dst[0];
	float *dst1 = dst[1];
	float *dst2 = dst[2];

	const float32x4_t c00 = vdupq_n_f32(matrix[0]);
	const float32x4_t c01 = vdupq_n_f32(matrix[1]);
	const float32x4_t c02 = vdupq_n_f32(matrix[2]);
	const float32x4_t c10 = vdupq_n_f32(matrix[3]);
	const float32x4_t c11 = vdupq_n_f32(matrix[4]);
	const float32x4_t c12 = vdupq_n_f32(matrix[5]);
	const float32x4_t c20 = vdupq_n_f32(matrix[6]);
	const float32x4_t c21 = vdupq_n_f32(matrix[7]);
	const float32x4_t c22 = vdupq_n_f32(matrix[8]);
	float32x4_t out0, out1, out2;

	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

#define XITER matrix_filter_line_neon_xiter
#define XARGS src0, src1, src2, c00, c01, c02, c10, c11, c12, c20, c21, c22, out0, out1, out2
	if (left != vec_left) {
		XITER(vec_left - 4, XARGS);

		neon_store_idxhi_f32(dst0 + vec_left - 4, out0, left % 4);
		neon_store_idxhi_f32(dst1 + vec_left - 4, out1, left % 4);
		neon_store_idxhi_f32(dst2 + vec_left - 4, out2, left % 4);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		XITER(j, XARGS);

		vst1q_f32(dst0 + j, out0);
		vst1q_f32(dst1 + j, out1);
		vst1q_f32(dst2 + j, out2);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		neon_store_idxlo_f32(dst0 + vec_right, out0, right % 4);
		neon_store_idxlo_f32(dst1 + vec_right, out1, right % 4);
		neon_store_idxlo_f32(dst2 + vec_right, out2, right % 4);
	}
#undef XITER
#undef XARGS
}


class MatrixOperationNeon final : public MatrixOperationImpl {
public:
	explicit MatrixOperationNeon(const Matrix3x3 &m) :
		MatrixOperationImpl(m)
	{}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		matrix_filter_line_neon(static_cast<const float *>(&m_matrix[0][0]), src, dst, left, right);
	}
};

} // namespace


std::unique_ptr<Operation> create_matrix_operation_neon(const Matrix3x3 &m)
{
	return ztd::make_unique<MatrixOperationNeon>(m);
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_ARM
