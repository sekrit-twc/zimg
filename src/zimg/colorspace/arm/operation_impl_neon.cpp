#ifdef ZIMG_ARM

#include <algorithm>
#include <cstdint>
#include <vector>
#include <arm_neon.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "colorspace/gamma.h"
#include "colorspace/operation.h"
#include "colorspace/operation_impl.h"
#include "operation_impl_arm.h"

#include "common/arm/neon_util.h"

#if defined(_M_ARM) || defined(__arm__)
  #define vcvtnq_s32_f32_(x) vcvtq_s32_f32(vaddq_f32(x, vdupq_n_f32(0.49999997f)))
#else
  #define vcvtnq_s32_f32_ vcvtnq_s32_f32
#endif

namespace zimg {
namespace colorspace {

namespace {

constexpr unsigned LUT_DEPTH = 16;

void to_linear_lut_filter_line(const float *RESTRICT lut, unsigned lut_depth, const float *src, float *dst, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const int32_t lut_limit = static_cast<int32_t>(1U) << lut_depth;

	const float32x4_t scale = vdupq_n_f32(0.5f * lut_limit);
	const float32x4_t offset = vdupq_n_f32(0.25f * lut_limit);
	const int32x4_t zero = vdupq_n_s32(0);
	const int32x4_t limit = vdupq_n_s32(lut_limit);

	for (unsigned j = left; j < vec_left; ++j) {
		float32x4_t x = vdupq_n_f32(src[j]);
		int idx = vgetq_lane_s32(vcvtnq_s32_f32_(vfmaq_f32(offset, x, scale)), 0);
		dst[j] = lut[std::min(std::max(idx, 0), lut_limit)];
	}
	for (unsigned j = vec_left; j < vec_right; j += 4) {
		float32x4_t x;
		int32x4_t xi;

		x = vld1q_f32(src + j);
		x = vfmaq_f32(offset, x, scale);
		xi = vcvtnq_s32_f32_(x);
		xi = vmaxq_s32(xi, zero);
		xi = vminq_s32(xi, limit);

		dst[j + 0] = lut[vgetq_lane_s32(xi, 0)];
		dst[j + 1] = lut[vgetq_lane_s32(xi, 1)];
		dst[j + 2] = lut[vgetq_lane_s32(xi, 2)];
		dst[j + 3] = lut[vgetq_lane_s32(xi, 3)];
	}
	for (unsigned j = vec_right; j < right; ++j) {
		float32x4_t x = vdupq_n_f32(src[j]);
		int idx = vgetq_lane_s32(vcvtnq_s32_f32_(vfmaq_f32(offset, x, scale)), 0);
		dst[j] = lut[std::min(std::max(idx, 0), lut_limit)];
	}
}

#if !defined(_MSC_VER) || defined(_M_ARM64)
void to_gamma_lut_filter_line(const float *RESTRICT lut, const float *src, float *dst, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	for (unsigned j = left; j < vec_left; ++j) {
		float32x4_t x = vdupq_n_f32(src[j]);
		int idx = vget_lane_u16(vreinterpret_u16_f16(vcvt_f16_f32(x)), 0);
		dst[j] = lut[idx];
	}
	for (unsigned j = vec_left; j < vec_right; j += 4) {
		float32x4_t x;
		uint16x4_t xi;

		x = vld1q_f32(src + j);
		xi = vreinterpret_u16_f16(vcvt_f16_f32(x));

		dst[j + 0] = lut[vget_lane_u16(xi, 0)];
		dst[j + 1] = lut[vget_lane_u16(xi, 1)];
		dst[j + 2] = lut[vget_lane_u16(xi, 2)];
		dst[j + 3] = lut[vget_lane_u16(xi, 3)];
	}
	for (unsigned j = vec_right; j < right; ++j) {
		float32x4_t x = vdupq_n_f32(src[j]);
		int idx = vget_lane_u16(vreinterpret_u16_f16(vcvt_f16_f32(x)), 0);
		dst[j] = lut[idx];
	}
}
#endif // !defined(_MSC_VER) || defined(_M_ARM64)


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


class ToLinearLutOperationNeon final : public Operation {
	std::vector<float> m_lut;
	unsigned m_lut_depth;
public:
	ToLinearLutOperationNeon(gamma_func func, unsigned lut_depth, float postscale) :
		m_lut((1UL << lut_depth) + 1),
		m_lut_depth{ lut_depth }
	{
		// Allocate an extra LUT entry so that indexing can be done by multipying by a power of 2.
		for (size_t i = 0; i < m_lut.size(); ++i) {
			float x = static_cast<float>(i) / (1 << lut_depth) * 2.0f - 0.5f;
			m_lut[i] = func(x) * postscale;
		}
	}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		to_linear_lut_filter_line(m_lut.data(), m_lut_depth, src[0], dst[0], left, right);
		to_linear_lut_filter_line(m_lut.data(), m_lut_depth, src[1], dst[1], left, right);
		to_linear_lut_filter_line(m_lut.data(), m_lut_depth, src[2], dst[2], left, right);
	}
};

#if !defined(_MSC_VER) || defined(_M_ARM64)
class ToGammaLutOperationNeon final : public Operation {
	std::vector<float> m_lut;
public:
	ToGammaLutOperationNeon(gamma_func func, float prescale) :
		m_lut(static_cast<uint32_t>(UINT16_MAX) + 1)
	{
		for (size_t i = 0; i <= UINT16_MAX; ++i) {
			uint16_t half = static_cast<uint16_t>(i);
			float x = vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(half))), 0);
			m_lut[i] = func(x * prescale);
		}
	}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		to_gamma_lut_filter_line(m_lut.data(), src[0], dst[0], left, right);
		to_gamma_lut_filter_line(m_lut.data(), src[1], dst[1], left, right);
		to_gamma_lut_filter_line(m_lut.data(), src[2], dst[2], left, right);
	}
};
#endif // !defined(_MSC_VER) || defined(_M_ARM64)

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
	return std::make_unique<MatrixOperationNeon>(m);
}

std::unique_ptr<Operation> create_gamma_operation_neon(const TransferFunction &transfer, const OperationParams &params)
{
#if !defined(_MSC_VER) || defined(_M_ARM64)
	if (!params.approximate_gamma)
		return nullptr;

	return std::make_unique<ToGammaLutOperationNeon>(transfer.to_gamma, transfer.to_gamma_scale);
#else
	return nullptr;
#endif
}

std::unique_ptr<Operation> create_inverse_gamma_operation_neon(const TransferFunction &transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	return std::make_unique<ToLinearLutOperationNeon>(transfer.to_linear, LUT_DEPTH, transfer.to_linear_scale);
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_ARM
