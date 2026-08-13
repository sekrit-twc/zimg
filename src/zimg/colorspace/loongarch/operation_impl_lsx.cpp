#ifdef ZIMG_LOONGARCH

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <lsxintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "colorspace/gamma.h"
#include "colorspace/operation.h"
#include "colorspace/operation_impl.h"
#include "depth/quantize.h"
#include "operation_impl_loongarch.h"

#include "common/loongarch/lsx_util.h"

namespace zimg::colorspace {

namespace {

constexpr unsigned LUT_DEPTH = 16;

inline FORCE_INLINE float half_to_float_rne(uint16_t h)
{
	uint32_t sign = (h & 0x8000u) << 16;
	uint32_t exp = (h >> 10) & 0x1Fu;
	uint32_t mant = h & 0x3FFu;
	uint32_t bits;

	if (exp == 0) {
		if (mant == 0) {
			bits = sign;
		} else {
			// Denormalized half -> normalized float
			while (!(mant & 0x400u)) {
				mant <<= 1;
				exp--;
			}
			bits = sign | ((exp + 127u - 14u) << 23) | ((mant & 0x3FFu) << 13);
		}
	} else if (exp == 31) {
		bits = sign | 0x7F800000u | (mant << 13);
	} else {
		bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
	}

	float v;
	memcpy(&v, &bits, 4);
	return v;
}

inline FORCE_INLINE uint16_t float_to_half_rne(float v)
{
	return zimg::depth::float_to_half(v);
}

inline FORCE_INLINE void float_to_half_rne_4(const float *src, uint16_t *dst)
{
	dst[0] = float_to_half_rne(src[0]);
	dst[1] = float_to_half_rne(src[1]);
	dst[2] = float_to_half_rne(src[2]);
	dst[3] = float_to_half_rne(src[3]);
}

inline FORCE_INLINE void matrix_filter_line_lsx_xiter(unsigned j, const float *src0, const float *src1, const float *src2,
                                                      const __m128 &c00, const __m128 &c01, const __m128 &c02,
                                                      const __m128 &c10, const __m128 &c11, const __m128 &c12,
                                                      const __m128 &c20, const __m128 &c21, const __m128 &c22,
                                                      __m128 &out0, __m128 &out1, __m128 &out2)
{
	__m128 a = (__m128)__lsx_vld(src0 + j, 0);
	__m128 b = (__m128)__lsx_vld(src1 + j, 0);
	__m128 c = (__m128)__lsx_vld(src2 + j, 0);

	out0 = __lsx_vfmadd_s(c01, b, __lsx_vfmul_s(c00, a));
	out0 = __lsx_vfmadd_s(c02, c, out0);

	out1 = __lsx_vfmadd_s(c11, b, __lsx_vfmul_s(c10, a));
	out1 = __lsx_vfmadd_s(c12, c, out1);

	out2 = __lsx_vfmadd_s(c21, b, __lsx_vfmul_s(c20, a));
	out2 = __lsx_vfmadd_s(c22, c, out2);
}

void matrix_filter_line_lsx(const float *matrix, const float * const * RESTRICT src, float * const * RESTRICT dst, unsigned left, unsigned right)
{
	const float *src0 = src[0];
	const float *src1 = src[1];
	const float *src2 = src[2];
	float *dst0 = dst[0];
	float *dst1 = dst[1];
	float *dst2 = dst[2];

	const __m128 c00 = (__m128)__lsx_vldrepl_w(&matrix[0], 0);
	const __m128 c01 = (__m128)__lsx_vldrepl_w(&matrix[1], 0);
	const __m128 c02 = (__m128)__lsx_vldrepl_w(&matrix[2], 0);
	const __m128 c10 = (__m128)__lsx_vldrepl_w(&matrix[3], 0);
	const __m128 c11 = (__m128)__lsx_vldrepl_w(&matrix[4], 0);
	const __m128 c12 = (__m128)__lsx_vldrepl_w(&matrix[5], 0);
	const __m128 c20 = (__m128)__lsx_vldrepl_w(&matrix[6], 0);
	const __m128 c21 = (__m128)__lsx_vldrepl_w(&matrix[7], 0);
	const __m128 c22 = (__m128)__lsx_vldrepl_w(&matrix[8], 0);
	__m128 out0, out1, out2;

	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

#define XITER matrix_filter_line_lsx_xiter
#define XARGS src0, src1, src2, c00, c01, c02, c10, c11, c12, c20, c21, c22, out0, out1, out2
	if (left != vec_left) {
		XITER(vec_left - 4, XARGS);

		lsx_store_idxhi_f32(dst0 + vec_left - 4, out0, left % 4);
		lsx_store_idxhi_f32(dst1 + vec_left - 4, out1, left % 4);
		lsx_store_idxhi_f32(dst2 + vec_left - 4, out2, left % 4);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		XITER(j, XARGS);

		__lsx_vst((__m128i)out0, dst0 + j, 0);
		__lsx_vst((__m128i)out1, dst1 + j, 0);
		__lsx_vst((__m128i)out2, dst2 + j, 0);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);

		lsx_store_idxlo_f32(dst0 + vec_right, out0, right % 4);
		lsx_store_idxlo_f32(dst1 + vec_right, out1, right % 4);
		lsx_store_idxlo_f32(dst2 + vec_right, out2, right % 4);
	}
#undef XITER
#undef XARGS
}

void to_linear_lut_filter_line(const float *RESTRICT lut, unsigned lut_depth, const float *src, float *dst, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	const int32_t lut_limit = static_cast<int32_t>(1U) << lut_depth;

	float scale_val = 0.5f * lut_limit;
	float offset_val = 0.25f * lut_limit;
	const __m128 scale = (__m128)__lsx_vldrepl_w(&scale_val, 0);
	const __m128 offset = (__m128)__lsx_vldrepl_w(&offset_val, 0);

	for (unsigned j = left; j < vec_left; ++j) {
		float val = src[j];
		__m128 x = (__m128)__lsx_vldrepl_w(&val, 0);
		__m128 tmp = __lsx_vfmadd_s(x, scale, offset);
		__m128i xi = __lsx_vftintrne_w_s(tmp);
		int idx = __lsx_vpickve2gr_w(xi, 0);
		dst[j] = lut[std::clamp(idx, 0, lut_limit)];
	}
	for (unsigned j = vec_left; j < vec_right; j += 4) {
		__m128 x;
		__m128i xi;

		x = (__m128)__lsx_vld(src + j, 0);
		x = __lsx_vfmadd_s(x, scale, offset);
		xi = __lsx_vftintrne_w_s(x);

		dst[j + 0] = lut[std::clamp(__lsx_vpickve2gr_w(xi, 0), 0, lut_limit)];
		dst[j + 1] = lut[std::clamp(__lsx_vpickve2gr_w(xi, 1), 0, lut_limit)];
		dst[j + 2] = lut[std::clamp(__lsx_vpickve2gr_w(xi, 2), 0, lut_limit)];
		dst[j + 3] = lut[std::clamp(__lsx_vpickve2gr_w(xi, 3), 0, lut_limit)];
	}
	for (unsigned j = vec_right; j < right; ++j) {
		float val = src[j];
		__m128 x = (__m128)__lsx_vldrepl_w(&val, 0);
		__m128 tmp = __lsx_vfmadd_s(x, scale, offset);
		__m128i xi = __lsx_vftintrne_w_s(tmp);
		int idx = __lsx_vpickve2gr_w(xi, 0);
		dst[j] = lut[std::clamp(idx, 0, lut_limit)];
	}
}

void to_gamma_lut_filter_line(const float *RESTRICT lut, const float *src, float *dst, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	for (unsigned j = left; j < vec_left; ++j) {
		uint16_t h = float_to_half_rne(src[j]);
		dst[j] = lut[h];
	}
	for (unsigned j = vec_left; j < vec_right; j += 4) {
		uint16_t h[4];
		float_to_half_rne_4(src + j, h);

		dst[j + 0] = lut[h[0]];
		dst[j + 1] = lut[h[1]];
		dst[j + 2] = lut[h[2]];
		dst[j + 3] = lut[h[3]];
	}
	for (unsigned j = vec_right; j < right; ++j) {
		uint16_t h = float_to_half_rne(src[j]);
		dst[j] = lut[h];
	}
}


class ToLinearLutOperationLSX final : public Operation {
	std::vector<float> m_lut;
	unsigned m_lut_depth;
public:
	ToLinearLutOperationLSX(gamma_func func, unsigned lut_depth, float postscale) :
		m_lut((1UL << lut_depth) + 1),
		m_lut_depth{ lut_depth }
	{
		for (size_t i = 0; i < m_lut.size(); ++i) {
			float x = static_cast<float>(i) / (1 << lut_depth) * 2.0f - 0.5f;
			m_lut[i] = func(x) * postscale;
		}
	}

	unsigned alignment_mask() const noexcept override { return 0x3; }

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const noexcept override
	{
		to_linear_lut_filter_line(m_lut.data(), m_lut_depth, src[0], dst[0], left, right);
		to_linear_lut_filter_line(m_lut.data(), m_lut_depth, src[1], dst[1], left, right);
		to_linear_lut_filter_line(m_lut.data(), m_lut_depth, src[2], dst[2], left, right);
	}
};

class ToGammaLutOperationLSX final : public Operation {
	std::vector<float> m_lut;
public:
	ToGammaLutOperationLSX(gamma_func func, float prescale) :
		m_lut(static_cast<uint32_t>(UINT16_MAX) + 1)
	{
		for (size_t i = 0; i <= UINT16_MAX; ++i) {
			uint16_t half = static_cast<uint16_t>(i);
			float x = half_to_float_rne(half);
			m_lut[i] = func(x * prescale);
		}
	}

	unsigned alignment_mask() const noexcept override { return 0x3; }

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const noexcept override
	{
		to_gamma_lut_filter_line(m_lut.data(), src[0], dst[0], left, right);
		to_gamma_lut_filter_line(m_lut.data(), src[1], dst[1], left, right);
		to_gamma_lut_filter_line(m_lut.data(), src[2], dst[2], left, right);
	}
};

class MatrixOperationLSX final : public MatrixOperationImpl {
public:
	explicit MatrixOperationLSX(const Matrix3x3 &m) :
		MatrixOperationImpl(m)
	{}

	unsigned alignment_mask() const noexcept override { return 0x3; }

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const noexcept override
	{
		matrix_filter_line_lsx(static_cast<const float *>(&m_matrix[0][0]), src, dst, left, right);
	}
};

} // namespace


std::unique_ptr<Operation> create_matrix_operation_lsx(const Matrix3x3 &m)
{
	return std::make_unique<MatrixOperationLSX>(m);
}

std::unique_ptr<Operation> create_gamma_operation_lsx(const TransferFunction &transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	return std::make_unique<ToGammaLutOperationLSX>(transfer.to_gamma, transfer.to_gamma_scale);
}

std::unique_ptr<Operation> create_inverse_gamma_operation_lsx(const TransferFunction &transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	return std::make_unique<ToLinearLutOperationLSX>(transfer.to_linear, LUT_DEPTH, transfer.to_linear_scale);
}

} // namespace zimg::colorspace

#endif // ZIMG_LOONGARCH
