#ifdef ZIMG_X86

#include <algorithm>
#include <cstdint>
#include <vector>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/x86/cpuinfo_x86.h"
#include "colorspace/gamma.h"
#include "colorspace/operation.h"
#include "colorspace/operation_impl.h"
#include "operation_impl_x86.h"

#include "common/x86/avx2_util.h"

namespace zimg::colorspace {

namespace {

constexpr unsigned LUT_DEPTH = 16;


inline FORCE_INLINE void copylut_i32_ps(float *dst, const float *lut, __m128i idx)
{
#if SIZE_MAX >= UINT64_MAX
	uint64_t tmp0 = _mm_cvtsi128_si64(idx);
	uint64_t tmp1 = _mm_extract_epi64(idx, 1);

	dst[0] = lut[tmp0 & 0xFFFFFFFFU];
	dst[1] = lut[tmp0 >> 32];
	dst[2] = lut[tmp1 & 0xFFFFFFFFU];
	dst[3] = lut[tmp1 >> 32];
#else
	dst[0] = lut[_mm_cvtsi128_si32(idx)];
	dst[1] = lut[_mm_extract_epi32(idx, 1)];
	dst[2] = lut[_mm_extract_epi32(idx, 2)];
	dst[3] = lut[_mm_extract_epi32(idx, 3)];
#endif
}


inline FORCE_INLINE void matrix_filter_line_avx2_xiter(unsigned j, const float *src0, const float *src1, const float *src2,
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
	x = _mm256_fmadd_ps(c01, b, x);
	x = _mm256_fmadd_ps(c02, c, x);
	out0 = x;

	y = _mm256_mul_ps(c10, a);
	y = _mm256_fmadd_ps(c11, b, y);
	y = _mm256_fmadd_ps(c12, c, y);
	out1 = y;

	z = _mm256_mul_ps(c20, a);
	z = _mm256_fmadd_ps(c21, b, z);
	z = _mm256_fmadd_ps(c22, c, z);
	out2 = z;
}

void matrix_filter_line_avx2(const float *matrix, const float * const * RESTRICT src, float * const * RESTRICT dst, unsigned left, unsigned right)
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

#define XITER matrix_filter_line_avx2_xiter
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

void to_linear_lut_filter_line_gather(const float *RESTRICT lut, unsigned lut_depth, const float *src, float *dst, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	const int32_t lut_limit = static_cast<int32_t>(1) << lut_depth;

	const __m256 scale = _mm256_set1_ps(0.5f * lut_limit);
	const __m256 offset = _mm256_set1_ps(0.25f * lut_limit);
	const __m256i zero = _mm256_setzero_si256();
	const __m256i limit = _mm256_set1_epi32(lut_limit);

	for (unsigned j = left; j < vec_left; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_cvt_ss2si(_mm_fmadd_ss(x, _mm256_castps256_ps128(scale), _mm256_castps256_ps128(offset)));
		dst[j] = lut[std::clamp(idx, 0, lut_limit)];
	}
	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m256 x;
		__m256i xi;

		x = _mm256_load_ps(src + j);
		x = _mm256_fmadd_ps(x, scale, offset);
		xi = _mm256_cvtps_epi32(x);
		xi = _mm256_max_epi32(xi, zero);
		xi = _mm256_min_epi32(xi, limit);
		x = _mm256_i32gather_ps(lut, xi, sizeof(float));
		_mm256_store_ps(dst + j, x);
	}
	for (unsigned j = vec_right; j < right; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_cvt_ss2si(_mm_fmadd_ss(x, _mm256_castps256_ps128(scale), _mm256_castps256_ps128(offset)));
		dst[j] = lut[std::clamp(idx, 0, lut_limit)];
	}
}

void to_linear_lut_filter_line_nogather(const float *RESTRICT lut, unsigned lut_depth, const float *src, float *dst, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	const int32_t lut_limit = static_cast<int32_t>(1) << lut_depth;

	const __m128 scale = _mm_set_ps1(0.5f * lut_limit);
	const __m128 offset = _mm_set_ps1(0.25f * lut_limit);
	const __m128i zero = _mm_setzero_si128();
	const __m128i limit = _mm_set1_epi32(lut_limit);

	for (unsigned j = left; j < vec_left; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_cvt_ss2si(_mm_fmadd_ss(x, scale, offset));
		dst[j] = lut[std::clamp(idx, 0, lut_limit)];
	}
	for (unsigned j = vec_left; j < vec_right; j += 4) {
		__m128 x;
		__m128i xi;

		x = _mm_load_ps(src + j);
		x = _mm_fmadd_ps(x, scale, offset);
		xi = _mm_cvtps_epi32(x);
		xi = _mm_max_epi32(xi, zero);
		xi = _mm_min_epi32(xi, limit);

		copylut_i32_ps(dst + j, lut, xi);
	}
	for (unsigned j = vec_right; j < right; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_cvt_ss2si(_mm_fmadd_ss(x, scale, offset));
		dst[j] = lut[std::clamp(idx, 0, lut_limit)];
	}
}

void to_gamma_lut_filter_line_gather(const float *RESTRICT lut, const float *src, float *dst, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	for (unsigned j = left; j < vec_left; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_extract_epi16(_mm_cvtps_ph(x, 0), 0);
		dst[j] = lut[idx];
	}
	for (unsigned j = vec_left; j < vec_right; j += 8) {
		__m256 x;
		__m256i xi;

		x = _mm256_load_ps(src + j);
		xi = _mm256_cvtepu16_epi32(_mm256_cvtps_ph(x, 0));
		x = _mm256_i32gather_ps(lut, xi, sizeof(float));
		_mm256_store_ps(dst + j, x);
	}
	for (unsigned j = vec_right; j < right; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_extract_epi16(_mm_cvtps_ph(x, 0), 0);
		dst[j] = lut[idx];
	}
}

void to_gamma_lut_filter_line_nogather(const float *RESTRICT lut, const float *src, float *dst, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	for (unsigned j = left; j < vec_left; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_extract_epi16(_mm_cvtps_ph(x, 0), 0);
		dst[j] = lut[idx];
	}
	for (unsigned j = vec_left; j < vec_right; j += 4) {
		__m128 x;
		__m128i xi;

		x = _mm_load_ps(src + j);
		xi = _mm_cvtepu16_epi32(_mm_cvtps_ph(x, 0));
		copylut_i32_ps(dst + j, lut, xi);
	}
	for (unsigned j = vec_right; j < right; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_extract_epi16(_mm_cvtps_ph(x, 0), 0);
		dst[j] = lut[idx];
	}
}


class MatrixOperationAVX2 final : public MatrixOperationImpl {
public:
	explicit MatrixOperationAVX2(const Matrix3x3 &m) :
		MatrixOperationImpl(m)
	{
	}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		matrix_filter_line_avx2(static_cast<const float *>(&m_matrix[0][0]), src, dst, left, right);
	}
};

class ToLinearLutOperationAVX2 : public Operation {
public:
	std::vector<float> m_lut;
	unsigned m_lut_depth;

	ToLinearLutOperationAVX2(gamma_func func, unsigned lut_depth, float postscale) :
		m_lut((1UL << lut_depth) + 1),
		m_lut_depth{ lut_depth }
	{
		EnsureSinglePrecision x87;

		// Allocate an extra LUT entry so that indexing can be done by multipying by a power of 2.
		for (size_t i = 0; i < m_lut.size(); ++i) {
			float x = static_cast<float>(i) / (1 << lut_depth) * 2.0f - 0.5f;
			m_lut[i] = func(x) * postscale;
		}
	}
};

class ToLinearLutOperationAVX2Gather final : public ToLinearLutOperationAVX2 {
public:
	using ToLinearLutOperationAVX2::ToLinearLutOperationAVX2;

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		to_linear_lut_filter_line_gather(m_lut.data(), m_lut_depth, src[0], dst[0], left, right);
		to_linear_lut_filter_line_gather(m_lut.data(), m_lut_depth, src[1], dst[1], left, right);
		to_linear_lut_filter_line_gather(m_lut.data(), m_lut_depth, src[2], dst[2], left, right);
	}
};

class ToLinearLutOperationAVX2NoGather final : public ToLinearLutOperationAVX2 {
public:
	using ToLinearLutOperationAVX2::ToLinearLutOperationAVX2;

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		to_linear_lut_filter_line_nogather(m_lut.data(), m_lut_depth, src[0], dst[0], left, right);
		to_linear_lut_filter_line_nogather(m_lut.data(), m_lut_depth, src[1], dst[1], left, right);
		to_linear_lut_filter_line_nogather(m_lut.data(), m_lut_depth, src[2], dst[2], left, right);
	}
};

class ToGammaLutOperationAVX2 : public Operation {
public:
	std::vector<float> m_lut;

	ToGammaLutOperationAVX2(gamma_func func, float prescale) :
		m_lut(static_cast<uint32_t>(UINT16_MAX) + 1)
	{
		EnsureSinglePrecision x87;

		for (size_t i = 0; i <= UINT16_MAX; ++i) {
			uint16_t half = static_cast<uint16_t>(i);
			float x = _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(half)));
			m_lut[i] = func(x * prescale);
		}
	}
};

class ToGammaLutOperationAVX2Gather final : public ToGammaLutOperationAVX2 {
public:
	using ToGammaLutOperationAVX2::ToGammaLutOperationAVX2;

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		to_gamma_lut_filter_line_gather(m_lut.data(), src[0], dst[0], left, right);
		to_gamma_lut_filter_line_gather(m_lut.data(), src[1], dst[1], left, right);
		to_gamma_lut_filter_line_gather(m_lut.data(), src[2], dst[2], left, right);
	}
};

class ToGammaLutOperationAVX2NoGather final : public ToGammaLutOperationAVX2 {
public:
	using ToGammaLutOperationAVX2::ToGammaLutOperationAVX2;

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		to_gamma_lut_filter_line_nogather(m_lut.data(), src[0], dst[0], left, right);
		to_gamma_lut_filter_line_nogather(m_lut.data(), src[1], dst[1], left, right);
		to_gamma_lut_filter_line_nogather(m_lut.data(), src[2], dst[2], left, right);
	}
};

} // namespace


std::unique_ptr<Operation> create_matrix_operation_avx2(const Matrix3x3 &m)
{
	return std::make_unique<MatrixOperationAVX2>(m);
}

std::unique_ptr<Operation> create_gamma_operation_avx2(const TransferFunction &transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	X86Capabilities caps = query_x86_capabilities();

	if (cpu_has_slow_gather(caps))
		return std::make_unique<ToGammaLutOperationAVX2NoGather>(transfer.to_gamma, transfer.to_gamma_scale);
	else
		return std::make_unique<ToGammaLutOperationAVX2Gather>(transfer.to_gamma, transfer.to_gamma_scale);
}

std::unique_ptr<Operation> create_inverse_gamma_operation_avx2(const TransferFunction &transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	X86Capabilities caps = query_x86_capabilities();

	if (cpu_has_slow_gather(caps))
		return std::make_unique<ToLinearLutOperationAVX2NoGather>(transfer.to_linear, LUT_DEPTH, transfer.to_linear_scale);
	else
		return std::make_unique<ToLinearLutOperationAVX2Gather>(transfer.to_linear, LUT_DEPTH, transfer.to_linear_scale);
}

} // namespace zimg::colorspace

#endif // ZIMG_X86
