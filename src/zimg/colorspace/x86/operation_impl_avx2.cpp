#ifdef ZIMG_X86

#include <algorithm>
#include <cstdint>
#include <vector>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/make_unique.h"
#include "colorspace/gamma.h"
#include "colorspace/operation.h"
#include "operation_impl_x86.h"

namespace zimg {
namespace colorspace {

namespace {

constexpr unsigned LUT_DEPTH = 16;

void to_linear_lut_filter_line(const float *RESTRICT lut, unsigned lut_depth, const float *src, float *dst, unsigned left, unsigned right)
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
		dst[j] = lut[std::min(std::max(idx, 0), lut_limit)];
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
		dst[j] = lut[std::min(std::max(idx, 0), lut_limit)];
	}
}

void to_gamma_lut_filter_line(const float *RESTRICT lut, const float *src, float *dst, unsigned left, unsigned right)
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


class ToLinearLutOperationAVX2 final : public Operation {
	std::vector<float> m_lut;
	unsigned m_lut_depth;
public:
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

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		to_linear_lut_filter_line(m_lut.data(), m_lut_depth, src[0], dst[0], left, right);
		to_linear_lut_filter_line(m_lut.data(), m_lut_depth, src[1], dst[1], left, right);
		to_linear_lut_filter_line(m_lut.data(), m_lut_depth, src[2], dst[2], left, right);
	}
};

class ToGammaLutOperationAVX2 final : public Operation {
	std::vector<float> m_lut;
public:
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

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		to_gamma_lut_filter_line(m_lut.data(), src[0], dst[0], left, right);
		to_gamma_lut_filter_line(m_lut.data(), src[1], dst[1], left, right);
		to_gamma_lut_filter_line(m_lut.data(), src[2], dst[2], left, right);
	}
};

} // namespace


std::unique_ptr<Operation> create_gamma_operation_avx2(const TransferFunction &transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	return ztd::make_unique<ToGammaLutOperationAVX2>(transfer.to_gamma, transfer.to_gamma_scale);
}

std::unique_ptr<Operation> create_inverse_gamma_operation_avx2(const TransferFunction &transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	return ztd::make_unique<ToLinearLutOperationAVX2>(transfer.to_linear, LUT_DEPTH, transfer.to_linear_scale);
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86
