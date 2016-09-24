#ifdef ZIMG_X86

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <emmintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/make_unique.h"
#include "common/zassert.h"
#include "colorspace.h"
#include "operation.h"
#include "operation_impl.h"
#include "operation_impl_x86.h"

namespace zimg {
namespace colorspace {

namespace {

const unsigned LUT_DEPTH = 15;

typedef float (*gamma_func)(float);

std::pair<float, float> get_scale_factor(TransferCharacteristics transfer, float peak_luminance)
{
	switch (transfer) {
	case TransferCharacteristics::ST_2084:
		return{ peak_luminance / ST2084_PEAK_LUMINANCE, ST2084_PEAK_LUMINANCE / peak_luminance };
	case TransferCharacteristics::ARIB_B67:
		return{ 1.0f / 12.0f, 12.0f };
	default:
		return{ 1.0f, 1.0f };
	}
}

gamma_func get_gamma_func(TransferCharacteristics transfer)
{
	switch (transfer) {
	case TransferCharacteristics::REC_709:
		return rec_709_gamma;
	case TransferCharacteristics::ST_2084:
		return st_2084_gamma;
	case TransferCharacteristics::ARIB_B67:
		return arib_b67_gamma;
	default:
		zassert_d(false, "bad transfer function");
		return nullptr;
	}
}

gamma_func get_inverse_gamma_func(TransferCharacteristics transfer)
{
	switch (transfer) {
	case TransferCharacteristics::REC_709:
		return rec_709_inverse_gamma;
	case TransferCharacteristics::ST_2084:
		return st_2084_inverse_gamma;
	case TransferCharacteristics::ARIB_B67:
		return arib_b67_inverse_gamma;
	default:
		zassert_d(false, "bad transfer function");
		return nullptr;
	}
}

void lut_filter_line(const float *RESTRICT lut, unsigned lut_depth, float prescale, const float *src, float *dst, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	const int lut_limit = 1 << lut_depth;

	const __m128 scale = _mm_set_ps1(prescale * lut_limit);
	const __m128i limit = _mm_set1_epi16(lut_limit + INT16_MIN);
	const __m128i bias_epi16 = _mm_set1_epi16(INT16_MIN);
	const __m128i bias_epi32 = _mm_set1_epi32(INT16_MIN);

	for (unsigned j = left; j < vec_left; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_cvt_ss2si(_mm_mul_ss(x, scale));
		dst[j] = lut[std::min(std::max(idx, 0), lut_limit)];
	}
	for (ptrdiff_t j = vec_left; j < vec_right; j += 4) {
		__m128 x;
		__m128i xi;

		x = _mm_load_ps(src + j);
		x = _mm_mul_ps(x, scale);
		xi = _mm_cvtps_epi32(x);
		xi = _mm_add_epi32(xi, bias_epi32);
		xi = _mm_packs_epi32(xi, xi);
		xi = _mm_min_epi16(xi, limit);
		xi = _mm_sub_epi16(xi, bias_epi16);

		dst[j + 0] = lut[_mm_extract_epi16(xi, 0)];
		dst[j + 1] = lut[_mm_extract_epi16(xi, 1)];
		dst[j + 2] = lut[_mm_extract_epi16(xi, 2)];
		dst[j + 3] = lut[_mm_extract_epi16(xi, 3)];
	}
	for (unsigned j = vec_right; j < right; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_cvt_ss2si(_mm_mul_ss(x, scale));
		dst[j] = lut[std::min(std::max(idx, 0), lut_limit)];
	}
}

class LutOperationSSE2 : public Operation {
	std::vector<float> m_lut;
	unsigned m_lut_depth;
	float m_prescale;
public:
	LutOperationSSE2(gamma_func func, unsigned lut_depth, float prescale, float postscale) :
		m_lut((1 << lut_depth) + 1),
		m_lut_depth{ lut_depth },
		m_prescale{ static_cast<float>(prescale) }
	{
		// Allocate an extra LUT entry so that indexing can be done by multipying by a power of 2.
		for (unsigned i = 0; i < m_lut.size(); ++i) {
			float x = static_cast<float>(i) / (1 << lut_depth);
			m_lut[i] = func(x) * postscale;
		}
	}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		lut_filter_line(m_lut.data(), m_lut_depth, m_prescale, src[0], dst[0], left, right);
		lut_filter_line(m_lut.data(), m_lut_depth, m_prescale, src[1], dst[1], left, right);
		lut_filter_line(m_lut.data(), m_lut_depth, m_prescale, src[2], dst[2], left, right);
	}
};

} // namespace


std::unique_ptr<Operation> create_gamma_to_linear_operation_sse2(TransferCharacteristics transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	zassert_d(!std::isnan(params.peak_luminance), "nan detected");
	std::pair<float, float> scale_factor = get_scale_factor(transfer, static_cast<float>(params.peak_luminance));
	gamma_func func = get_inverse_gamma_func(transfer);
	return ztd::make_unique<LutOperationSSE2>(func, LUT_DEPTH, 1.0f, scale_factor.second);
}

std::unique_ptr<Operation> create_linear_to_gamma_operation_sse2(TransferCharacteristics transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	zassert_d(!std::isnan(params.peak_luminance), "nan detected");
	std::pair<float, float> scale_factor = get_scale_factor(transfer, static_cast<float>(params.peak_luminance));
	gamma_func func = get_gamma_func(transfer);
	return ztd::make_unique<LutOperationSSE2>(func, LUT_DEPTH, scale_factor.first, 1.0f);
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86
