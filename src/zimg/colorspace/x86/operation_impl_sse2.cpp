#ifdef ZIMG_X86

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <emmintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "colorspace/gamma.h"
#include "colorspace/operation.h"
#include "colorspace/operation_impl.h"
#include "operation_impl_x86.h"

#include "common/x86/sse2_util.h"

namespace zimg {
namespace colorspace {

namespace {

constexpr unsigned LUT_DEPTH = 16;

template <class T, class U>
T bit_cast(const U &x) noexcept
{
	static_assert(sizeof(T) == sizeof(U), "object sizes must match");
	static_assert(std::is_pod<T>::value && std::is_pod<U>::value, "object types must be POD");

	T ret;
	std::copy_n(reinterpret_cast<const char *>(&x), sizeof(x), reinterpret_cast<char *>(&ret));
	return ret;
}

void to_linear_lut_filter_line(const float *RESTRICT lut, unsigned lut_depth, const float *src, float *dst, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	const int32_t lut_limit = static_cast<int32_t>(1) << lut_depth;

	const __m128 scale = _mm_set_ps1(0.5f * lut_limit);
	const __m128 offset = _mm_set_ps1(0.25f * lut_limit);
	const __m128i limit = _mm_set1_epi16(std::min(lut_limit + INT16_MIN, static_cast<int32_t>(INT16_MAX)));
	const __m128i bias_epi16 = _mm_set1_epi16(INT16_MIN);
	const __m128i bias_epi32 = _mm_set1_epi32(INT16_MIN);

	for (unsigned j = left; j < vec_left; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_cvt_ss2si(_mm_add_ss(_mm_mul_ss(x, scale), offset));
		dst[j] = lut[std::min(std::max(idx, 0), lut_limit)];
	}
	for (ptrdiff_t j = vec_left; j < static_cast<ptrdiff_t>(vec_right); j += 4) {
		__m128 x;
		__m128i xi;

		x = _mm_load_ps(src + j);
		x = _mm_mul_ps(x, scale);
		x = _mm_add_ps(x, offset);
		xi = _mm_cvtps_epi32(x);
		xi = _mm_add_epi32(xi, bias_epi32);
		xi = _mm_packs_epi32(xi, xi);
		xi = _mm_min_epi16(xi, limit);
		xi = _mm_sub_epi16(xi, bias_epi16);

#if SIZE_MAX >= UINT64_MAX
		uint64_t tmp = _mm_cvtsi128_si64(xi);
		dst[j + 0] = lut[tmp & 0xFFFFU];
		dst[j + 1] = lut[(tmp >> 16) & 0xFFFFU];
		dst[j + 2] = lut[(tmp >> 32) & 0xFFFFU];
		dst[j + 3] = lut[tmp >> 48];
#else
		uint32_t tmp0 = _mm_cvtsi128_si32(xi);
		uint32_t tmp1 = _mm_cvtsi128_si32(_mm_shuffle_epi32(xi, _MM_SHUFFLE(3, 2, 0, 1)));
		dst[j + 0] = lut[tmp0 & 0xFFFFU];
		dst[j + 1] = lut[tmp0 >> 16];
		dst[j + 2] = lut[tmp1 & 0xFFFFU];
		dst[j + 3] = lut[tmp1 >> 16];
#endif
	}
	for (unsigned j = vec_right; j < right; ++j) {
		__m128 x = _mm_load_ss(src + j);
		int idx = _mm_cvt_ss2si(_mm_add_ss(_mm_mul_ss(x, scale), offset));
		dst[j] = lut[std::min(std::max(idx, 0), lut_limit)];
	}
}

void to_gamma_lut_filter_line(const float *RESTRICT lut, const float *src, float *dst, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	for (unsigned j = left; j < vec_left; ++j) {
		__m128i x = _mm_castps_si128(_mm_load_ss(src + j));
		__m128i msb = _mm_srli_epi32(x, 16);
		__m128i lsb = _mm_and_si128(_mm_srli_epi32(x, 15), _mm_set1_epi32(1));
		x = mm_packus_epi32(msb, lsb);
		x = _mm_adds_epi16(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2)));

		dst[j] = lut[_mm_cvtsi128_si32(x)];
	}
	for (unsigned j = vec_left; j < vec_right; j += 4) {
		__m128i x = _mm_castps_si128(_mm_load_ps(src + j));
		__m128i msb = _mm_srli_epi32(x, 16);
		__m128i lsb = _mm_and_si128(_mm_srli_epi32(x, 15), _mm_set1_epi32(1));
		x = mm_packus_epi32(msb, lsb);
		x = _mm_adds_epi16(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2)));

#if SIZE_MAX >= UINT64_MAX
		uint64_t tmp = _mm_cvtsi128_si64(x);
		dst[j + 0] = lut[tmp & 0xFFFFU];
		dst[j + 1] = lut[(tmp >> 16) & 0xFFFFU];
		dst[j + 2] = lut[(tmp >> 32) & 0xFFFFU];
		dst[j + 3] = lut[tmp >> 48];
#else
		uint32_t tmp0 = _mm_cvtsi128_si32(x);
		uint32_t tmp1 = _mm_cvtsi128_si32(_mm_shuffle_epi32(x, _MM_SHUFFLE(3, 2, 0, 1)));
		dst[j + 0] = lut[tmp0 & 0xFFFFU];
		dst[j + 1] = lut[tmp0 >> 16];
		dst[j + 2] = lut[tmp1 & 0xFFFFU];
		dst[j + 3] = lut[tmp1 >> 16];
#endif
	}
	for (unsigned j = vec_right; j < right; ++j) {
		__m128i x = _mm_castps_si128(_mm_load_ss(src + j));
		__m128i msb = _mm_srli_epi32(x, 16);
		__m128i lsb = _mm_and_si128(_mm_srli_epi32(x, 15), _mm_set1_epi32(1));
		x = mm_packus_epi32(msb, lsb);
		x = _mm_adds_epi16(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2)));

		dst[j] = lut[_mm_cvtsi128_si32(x)];
	}
}


class ToLinearLutOperationSSE2 final : public Operation {
	std::vector<float> m_lut;
	unsigned m_lut_depth;
public:
	ToLinearLutOperationSSE2(gamma_func func, unsigned lut_depth, float postscale) :
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

class ToGammaLutOperationSSE2 final : public Operation {
	std::vector<float> m_lut;
public:
	ToGammaLutOperationSSE2(gamma_func func, float prescale) :
		m_lut(static_cast<uint32_t>(UINT16_MAX) + 1)
	{
		EnsureSinglePrecision x87;

		for (size_t i = 0; i <= UINT16_MAX; ++i) {
			float x = bit_cast<float>(static_cast<uint32_t>(i << 16));
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


std::unique_ptr<Operation> create_gamma_operation_sse2(const TransferFunction &transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	return std::make_unique<ToGammaLutOperationSSE2>(transfer.to_gamma, transfer.to_gamma_scale);
}

std::unique_ptr<Operation> create_inverse_gamma_operation_sse2(const TransferFunction &transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	return std::make_unique<ToLinearLutOperationSSE2>(transfer.to_linear, LUT_DEPTH, transfer.to_linear_scale);
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86
