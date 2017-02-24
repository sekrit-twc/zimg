#include <algorithm>
#include <cfloat>
#include <cmath>
#include "common/except.h"
#include "common/libm_wrapper.h"
#include "common/zassert.h"
#include "colorspace.h"
#include "gamma.h"

namespace zimg {
namespace colorspace {

namespace {

constexpr float REC709_ALPHA = 1.09929682680944f;
constexpr float REC709_BETA = 0.018053968510807f;

constexpr float SRGB_ALPHA = 1.055f;
constexpr float SRGB_BETA = 0.0031308f;

constexpr float ST2084_M1 = 0.1593017578125f;
constexpr float ST2084_M2 = 78.84375f;
constexpr float ST2084_C1 = 0.8359375f;
constexpr float ST2084_C2 = 18.8515625f;
constexpr float ST2084_C3 = 18.6875f;

constexpr float ARIB_B67_A = 0.17883277f;
constexpr float ARIB_B67_B = 0.28466892f;
constexpr float ARIB_B67_C = 0.55991073f;


float ootf_1_2(float x) noexcept
{
	return x < 0.0f ? x : zimg_x_powf(x, 1.2f);
}

float inverse_ootf_1_2(float x) noexcept
{
	return x < 0.0f ? x : zimg_x_powf(x, 1.0f / 1.2f);
}

} // namespace


float rec_709_oetf(float x) noexcept
{
	if (x < REC709_BETA)
		x = x * 4.5f;
	else
		x = REC709_ALPHA * zimg_x_powf(x, 0.45f) - (REC709_ALPHA - 1.0f);

	return x;
}

float rec_709_inverse_oetf(float x) noexcept
{
	if (x < 4.5f * REC709_BETA)
		x = x / 4.5f;
	else
		x = zimg_x_powf((x + (REC709_ALPHA - 1.0f)) / REC709_ALPHA, 1.0f / 0.45f);

	return x;
}

float arib_b67_oetf(float x) noexcept
{
	// Prevent negative pixels from yielding NAN.
	x = std::max(x, 0.0f);

	if (x <= (1.0f / 12.0f))
		x = zimg_x_sqrtf(3.0f * x);
	else
		x = ARIB_B67_A * zimg_x_logf(12.0f * x - ARIB_B67_B) + ARIB_B67_C;

	return x;
}

float arib_b67_inverse_oetf(float x) noexcept
{
	// Prevent negative pixels expanding into positive values.
	x = std::max(x, 0.0f);

	if (x <= 0.5f)
		x = (x * x) * (1.0f / 3.0f);
	else
		x = (zimg_x_expf((x - ARIB_B67_C) / ARIB_B67_A) + ARIB_B67_B) / 12.0f;

	return x;
}

// Ignore the BT.1886 provisions for limited contrast and assume an ideal CRT.
float rec_1886_eotf(float x) noexcept
{
	return x < 0.0f ? 0.0f : zimg_x_powf(x, 2.4f);
}

float rec_1886_inverse_eotf(float x) noexcept
{
	return x < 0.0f ? 0.0f : zimg_x_powf(x, 1.0f / 2.4f);
}

float srgb_eotf(float x) noexcept
{
	if (x < 12.92f * SRGB_BETA)
		x = x / 12.92f;
	else
		x = zimg_x_powf((x + (SRGB_ALPHA - 1.0f)) / SRGB_ALPHA, 2.4f);

	return x;
}

float srgb_inverse_eotf(float x) noexcept
{
	if (x < SRGB_BETA)
		x = x * 12.92f;
	else
		x = SRGB_ALPHA * zimg_x_powf(x, 1.0f / 2.4f) - (SRGB_ALPHA - 1.0f);

	return x;
}

float st_2084_eotf(float x) noexcept
{
	// Filter negative values to avoid NAN.
	if (x > 0.0f) {
		float xpow = zimg_x_powf(x, 1.0f / ST2084_M2);
		float num = std::max(xpow - ST2084_C1, 0.0f);
		float den = std::max(ST2084_C2 - ST2084_C3 * xpow, FLT_MIN);
		x = zimg_x_powf(num / den, 1.0f / ST2084_M1);
	} else {
		x = 0.0f;
	}

	return x;
}

float st_2084_inverse_eotf(float x) noexcept
{
	// Filter negative values to avoid NAN, and also special-case 0 so that (f(g(0)) == 0).
	if (x > 0.0f) {
		float xpow = zimg_x_powf(x, ST2084_M1);
#if 0
		// Original formulation from SMPTE ST 2084:2014 publication.
		float num = ST2084_C1 + ST2084_C2 * xpow;
		float den = 1.0f + ST2084_C3 * xpow;
		x = zimg_x_powf(num / den, ST2084_M2);
#else
		// More stable arrangement that avoids some cancellation error.
		float num = (ST2084_C1 - 1.0f) + (ST2084_C2 - ST2084_C3) * xpow;
		float den = 1.0f + ST2084_C3 * xpow;
		x = zimg_x_powf(1.0f + num / den, ST2084_M2);
#endif
	} else {
		x = 0.0f;
	}

	return x;
}

// Applies a per-channel correction instead of the iterative method specified in Rec.2100.
float arib_b67_eotf(float x) noexcept
{
	return ootf_1_2(arib_b67_inverse_oetf(x));
}

float arib_b67_inverse_eotf(float x) noexcept
{
	return arib_b67_oetf(inverse_ootf_1_2(x));
}

// Apples a 1.2 pure-power OOTF instead of the chained Rec.709/Rec.1886 method described in Rec.2100.
float st_2084_oetf(float x) noexcept
{
	return st_2084_inverse_eotf(ootf_1_2(x));
}

float st_2084_inverse_oetf(float x) noexcept
{
	return inverse_ootf_1_2(st_2084_eotf(x));
}


TransferFunction select_transfer_function(TransferCharacteristics transfer, double peak_luminance, bool scene_referred)
{
	TransferFunction func{};

	func.to_linear_scale = 1.0f;
	func.to_gamma_scale = 1.0f;

	switch (transfer) {
	case TransferCharacteristics::REC_709:
		func.to_linear = scene_referred ? rec_709_inverse_oetf : rec_1886_eotf;
		func.to_gamma = scene_referred ? rec_709_oetf : rec_1886_inverse_eotf;
		break;
	case TransferCharacteristics::SRGB:
		func.to_linear = srgb_eotf;
		func.to_gamma = srgb_inverse_eotf;
		break;
	case TransferCharacteristics::ST_2084:
		func.to_linear = scene_referred ? st_2084_inverse_oetf : st_2084_eotf;
		func.to_gamma = scene_referred ? st_2084_oetf : st_2084_inverse_eotf;
		func.to_linear_scale = static_cast<float>(ST2084_PEAK_LUMINANCE / peak_luminance);
		func.to_gamma_scale = static_cast<float>(peak_luminance / ST2084_PEAK_LUMINANCE);
		break;
	case TransferCharacteristics::ARIB_B67:
		func.to_linear = scene_referred ? arib_b67_inverse_oetf : arib_b67_eotf;
		func.to_gamma = scene_referred ? arib_b67_oetf : arib_b67_inverse_eotf;
		func.to_linear_scale = 12.0f;
		func.to_gamma_scale = 1.0f / 12.0f;
		break;
	default:
		error::throw_<error::InternalError>("invalid transfer characteristics");
		break;
	}

	return func;
}

#if defined(_MSC_VER) && defined(_M_IX86)
EnsureSinglePrecision::EnsureSinglePrecision() noexcept : m_fpu_word(_control87(0, 0))
{
	_control87(_PC_24, _MCW_PC);
}

EnsureSinglePrecision::~EnsureSinglePrecision() { _control87(m_fpu_word, _MCW_PC); }
#endif

} // namespace colorspace
} // namespace zimg
