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

constexpr float SMPTE_240M_ALPHA = 1.1115;
constexpr float SMPTE_240M_BETA  = 0.0228;

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


// Chosen for compatibility with higher precision REC709_ALPHA/REC709_BETA.
// See: ITU-R BT.2390-2 5.3.1
constexpr float ST2084_OOTF_SCALE = 59.49080238715383f;


float ootf_1_2(float x) noexcept
{
	return x < 0.0f ? x : zimg_x_powf(x, 1.2f);
}

float inverse_ootf_1_2(float x) noexcept
{
	return x < 0.0f ? x : zimg_x_powf(x, 1.0f / 1.2f);
}

float ootf_st2084(float x) noexcept
{
	return rec_1886_eotf(rec_709_oetf(x * ST2084_OOTF_SCALE)) / 100.0f;
}

float inverse_ootf_st2084(float x) noexcept
{
	return rec_709_inverse_oetf(rec_1886_inverse_eotf(x * 100.0f)) / ST2084_OOTF_SCALE;
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

float log100_oetf(float x) noexcept
{
	return x <= 0.01f ? 0 : 1.0f + log10f(x) / 2.0f;
}

float log100_inverse_oetf(float x) noexcept
{
	return x <= 0.0f ? 0.01f : zimg_x_powf(10, 2 * (x - 1.0f));
}

float log316_oetf(float x) noexcept
{
	return x <= 0.0316227766f ? 0 : 1.0f + log10f(x) / 2.5f;
}

float log316_inverse_oetf(float x) noexcept
{
	return x <= 0.0f ? 0.00316227766f : zimg_x_powf(10, 2.5 * (x - 1.0f));
}

float rec_470m_eotf(float x) noexcept
{
	return x < 0.0f ? 0.0f : zimg_x_powf(x, 2.2f);
}

float rec_470m_inverse_eotf(float x) noexcept
{
	return x < 0.0f ? 0.0f : zimg_x_powf(x, 1.0f / 2.2f);
}

float rec_470bg_eotf(float x) noexcept
{
	return x < 0.0f ? 0.0f : zimg_x_powf(x, 2.8f);
}

float rec_470bg_inverse_eotf(float x) noexcept
{
	return x < 0.0f ? 0.0f : zimg_x_powf(x, 1.0f / 2.8f);
}

float smpte_240m_oetf(float x) noexcept
{
	if (x < 4.0f * SMPTE_240M_BETA)
		x = x / 4.0f;
	else
		x = zimg_x_powf((x + (SMPTE_240M_ALPHA - 1.0f)) / SMPTE_240M_ALPHA, 1.0f / 0.45f);

	return x;
}

float smpte_240m_inverse_oetf(float x) noexcept
{
	if (x < SMPTE_240M_BETA)
		x = x * 4.0f;
	else
		x = SMPTE_240M_ALPHA * zimg_x_powf(x, 0.45f) - (SMPTE_240M_ALPHA - 1.0f);

	return x;
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

float xvycc_eotf(float x) noexcept
{
	return copysign(rec_1886_eotf(fabs(x)), x);
}

float xvycc_inverse_eotf(float x) noexcept
{
	return copysign(rec_1886_inverse_eotf(fabs(x)), x);
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

float st_2084_oetf(float x) noexcept
{
	return st_2084_inverse_eotf(ootf_st2084(x));
}

float st_2084_inverse_oetf(float x) noexcept
{
	return inverse_ootf_st2084(st_2084_eotf(x));
}


TransferFunction select_transfer_function(TransferCharacteristics transfer, double peak_luminance, bool scene_referred)
{
	zassert_d(!std::isnan(peak_luminance), "nan detected");

	TransferFunction func{};

	func.to_linear_scale = 1.0f;
	func.to_gamma_scale = 1.0f;

	switch (transfer) {
	case TransferCharacteristics::LOG_100:
		func.to_linear = log100_inverse_oetf;
		func.to_gamma = log100_oetf;
		break;
	case TransferCharacteristics::LOG_316:
		func.to_linear = log316_inverse_oetf;
		func.to_gamma = log316_oetf;
		break;
	case TransferCharacteristics::REC_709:
		func.to_linear = scene_referred ? rec_709_inverse_oetf : rec_1886_eotf;
		func.to_gamma = scene_referred ? rec_709_oetf : rec_1886_inverse_eotf;
		break;
	case TransferCharacteristics::REC_470_M:
		func.to_linear = rec_470m_eotf;
		func.to_gamma = rec_470m_inverse_eotf;
		break;
	case TransferCharacteristics::REC_470_BG:
		func.to_linear = rec_470bg_eotf;
		func.to_gamma = rec_470bg_inverse_eotf;
		break;
	case TransferCharacteristics::SMPTE_240M:
		func.to_linear = scene_referred ? smpte_240m_inverse_oetf : rec_1886_eotf;
		func.to_gamma = scene_referred ? smpte_240m_oetf : rec_1886_inverse_eotf;
		break;
	case TransferCharacteristics::XVYCC:
		func.to_linear = xvycc_eotf;
		func.to_gamma = xvycc_inverse_eotf;
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
		func.to_linear_scale = scene_referred ? 12.0f : static_cast<float>(1000.0 / peak_luminance);
		func.to_gamma_scale = scene_referred ? 1.0f / 12.0f : static_cast<float>(peak_luminance / 1000.0);
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
