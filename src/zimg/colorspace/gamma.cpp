#include <algorithm>
#include <cfloat>
#include <cmath>

#include "common/libm_wrapper.h"
#include "gamma.h"

// MSVC 32-bit compiler generates x87 instructions when operating on floats
// returned from external functions. Force single precision to avoid errors.
#if defined(_MSC_VER) && defined(_M_IX86)
  #define fpu_save() _control87(0, 0)
  #define fpu_set_single() _control87(_PC_24, _MCW_PC)
  #define fpu_restore(x) _control87((x), _MCW_PC)
#else
  #define fpu_save() 0
  #define fpu_set_single() (void)0
  #define fpu_restore(x) (void)x
#endif /* _MSC_VER */

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


class EnsureSinglePrecision {
	unsigned m_fpu_word;
public:
	EnsureSinglePrecision() noexcept : m_fpu_word{ fpu_save() } { fpu_set_single(); }
	EnsureSinglePrecision(const EnsureSinglePrecision &) = delete;

	~EnsureSinglePrecision() { fpu_restore(m_fpu_word); }

	EnsureSinglePrecision &operator=(const EnsureSinglePrecision &) = delete;
};

} // namespace


float rec_709_oetf(float x) noexcept
{
	EnsureSinglePrecision x87;

	if (x < REC709_BETA)
		x = x * 4.5f;
	else
		x = REC709_ALPHA * zimg_x_powf(x, 0.45f) - (REC709_ALPHA - 1.0f);

	return x;
}

float rec_709_inverse_oetf(float x) noexcept
{
	EnsureSinglePrecision x87;

	if (x < 4.5f * REC709_BETA)
		x = x / 4.5f;
	else
		x = zimg_x_powf((x + (REC709_ALPHA - 1.0f)) / REC709_ALPHA, 1.0f / 0.45f);

	return x;
}

float arib_b67_oetf(float x) noexcept
{
	EnsureSinglePrecision x87;

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
	EnsureSinglePrecision x87;

	// Prevent negative pixels expanding into positive values.
	x = std::max(x, 0.0f);

	if (x <= 0.5f)
		x = (x * x) * (1.0f / 3.0f);
	else
		x = (zimg_x_expf((x - ARIB_B67_C) / ARIB_B67_A) + ARIB_B67_B) / 12.0f;

	return x;
}

// Ignore the BT.1886 provisions for limited contrast and assume an ideal CRT.
float srgb_eotf(float x) noexcept
{
	EnsureSinglePrecision x87;

	if (x < 12.92f * SRGB_BETA)
		x = x / 12.92f;
	else
		x = zimg_x_powf((x + (SRGB_ALPHA - 1.0f)) / SRGB_ALPHA, 2.4f);

	return x;
}

float srgb_inverse_eotf(float x) noexcept
{
	EnsureSinglePrecision x87;

	if (x < SRGB_BETA)
		x = x * 12.92f;
	else
		x = SRGB_ALPHA * zimg_x_powf(x, 1.0f / 2.4f) - (SRGB_ALPHA - 1.0f);

	return x;
}

float st_2084_eotf(float x) noexcept
{
	EnsureSinglePrecision x87;

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
	EnsureSinglePrecision x87;

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

} // namespace colorspace
} // namespace zimg
