#pragma once

#ifndef ZIMG_COLORSPACE_GAMMA_H_
#define ZIMG_COLORSPACE_GAMMA_H_

namespace zimg {
namespace colorspace {

enum class TransferCharacteristics;

constexpr float ST2084_PEAK_LUMINANCE = 10000.0f; // Units of cd/m^2.

typedef float (*gamma_func)(float);

// Scene-referred transfer functions.
float rec_709_oetf(float x) noexcept;
float rec_709_inverse_oetf(float x) noexcept;

float arib_b67_oetf(float x) noexcept;
float arib_b67_inverse_oetf(float x) noexcept;

// Display-referred transfer functions.
float rec_1886_eotf(float x) noexcept;
float rec_1886_inverse_eotf(float x) noexcept;

float srgb_eotf(float x) noexcept;
float srgb_inverse_eotf(float x) noexcept;

float st_2084_eotf(float x) noexcept;
float st_2084_inverse_eotf(float x) noexcept;

// Derived functions.
float arib_b67_eotf(float x) noexcept;
float arib_b67_inverse_eotf(float x) noexcept;

float st_2084_oetf(float x) noexcept;
float st_2084_inverse_oetf(float x) noexcept;


struct TransferFunction {
	gamma_func to_linear;
	gamma_func to_gamma;
	float to_linear_scale;
	float to_gamma_scale;
};

TransferFunction select_transfer_function(TransferCharacteristics transfer, double peak_luminance, bool scene_referred);


// MSVC 32-bit compiler generates x87 instructions when operating on floats
// returned from external functions. The caller must set the x87 precision to
// 24-bit (single precision) to ensure reproducible results.
#if defined(_MSC_VER) && defined(_M_IX86)
class EnsureSinglePrecision {
	unsigned m_fpu_word;
public:
	EnsureSinglePrecision() noexcept;
	EnsureSinglePrecision(const EnsureSinglePrecision &) = delete;

	~EnsureSinglePrecision();

	EnsureSinglePrecision &operator=(const EnsureSinglePrecision &) = delete;
};
#else
struct EnsureSinglePrecision {
	EnsureSinglePrecision() {}
	EnsureSinglePrecision(const EnsureSinglePrecision &) = delete;

	~EnsureSinglePrecision() {}

	EnsureSinglePrecision &operator=(const EnsureSinglePrecision &) = delete;
};
#endif

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_GAMMA_H_
