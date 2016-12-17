#pragma once

#ifndef ZIMG_COLORSPACE_GAMMA_H_
#define ZIMG_COLORSPACE_GAMMA_H_

namespace zimg {
namespace colorspace {

constexpr float ST2084_PEAK_LUMINANCE = 10000.0f; // Units of cd/m^2.

// Scene-referred transfer functions.
float rec_709_oetf(float x) noexcept;
float rec_709_inverse_oetf(float x) noexcept;

float arib_b67_oetf(float x) noexcept;
float arib_b67_inverse_oetf(float x) noexcept;

// Display-referred transfer functions.
float srgb_eotf(float x) noexcept;
float srgb_inverse_eotf(float x) noexcept;

float st_2084_eotf(float x) noexcept;
float st_2084_inverse_eotf(float x) noexcept;

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_GAMMA_H_
