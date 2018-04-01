#pragma once

#ifdef ZIMG_X86_AVX512

#ifndef ZIMG_COLORSPACE_X86_GAMMA_CONSTANTS_H_
#define ZIMG_COLORSPACE_X86_GAMMA_CONSTANTS_H_

namespace zimg {
namespace colorspace {
namespace avx512constants {

struct Rec1886EOTF {
	// 5-th order polynomial on domain [1, 2).
	static const float horner[6];
	// Exponent lookup table for range reduction, [-15, +0].
	static const float table alignas(64)[16];
};

struct Rec1886InverseEOTF {
	// 5-th order polynomial on domain [1, 2).
	static const float horner[6];
	// Exponent lookup table for range reduction, [-31, +0].
	static const float table alignas(64)[32];
};

struct ST2084EOTF {
	// 32 4-th order polynomials on uniform domain [i / 32, (i + 1) / 32).
	static const float horner0 alignas(64)[32];
	static const float horner1 alignas(64)[32];
	static const float horner2 alignas(64)[32];
	static const float horner3 alignas(64)[32];
	static const float horner4 alignas(64)[32];
};

struct ST2084InverseEOTF {
	// 32 4-th order polynomials on logarithmic domain [2 ^ i, 2 ^ (1 + 1)).
	static const float horner0 alignas(64)[32];
	static const float horner1 alignas(64)[32];
	static const float horner2 alignas(64)[32];
	static const float horner3 alignas(64)[32];
	static const float horner4 alignas(64)[32];
};

// Debug implementations.
float rec_1886_eotf(float x);
float rec_1886_inverse_eotf(float x);

float st_2084_eotf(float x);
float st_2084_inverse_eotf(float x);

} // namespace avx512constants
} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_X86_GAMMA_CONSTANTS_H_

#endif // ZIMG_X86_AVX512
