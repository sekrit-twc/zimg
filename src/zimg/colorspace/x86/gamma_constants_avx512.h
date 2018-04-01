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

// Debug implementations.
float rec_1886_eotf(float x);
float rec_1886_inverse_eotf(float x);

} // namespace avx512constants
} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_X86_GAMMA_CONSTANTS_H_

#endif // ZIMG_X86_AVX512
