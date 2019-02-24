#pragma once

#ifndef ZIMG_ALIGN_H_
#define ZIMG_ALIGN_H_

namespace zimg {

/**
 * 64-byte alignment allows the use of instructions up to AVX-512.
 */
#ifdef ZIMG_X86
constexpr int ALIGNMENT = 64;
constexpr int ALIGNMENT_RELAXED = 32;
#else
constexpr int ALIGNMENT = sizeof(long double);
constexpr int ALIGNMENT_RELAXED = sizeof(long double);
#endif

/**
 * Round up the argument x to the nearest multiple of n.
 * x must be non-negative and n must be positive and power-of-2.
 */
template <class T>
constexpr T ceil_n(T x, unsigned n) { return (x + (n - 1)) & ~static_cast<T>(n - 1); }

/**
 * Round down the argument x to the nearest multiple of n.
 * x must be non-negative and n must be positive and power-of-2.
 */
template <class T>
constexpr T floor_n(T x, unsigned n) { return x & ~static_cast<T>(n - 1); }

/**
 * Helper struct that computes alignment in units of object count.
 *
 * @tparam T type of object
 */
template <class T>
struct AlignmentOf {
	static constexpr unsigned value = ALIGNMENT / sizeof(T);
};

} // namespace zimg

#endif // ZIMG_ALIGN_H_
