#pragma once

#ifndef ZIMG_DEPTH_DITHER_IMPL_H_
#define ZIMG_DEPTH_DITHER_IMPL_H_

#include <cstdint>
#include "Common/align.h"
#include "dither.h"

namespace zimg {;

enum class CPUClass;

namespace depth {;

/**
 * Base class for ordered dither implementations.
 */
class OrderedDither : public DitherConvert {
protected:	
	/**
	 * Array of fixed dither offsets to add (range -0.5 to 0.5).
	 * This is a 64x64 array.
	 */
	AlignedVector<float> m_dither;

	/**
	 * Initialize the implementation with the given coefficients.
	 *
	 * @param dither coefficient table
	 */
	explicit OrderedDither(const float *dither);
public:
	static const int NUM_DITHERS = 64 * 64;
	static const int NUM_DITHERS_H = 64;
	static const int NUM_DITHERS_V = 64;
};

/**
 * Create a concrete OrderedDither.
 * Several dither modes are implemented through OrderedDither, including NONE, ORDERED, and RANDOM.
 *
 * @param type dither type
 * @param cpu create implementation optimized for given cpu
 */
DitherConvert *create_ordered_dither(DitherType type, CPUClass cpu);

}; // namespace depth
}; // namespace zimg

#endif // ZIMG_DEPTH_DITHER_IMPL_H_
