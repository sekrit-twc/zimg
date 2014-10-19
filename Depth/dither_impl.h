#pragma once

#ifndef ZIMG_DEPTH_DITHER_IMPL_H_
#define ZIMG_DEPTH_DITHER_IMPL_H_

#include <cstdint>
#include "Common/align.h"
#include "dither.h"

namespace zimg {
namespace depth {

/**
 * Base class for ordered dither implementations.
 */
class OrderedDither : public DitherConvert {
protected:	
	/**
	 * Array of fixed dither offsets to add (range -0.5 to 0.5).
	 * This is a n x m array, where the width is given by m_period.
	 */
	AlignedVector<float> m_dither;

	/**
	 * Number of consecutive horizontal dithers.
	 */
	int m_period;

	/**
	 * Initialize the implementation with the given coefficients.
	 *
	 * @param first first dither
	 * @param last last dither
	 * @param period number of consecutive horizontal dithers.
	 */
	OrderedDither(const float *first, const float *last, int period);
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
