#pragma once

#ifndef ZIMG_PIXEL_H_
#define ZIMG_PIXEL_H_

#include <cstdint>
#include "align.h"

namespace zimg {;

/**
 * Enum for supported input and output formats.
 *
 * BYTE = 8 bits
 * WORD = 16 bits
 * HALF = 16 bits
 * FLOAT = 32 bits
 */
enum class PixelType {
	BYTE,
	WORD,
	HALF,
	FLOAT
};

/**
 * Struct defining the set of parameters required to convert between pixel types.
 */
struct PixelFormat {
	PixelType type;
	unsigned depth;
	bool fullrange;
	bool chroma;
};

/**
 * Get the size in bytes of a pixel.
 *
 * @param type type of pixel
 * @return size of pixel
 */
inline unsigned pixel_size(PixelType type)
{
	switch (type) {
	case PixelType::BYTE:
		return 1;
	case PixelType::WORD:
	case PixelType::HALF:
		return sizeof(uint16_t);
	case PixelType::FLOAT:
		return sizeof(float);
	default:
		return 0;
	}
}

/**
 * Get the system alignment in units of pixels.
 *
 * @param type type of pixel
 * @return alignment in pixels
 */
inline unsigned pixel_alignment(PixelType type)
{
	return ALIGNMENT / pixel_size(type);
}

/**
 * Get a default pixel format for a given pixel type.
 * The default format is TV-range, non-chroma, and uses all available bits.
 *
 * @param type type of pixel
 * @return default format
 */
inline PixelFormat default_pixel_format(PixelType type)
{
	return{ type, pixel_size(type) * 8, false, false };
}

/**
 * Compare PixelFormat structures for equality.
 * Integer formats are considered equal if all fields match,
 * whereas floating-point formats are defined only by their PixelType.
 *
 * @param a lhs structure
 * @param b rhs structure
 * @return true if equal, else false
 */
inline bool operator==(const PixelFormat &a, const PixelFormat &b)
{
	if (a.type >= PixelType::HALF)
		return a.type == b.type;
	else
		return a.type == b.type && a.depth == b.depth && a.fullrange == b.fullrange && a.chroma == b.chroma;
}

/**
 * Compare PixelFormat structures for inequality.
 *
 * @see operator==(const PixelFormat &, const PixelFormat &)
 */
inline bool operator!=(const PixelFormat &a, const PixelFormat &b)
{
	return !(a == b);
}

} // namespace zimg

#endif // ZIMG_PIXEL_H_
