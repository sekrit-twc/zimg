#pragma once

#ifndef ZIMG_PIXEL_H_
#define ZIMG_PIXEL_H_

#include <cstdint>

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
	int depth;
	bool tv;
	bool chroma;
};

/**
 * Get the size in bytes of a pixel.
 *
 * @param type type of pixel
 * @return size of pixel
 */
inline int pixel_size(PixelType type)
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
 * Get a default pixel format for a given pixel type.
 *
 * @param type type of pixel
 * @return default format
 */
inline PixelFormat default_pixel_format(PixelType type)
{
	return{ type, pixel_size(type) * 8, true, false };
}

} // namespace zimg

#endif // ZIMG_PIXEL_H_
