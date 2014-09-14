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

} // namespace zimg

#endif // ZIMG_PIXEL_H_
