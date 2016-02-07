#pragma once

#ifndef ZIMG_PIXEL_H_
#define ZIMG_PIXEL_H_

#include <cstdint>
#include <limits>
#include "align.h"
#include "zassert.h"

namespace zimg {

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
 * Struct defining the memory characteristics of pixel types.
 */
struct PixelTraits {
	unsigned size;
	unsigned depth;
	unsigned alignment;
	bool is_integer;
};

/**
 * Query traits for a given pixel type.
 *
 * @param type pixel type
 * @return static reference to traits structure
 */
inline const PixelTraits &pixel_get_traits(PixelType type)
{
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE-754 not detected");

	static const PixelTraits traits[] = {
		{ sizeof(uint8_t),   8, AlignmentOf<uint8_t>::value,  true },
		{ sizeof(uint16_t), 16, AlignmentOf<uint16_t>::value, true },
		{ sizeof(uint16_t), 16, AlignmentOf<uint16_t>::value, false },
		{ sizeof(float),    32, AlignmentOf<float>::value,    false },
	};
	static_assert(sizeof(traits) / sizeof(traits[0]) == static_cast<int>(PixelType::FLOAT) + 1,
	              "table size incorrect");

	_zassert_d(type >= PixelType::BYTE && type <= PixelType::FLOAT, "pixel type out of range");
	return traits[static_cast<int>(type)];
}

/**
 * Query the size in bytes of a pixel.
 *
 * @param type pixel type
 * @return size
 */
inline unsigned pixel_size(PixelType type)
{
	return pixel_get_traits(type).size;
}

/**
 * Query the maximum bit depth that can be stored in a pixel type.
 *
 * @param type pixel type
 * @return bit depth
 */
inline unsigned pixel_depth(PixelType type)
{
	return pixel_get_traits(type).depth;
}

/**
 * Query the alignment of a pixel type in units of pixels.
 *
 * @param type pixel type
 * @return alignment
 */
inline unsigned pixel_alignment(PixelType type)
{
	return pixel_get_traits(type).alignment;
}

/**
 * Query if the pixel type is integral.
 *
 * @param type pixel type
 * @return true if integral, else false
 */
inline bool pixel_is_integer(PixelType type)
{
	return pixel_get_traits(type).is_integer;
}

/**
 * Query if the pixel type is floating point.
 *
 * @param type pixel type
 * @return true if float, else false
 */
inline bool pixel_is_float(PixelType type)
{
	return !pixel_is_integer(type);
}

/**
 * Struct defining the set of parameters required to convert between pixel types.
 */
struct PixelFormat {
	PixelType type;
	unsigned depth;
	bool fullrange;
	bool chroma;
	bool ycgco;

	/**
	 * Default construct PixelFormat, initializing it with an invalid format.
	 */
	PixelFormat() :
		type{},
		depth{},
		fullrange{},
		chroma{},
		ycgco{}
	{
	}

	/**
	 * Construct PixelFormat with default parameters for a given pixel type,
	 * creating a limited-range, luma format with the maximum depth of the type.
	 *
	 * @param type pixel type
	 */
	PixelFormat(PixelType type) :
		type{ type },
		depth{ pixel_depth(type) },
		fullrange{},
		chroma{},
		ycgco{}
	{
	}

	/**
	 * Initialize PixelFormat with given parameters.
	 *
	 * @param type pixel type
	 * @param depth bit depth
	 * @param fullrange true if full range, else false
	 * @param chroma true if chroma, else false
	 */
	PixelFormat(PixelType type, unsigned depth, bool fullrange = false, bool chroma = false, bool ycgco = false) :
		type{ type },
		depth{ depth },
		fullrange{ fullrange },
		chroma{ chroma },
		ycgco{ ycgco }
	{
	}
};

/**
 * Compare PixelFormat structures for equality.
 * Integer formats are considered equal if all fields match, whereas
 * floating-point formats are defined only by their type and chroma.
 *
 * @param a lhs structure
 * @param b rhs structure
 * @return true if equal, else false
 */
inline bool operator==(const PixelFormat &a, const PixelFormat &b)
{
	if (pixel_is_float(a.type))
		return a.type == b.type && a.chroma == b.chroma;
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
