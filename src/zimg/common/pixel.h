#pragma once

#ifndef ZIMG_PIXEL_H_
#define ZIMG_PIXEL_H_

#include <cstddef>
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
 * Table used by {@link pixel_get_traits}.
 */
constexpr PixelTraits pixel_traits_table[] = {
	{ sizeof(uint8_t),   8, AlignmentOf<uint8_t>,  true },
	{ sizeof(uint16_t), 16, AlignmentOf<uint16_t>, true },
	{ sizeof(uint16_t), 16, AlignmentOf<uint16_t>, false },
	{ sizeof(float),    32, AlignmentOf<float>,    false },
};

/**
 * Query traits for a given pixel type.
 *
 * @param type pixel type
 * @return static reference to traits structure
 */
constexpr const PixelTraits &pixel_get_traits(PixelType type) noexcept
{
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE-754 not detected");
	static_assert(sizeof(pixel_traits_table) / sizeof(pixel_traits_table[0]) == static_cast<int>(PixelType::FLOAT) + 1,
	              "table size incorrect");
	// zassert_d(type >= PixelType::BYTE && type <= PixelType::FLOAT, "pixel type out of range");
	return pixel_traits_table[static_cast<int>(type)];
}

/**
 * Query the size in bytes of a pixel.
 *
 * @param type pixel type
 * @return size
 */
constexpr unsigned pixel_size(PixelType type) noexcept
{
	return pixel_get_traits(type).size;
}

/**
 * Query the maximum bit depth that can be stored in a pixel type.
 *
 * @param type pixel type
 * @return bit depth
 */
constexpr unsigned pixel_depth(PixelType type) noexcept
{
	return pixel_get_traits(type).depth;
}

/**
 * Query the alignment of a pixel type in units of pixels.
 *
 * @param type pixel type
 * @return alignment
 */
constexpr unsigned pixel_alignment(PixelType type) noexcept
{
	return pixel_get_traits(type).alignment;
}

/**
 * Query if the pixel type is integral.
 *
 * @param type pixel type
 * @return true if integral, else false
 */
constexpr bool pixel_is_integer(PixelType type) noexcept
{
	return pixel_get_traits(type).is_integer;
}

/**
 * Query if the pixel type is floating point.
 *
 * @param type pixel type
 * @return true if float, else false
 */
constexpr bool pixel_is_float(PixelType type) noexcept
{
	return !pixel_is_integer(type);
}

/**
 * Query the maximum image width that can be supported for a pixel type.
 *
 * @param type pixel type
 * @return maximum width
 */
constexpr unsigned pixel_max_width(PixelType type) noexcept
{
	return static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) > std::numeric_limits<unsigned>::max()
		? floor_n(std::numeric_limits<unsigned>::max(), pixel_alignment(type))
		: floor_n(static_cast<unsigned>(std::numeric_limits<ptrdiff_t>::max()), ALIGNMENT) / pixel_size(type);
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
	constexpr PixelFormat() noexcept :
		type{},
		depth{},
		fullrange{},
		chroma{},
		ycgco{}
	{}

	/**
	 * Construct PixelFormat with default parameters for a given pixel type,
	 * creating a limited-range, luma format with the maximum depth of the type.
	 *
	 * @param type pixel type
	 */
	constexpr PixelFormat(PixelType type) noexcept :
		type{ type },
		depth{ pixel_depth(type) },
		fullrange{},
		chroma{},
		ycgco{}
	{}

	/**
	 * Initialize PixelFormat with given parameters.
	 *
	 * @param type pixel type
	 * @param depth bit depth
	 * @param fullrange true if full range, else false
	 * @param chroma true if chroma, else false
	 */
	constexpr PixelFormat(PixelType type, unsigned depth, bool fullrange = false, bool chroma = false, bool ycgco = false) noexcept :
		type{ type },
		depth{ depth },
		fullrange{ fullrange },
		chroma{ chroma },
		ycgco{ ycgco }
	{}
};

/**
 * Compare PixelFormat structures for equality.
 *
 * Integer formats are considered equal if all fields match, whereas
 * floating-point formats are defined only by their type and chroma.
 *
 * @param a lhs structure
 * @param b rhs structure
 * @return true if equal, else false
 */
constexpr bool operator==(const PixelFormat &a, const PixelFormat &b) noexcept
{
	return pixel_is_float(a.type)
		? a.type == b.type && a.chroma == b.chroma
		: a.type == b.type && a.depth == b.depth && a.fullrange == b.fullrange && a.chroma == b.chroma;
}

/**
 * Compare PixelFormat structures for inequality.
 *
 * @see operator==(const PixelFormat &, const PixelFormat &)
 */
constexpr bool operator!=(const PixelFormat &a, const PixelFormat &b) noexcept
{
	return !(a == b);
}

} // namespace zimg

#endif // ZIMG_PIXEL_H_
