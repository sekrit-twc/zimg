#pragma once

#ifndef ZIMG_DEPTH_DEPTH_H_
#define ZIMG_DEPTH_DEPTH_H_

#include <cstddef>
#include <memory>

namespace zimg {;

enum class CPUClass;

template <class T>
class ImagePlane;

namespace depth {;

class DepthConvert;
class DitherConvert;

/**
 * Enum for dithering modes.
 */
enum class DitherType {
	DITHER_NONE,
	DITHER_ORDERED,
	DITHER_RANDOM,
	DITHER_ERROR_DIFFUSION
};

/**
 * Depth: converts between pixel types and bit depths.
 *
 * Each instance is applicable only for its given dither type.
 */
class Depth {
	std::shared_ptr<DepthConvert> m_depth;
	std::shared_ptr<DitherConvert> m_dither;
	bool m_error_diffusion;
public:
	/**
	 * Initialize a null context. Cannot be used for execution.
	 */
	Depth() = default;

	/**
	 * Initialize a context to apply a given dither type.
	 *
	 * @param type dither type
	 * @param cpu create context optimized for given cpu
	 * @throws ZimgIllegalArgument on invalid dither type
	 * @throws ZimgOutOfMemory if out of memory
	 */
	Depth(DitherType type, CPUClass cpu);

	/**
	 * Get the size of the temporary buffer required by the conversion.
	 *
	 * @param width width of image line.
	 * @return the size of the temporary buffer in units of floats
	 */
	size_t tmp_size(int width) const;

	/**
	 * Process an image.
	 *
	 * @param src input plane
	 * @param dst output plane
	 * @param tmp temporary buffer (@see Depth::tmp_size)
	 * @throws ZimgUnsupportedError if conversion not supported
	 */
	void process(const ImagePlane<const void> &src, const ImagePlane<void> &dst, void *tmp) const;
};

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DEPTH_H_
