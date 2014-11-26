#pragma once

#ifndef ZIMG_COLORSPACE_COLORSPACE_H_
#define ZIMG_COLORSPACE_COLORSPACE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "operation.h"

namespace zimg {;

enum class PixelType;
enum class CPUClass;

template <class T>
class ImagePlane;

namespace colorspace {;

struct ColorspaceDefinition;

/**
 * ColorspaceConversion: converts between colorspaces.
 *
 * Each instance is applicable only for its given set of source and destination colorspace.
 */
class ColorspaceConversion {
	std::shared_ptr<PixelAdapter> m_pixel_adapter;
	std::vector<std::shared_ptr<Operation>> m_operations;

	void load_line(const void *src, float *dst, int width, PixelType type) const;
	void store_line(const float *src, void *dst, int width, PixelType type) const;
public:
	/**
	 * Initialize a null context. Cannot be used for execution.
	 */
	ColorspaceConversion() = default;

	/**
	 * Initialize a context to apply a given colorspace conversion.
	 *
	 * @param in input colorspace
	 * @param out output colorspace
	 * @param cpu create context optimized for given cpu
	 * @throws ZimgIllegalArgument on invalid colorspace definition
	 * @throws ZimgOutOfMemory if out of memory
	 */
	ColorspaceConversion(const ColorspaceDefinition &in, const ColorspaceDefinition &out, CPUClass cpu);

	/**
	 * Get the size of the temporary buffer required by the conversion.
	 *
	 * @param width width of image line.
	 * @return the size of the temporary buffer in units of floats
	 */
	size_t tmp_size(int width) const;

	/**
	 * Process an image. The input and output pixel formats must match.
	 *
	 * @param src pointer to three input planes
	 * @param dst pointer to three output planes
	 * @param tmp temporary buffer (@see ColorspaceConversion::tmp_size)
	 * @throws ZimgUnsupportedError if pixel type not supported
	 */
	void process(const ImagePlane<const void> *src, const ImagePlane<void> *dst, void *tmp) const;
};

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_COLORSPACE_H_
