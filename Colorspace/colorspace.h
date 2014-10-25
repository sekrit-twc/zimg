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

namespace colorspace {;

struct ColorspaceDefinition;

/**
 * ColorspaceConversion: converts between colorspaces.
 *
 * Each instance is applicable only for its given set of source and destination colorspace.
 */
class ColorspaceConversion {
	std::unique_ptr<PixelAdapater> m_pixel_adapter;
	std::vector<std::unique_ptr<Operation>> m_operations;
	bool m_input_is_yuv;
	bool m_output_is_yuv;

	void load_line(const void *src, float *dst, int width, bool tv, bool chroma, PixelType type) const;
	void store_line(const float *src, void *dst, int width, bool tv, bool chroma, PixelType type) const;
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
	 * Get the size of the temporary buffer required by the convesion.
	 *
	 * @param width width of image line.
	 */
	size_t tmp_size(int width) const;

	/**
	 * Process an image. The input and output pixel formats must match.
	 *
	 * @param type pixel type of image
	 * @param src pointer to three input planes pointers
	 * @param dst pointer to three output planes pointers
	 * @param tmp temporary buffer (@see ColorspaceConversion::tmp_size)
	 * @param width width of image
	 * @param height height of image
	 * @param src_stride stride of each input plane
	 * @param dst_stride stride of each output plane
	 * @param tv_in whether input is TV range, only applicable to integer pixel formats
	 * @param tv_out whether output is TV range, only applicable to integer pixel formats
	 * @throws ZimgUnsupportedError if pixel type not supported
	 */
	void process(PixelType type, const void * const *src, void * const *dst, void *tmp, int width, int height, const int *src_stride, const int *dst_stride, bool tv_in, bool tv_out) const;
};

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_COLORSPACE_H_
