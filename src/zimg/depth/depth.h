#pragma once

#ifndef ZIMG_DEPTH_DEPTH_H_
#define ZIMG_DEPTH_DEPTH_H_

#include <memory>
#include "common/builder.h"
#include "common/pixel.h"

namespace zimg {;

enum class CPUClass;
struct PixelFormat;

namespace graph {;

class ImageFilter;

} // namespace graph


namespace depth {;

enum class DitherType {
	DITHER_NONE,
	DITHER_ORDERED,
	DITHER_RANDOM,
	DITHER_ERROR_DIFFUSION
};

struct DepthConversion {
	unsigned width;
	unsigned height;

#include "common/builder.h"
	BUILDER_MEMBER(PixelFormat, pixel_in)
	BUILDER_MEMBER(PixelFormat, pixel_out)
	BUILDER_MEMBER(DitherType, dither_type)
	BUILDER_MEMBER(CPUClass, cpu)
#undef BUILDER_MEMBER

	DepthConversion(unsigned width, unsigned height);

	std::unique_ptr<graph::ImageFilter> create() const;
};

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DEPTH2_H_
