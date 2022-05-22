#pragma once

#ifndef ZIMG_DEPTH_DEPTH_H_
#define ZIMG_DEPTH_DEPTH_H_

#include <array>
#include <memory>
#include "common/pixel.h"

namespace graphengine {
class Filter;
}


namespace zimg {

enum class CPUClass;

namespace graph {

class ImageFilter;

} // namespace graph


namespace depth {

enum class DitherType {
	NONE,
	ORDERED,
	RANDOM,
	ERROR_DIFFUSION,
};

struct DepthConversion {
	struct result {
		std::array<std::unique_ptr<graphengine::Filter>, 4> filters;
		std::array<const graphengine::Filter *, 4> filter_refs;

		result();

		result(std::unique_ptr<graphengine::Filter> filter, const bool planes[4]);
	};

	unsigned width;
	unsigned height;

#include "common/builder.h"
	BUILDER_MEMBER(PixelFormat, pixel_in)
	BUILDER_MEMBER(PixelFormat, pixel_out)
	BUILDER_MEMBER(DitherType, dither_type)
#define COMMA ,
	BUILDER_MEMBER(std::array<bool COMMA 4>, planes)
#undef COMMA
	BUILDER_MEMBER(CPUClass, cpu)
#undef BUILDER_MEMBER

	DepthConversion(unsigned width, unsigned height);

	std::unique_ptr<graph::ImageFilter> create() const;

	result create_ge() const;
};

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DEPTH2_H_
