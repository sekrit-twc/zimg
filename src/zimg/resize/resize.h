#pragma once

#ifndef ZIMG_RESIZE_RESIZE_H_
#define ZIMG_RESIZE_RESIZE_H_

#include <memory>
#include <utility>
#include "common/builder.h"

namespace zimg {;

enum class CPUClass;
enum class PixelType;

namespace graph {;

class ImageFilter;

} // namespace graph


namespace resize {;

class Filter;

struct ResizeConversion {
	typedef std::pair<std::unique_ptr<graph::ImageFilter>, std::unique_ptr<graph::ImageFilter>> filter_pair;

	unsigned src_width;
	unsigned src_height;
	PixelType type;

#include "common/builder.h"
	BUILDER_MEMBER(unsigned, depth);
	BUILDER_MEMBER(const Filter *, filter);
	BUILDER_MEMBER(unsigned, dst_width);
	BUILDER_MEMBER(unsigned, dst_height);
	BUILDER_MEMBER(double, shift_w);
	BUILDER_MEMBER(double, shift_h);
	BUILDER_MEMBER(double, subwidth);
	BUILDER_MEMBER(double, subheight);
	BUILDER_MEMBER(CPUClass, cpu);
#undef BUILDER_MEMBER

	ResizeConversion(unsigned src_width, unsigned src_height, PixelType type);

	filter_pair create() const;
};

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_H_
