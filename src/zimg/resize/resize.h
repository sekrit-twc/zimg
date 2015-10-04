#pragma once

#ifndef ZIMG_RESIZE_RESIZE_H_
#define ZIMG_RESIZE_RESIZE_H_

#include <utility>
#include "graph/zfilter.h"

namespace zimg {;

enum class CPUClass;
enum class PixelType;

namespace graph {;

class ImageFilter;

} // namespace graph


namespace resize {;

class Filter;

std::pair<graph::ImageFilter *, graph::ImageFilter *> create_resize(
	const Filter &filter, PixelType type, unsigned depth, int src_width, int src_height, int dst_width, int dst_height,
	double shift_w, double shift_h, double subwidth, double subheight, CPUClass cpu);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_H_
