#pragma once

#ifndef ZIMG_RESIZE_RESIZE_H_
#define ZIMG_RESIZE_RESIZE_H_

#include <utility>
#include "graph/image_filter.h"

namespace zimg {;

enum class CPUClass;
enum class PixelType;

namespace graph {;

class ImageFilter;

} // namespace graph


namespace resize {;

class Filter;

std::pair<graph::ImageFilter *, graph::ImageFilter *> create_resize(
	const Filter &filter, PixelType type, unsigned depth, unsigned src_width, unsigned src_height, unsigned dst_width, unsigned dst_height,
	double shift_w, double shift_h, double subwidth, double subheight, CPUClass cpu);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_H_
