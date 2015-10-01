#pragma once

#ifndef ZIMG_RESIZE_RESIZE2_H_
#define ZIMG_RESIZE_RESIZE2_H_

#include <utility>
#include "Common/zfilter.h"

namespace zimg {;

enum class CPUClass;
enum class PixelType;

class IZimgFilter;

namespace resize {;

class Filter;

std::pair<IZimgFilter *, IZimgFilter *> create_resize(
	const Filter &filter, PixelType type, unsigned depth, int src_width, int src_height, int dst_width, int dst_height,
	double shift_w, double shift_h, double subwidth, double subheight, CPUClass cpu);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE2_H_
