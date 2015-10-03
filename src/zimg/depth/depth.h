#pragma once

#ifndef ZIMG_DEPTH_DEPTH_H_
#define ZIMG_DEPTH_DEPTH_H_

#include <memory>

namespace zimg {;

enum class CPUClass;
struct PixelFormat;

namespace graph {;

class IZimgFilter;

} // namespace graph


namespace depth {;

enum class DitherType {
	DITHER_NONE,
	DITHER_ORDERED,
	DITHER_RANDOM,
	DITHER_ERROR_DIFFUSION
};

graph::IZimgFilter *create_depth(DitherType type, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DEPTH2_H_
