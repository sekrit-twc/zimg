#pragma once

#ifndef ZIMG_DEPTH_DEPTH2_H_
#define ZIMG_DEPTH_DEPTH2_H_

#include <memory>
#include "Common/zfilter.h"

namespace zimg {;

enum class CPUClass;
struct PixelFormat;

namespace depth {;

#ifndef ZIMG_DEPTH_DEPTH_H_
enum class DitherType {
	DITHER_NONE,
	DITHER_ORDERED,
	DITHER_RANDOM,
	DITHER_ERROR_DIFFUSION
};
#endif

IZimgFilter *create_depth2(DitherType type, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DEPTH2_H_
