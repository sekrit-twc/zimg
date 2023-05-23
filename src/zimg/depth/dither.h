#pragma once

#ifndef ZIMG_DEPTH_DITHER_H_
#define ZIMG_DEPTH_DITHER_H_

#include <memory>
#include "depth.h"

namespace graphengine {
class Filter;
}

namespace zimg {
enum class CPUClass;
struct PixelFormat;
}

namespace zimg::depth {

enum class DitherType;

typedef void (*dither_convert_func)(const float *dither, unsigned dither_offset, unsigned dither_mask,
                                    const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right);
typedef void (*dither_f16c_func)(const void *src, void *dst, unsigned left, unsigned right);

DepthConversion::result create_dither(DitherType type, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, const bool planes[4], CPUClass cpu);

} // namespace zimg::depth

#endif // ZIMG_DEPTH_DITHER_H_
