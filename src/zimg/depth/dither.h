#pragma once

#ifndef ZIMG_DEPTH_DITHER_H_
#define ZIMG_DEPTH_DITHER_H_

#include <tuple>

namespace zimg {;

enum class CPUClass;

struct PixelFormat;

namespace graph {;

class IZimgFilter;

} // namespace graph


namespace depth {;

enum class DitherType;

typedef void (*dither_convert_func)(const float *dither, unsigned dither_offset, unsigned dither_mask,
                                    const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right);
typedef void (*dither_f16c_func)(const void *src, void *dst, unsigned left, unsigned right);

graph::IZimgFilter *create_dither(DitherType type, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DITHER_H_
