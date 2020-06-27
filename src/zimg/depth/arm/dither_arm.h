#pragma once

#ifdef ZIMG_ARM

#ifndef ZIMG_DEPTH_ARM_DITHER_ARM_H_
#define ZIMG_DEPTH_ARM_DITHER_ARM_H_

#include <memory>
#include "depth/dither.h"

namespace zimg {

namespace graph {

class ImageFilter;

} // namespace graph


namespace depth {

#define DECLARE_ORDERED_DITHER(x, cpu) \
void ordered_dither_##x##_##cpu(const float *dither, unsigned dither_offset, unsigned dither_mask, \
                                const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)

DECLARE_ORDERED_DITHER(b2b, neon);
DECLARE_ORDERED_DITHER(b2w, neon);
DECLARE_ORDERED_DITHER(w2b, neon);
DECLARE_ORDERED_DITHER(w2w, neon);
DECLARE_ORDERED_DITHER(f2b, neon);
DECLARE_ORDERED_DITHER(f2w, neon);

#undef DECLARE_ORDERED_DITHER

dither_convert_func select_ordered_dither_func_arm(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_ARM_DITHER_ARM_H_

#endif // ZIMG_ARM
