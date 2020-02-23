#pragma once

#ifdef ZIMG_ARM

#ifndef ZIMG_DEPTH_ARM_DEPTH_CONVERT_ARM_H_
#define ZIMG_DEPTH_ARM_DEPTH_CONVERT_ARM_H_

#include "depth/depth_convert.h"

namespace zimg {
namespace depth {

#define DECLARE_LEFT_SHIFT(x, cpu) \
void left_shift_##x##_##cpu(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)

DECLARE_LEFT_SHIFT(b2b, neon);
DECLARE_LEFT_SHIFT(b2w, neon);
DECLARE_LEFT_SHIFT(w2b, neon);
DECLARE_LEFT_SHIFT(w2w, neon);

#undef DECLARE_LEFT_SHIFT

left_shift_func select_left_shift_func_arm(PixelType pixel_in, PixelType pixel_out, CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_ARM_DEPTH_CONVERT_ARM_H_

#endif // ZIMG_ARM
