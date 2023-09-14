#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_X86_DEPTH_CONVERT_X86_H_
#define ZIMG_DEPTH_X86_DEPTH_CONVERT_X86_H_

#include "depth/depth_convert.h"

namespace zimg::depth {

#define DECLARE_LEFT_SHIFT(x, cpu) \
void left_shift_##x##_##cpu(const void *src, void *dst, unsigned shift, unsigned left, unsigned right)
#define DECLARE_DEPTH_CONVERT(x, cpu) \
void depth_convert_##x##_##cpu(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right)

DECLARE_LEFT_SHIFT(b2b, sse2);
DECLARE_LEFT_SHIFT(b2w, sse2);
DECLARE_LEFT_SHIFT(w2b, sse2);
DECLARE_LEFT_SHIFT(w2w, sse2);
DECLARE_LEFT_SHIFT(b2b, avx2);
DECLARE_LEFT_SHIFT(b2w, avx2);
DECLARE_LEFT_SHIFT(w2b, avx2);
DECLARE_LEFT_SHIFT(w2w, avx2);
DECLARE_LEFT_SHIFT(b2b, avx512);
DECLARE_LEFT_SHIFT(b2w, avx512);
DECLARE_LEFT_SHIFT(w2b, avx512);
DECLARE_LEFT_SHIFT(w2w, avx512);

DECLARE_DEPTH_CONVERT(b2f, sse2);
DECLARE_DEPTH_CONVERT(w2f, sse2);
DECLARE_DEPTH_CONVERT(b2h, avx2);
DECLARE_DEPTH_CONVERT(b2f, avx2);
DECLARE_DEPTH_CONVERT(w2h, avx2);
DECLARE_DEPTH_CONVERT(w2f, avx2);
DECLARE_DEPTH_CONVERT(b2h, avx512);
DECLARE_DEPTH_CONVERT(b2f, avx512);
DECLARE_DEPTH_CONVERT(w2h, avx512);
DECLARE_DEPTH_CONVERT(w2f, avx512);

#undef DECLARE_LEFT_SHIFT
#undef DECLARE_DEPTH_CONVERT

left_shift_func select_left_shift_func_x86(PixelType pixel_in, PixelType pixel_out, CPUClass cpu);

depth_convert_func select_depth_convert_func_x86(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

depth_f16c_func select_depth_f16c_func_x86(bool to_half, CPUClass cpu);

bool needs_depth_f16c_func_x86(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

} // namespace zimg::depth

#endif // ZIMG_DEPTH_X86_DEPTH_CONVERT_X86_H_

#endif // ZIMG_X86
