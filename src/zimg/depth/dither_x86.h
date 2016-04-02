#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_DITHER_X86_H_
#define ZIMG_DEPTH_DITHER_X86_H_

#include "dither.h"

namespace zimg {
namespace depth {

#define DECLARE_ORDERED_DITHER(x, cpu) \
void ordered_dither_##x##_##cpu(const float *dither, unsigned dither_offset, unsigned dither_mask, \
                                const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)

DECLARE_ORDERED_DITHER(b2b, sse2);
DECLARE_ORDERED_DITHER(b2w, sse2);
DECLARE_ORDERED_DITHER(w2b, sse2);
DECLARE_ORDERED_DITHER(w2w, sse2);
DECLARE_ORDERED_DITHER(f2b, sse2);
DECLARE_ORDERED_DITHER(f2w, sse2);

DECLARE_ORDERED_DITHER(b2b, avx2);
DECLARE_ORDERED_DITHER(b2w, avx2);
DECLARE_ORDERED_DITHER(w2b, avx2);
DECLARE_ORDERED_DITHER(w2w, avx2);
DECLARE_ORDERED_DITHER(h2b, avx2);
DECLARE_ORDERED_DITHER(h2w, avx2);
DECLARE_ORDERED_DITHER(f2b, avx2);
DECLARE_ORDERED_DITHER(f2w, avx2);

#undef DECLARE_ORDERED_DITHER

dither_convert_func select_ordered_dither_func_x86(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

dither_f16c_func select_dither_f16c_func_x86(CPUClass cpu);

bool needs_dither_f16c_func_x86(CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DITHER_X86_H_

#endif // ZIMG_X86
