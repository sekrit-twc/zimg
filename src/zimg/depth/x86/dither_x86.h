#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_X86_DITHER_X86_H_
#define ZIMG_DEPTH_X86_DITHER_X86_H_

#include <memory>
#include "depth/dither.h"

namespace graphengine {
class Filter;
}

namespace zimg::depth {

#define DECLARE_ORDERED_DITHER(x, cpu) \
void ordered_dither_##x##_##cpu(const float *dither, unsigned dither_offset, unsigned dither_mask, \
                                const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right);

DECLARE_ORDERED_DITHER(b2b, avx2)
DECLARE_ORDERED_DITHER(b2w, avx2)
DECLARE_ORDERED_DITHER(w2b, avx2)
DECLARE_ORDERED_DITHER(w2w, avx2)
DECLARE_ORDERED_DITHER(h2b, avx2)
DECLARE_ORDERED_DITHER(h2w, avx2)
DECLARE_ORDERED_DITHER(f2b, avx2)
DECLARE_ORDERED_DITHER(f2w, avx2)

DECLARE_ORDERED_DITHER(b2b, avx512)
DECLARE_ORDERED_DITHER(b2w, avx512)
DECLARE_ORDERED_DITHER(w2b, avx512)
DECLARE_ORDERED_DITHER(w2w, avx512)
DECLARE_ORDERED_DITHER(h2b, avx512)
DECLARE_ORDERED_DITHER(h2w, avx512)
DECLARE_ORDERED_DITHER(f2b, avx512)
DECLARE_ORDERED_DITHER(f2w, avx512)

#undef DECLARE_ORDERED_DITHER

dither_convert_func select_ordered_dither_func_x86(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

std::unique_ptr<graphengine::Filter> create_error_diffusion_avx2(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out);

std::unique_ptr<graphengine::Filter> create_error_diffusion_x86(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

} // namespace zimg::depth

#endif // ZIMG_DEPTH_X86_DITHER_X86_H_

#endif // ZIMG_X86
