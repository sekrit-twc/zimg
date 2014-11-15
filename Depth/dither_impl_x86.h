#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_DITHER_IMPL_X86_H_
#define ZIMG_DEPTH_DITHER_IMPL_X86_H_

namespace zimg {;

enum class CPUClass;

namespace depth {;

class DitherConvert;

DitherConvert *create_ordered_dither_sse2(const float *dither);
DitherConvert *create_ordered_dither_avx2(const float *dither);

DitherConvert *create_ordered_dither_x86(const float *dither, CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DITHER_IMPL_X86_H_
#endif // ZIMG_X86
