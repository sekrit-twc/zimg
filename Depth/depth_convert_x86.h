#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_DEPTH_CONVERT_X86_H_
#define ZIMG_DEPTH_DEPTH_CONVERT_X86_H_

namespace zimg {;

enum class CPUClass;

namespace depth {;

class DepthConvert;

DepthConvert *create_depth_convert_sse2();
DepthConvert *create_depth_convert_avx2();

DepthConvert *create_depth_convert_x86(CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DEPTH_CONVERT_X86_H_

#endif // ZIMG_X86
