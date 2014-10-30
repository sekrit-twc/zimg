#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_
#define ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_

namespace zimg {;

enum class CPUClass;

namespace colorspace {;

class PixelAdapter;

PixelAdapter *create_pixel_adapter_f16c();

/**
 * Create an appropriate x86 optimized PixelAdapter for the given CPU.
 */
PixelAdapter *create_pixel_adapter_x86(CPUClass cpu);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_

#endif // ZIMG_X86
