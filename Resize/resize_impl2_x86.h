#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_RESIZE_RESIZE_IMPL2_X86_H_
#define ZIMG_RESIZE_RESIZE_IMPL2_X86_H_

namespace zimg {;

enum class CPUClass;
enum class PixelType;

class IZimgFilter;

namespace resize {;

struct FilterContext;

IZimgFilter *create_resize_impl2_h_x86(const FilterContext &context, unsigned height, PixelType type, unsigned depth, CPUClass cpu);

IZimgFilter *create_resize_impl2_v_x86(const FilterContext &context, unsigned width, PixelType type, unsigned depth, CPUClass cpu);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_IMPL2_X86_H_

#endif // ZIMG_X86
