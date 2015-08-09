#pragma once

#ifndef ZIMG_RESIZE_RESIZE_IMPL2_H_
#define ZIMG_RESIZE_RESIZE_IMPL2_H_

namespace zimg {;

enum class CPUClass;
enum class PixelType;

class IZimgFilter;

namespace resize {;

class Filter;

IZimgFilter *create_resize_impl2(const Filter &f, PixelType type, bool horizontal, unsigned src_dim, unsigned dst_dim, unsigned height,
                                 double shift, double subwidth, CPUClass cpu);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_IMPL2_H_
