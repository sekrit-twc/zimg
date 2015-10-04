#pragma once

#ifndef ZIMG_DEPTH_DEPTH_CONVERT_H_
#define ZIMG_DEPTH_DEPTH_CONVERT_H_

namespace zimg {;

struct PixelFormat;

enum class PixelType;
enum class CPUClass;

namespace graph {;

class ImageFilter;

} // namespace graph


namespace depth {;

typedef void (*left_shift_func)(const void *src, void *dst, unsigned shift, unsigned left, unsigned right);
typedef void (*depth_convert_func)(const void *src, void *dst, float scale, float offset, unsigned left, unsigned right);
typedef void (*depth_f16c_func)(const void *src, void *dst, unsigned left, unsigned right);

graph::ImageFilter *create_left_shift(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

graph::ImageFilter *create_convert_to_float(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DEPTH_CONVERT_H_
