#pragma once

#ifndef ZIMG_COLORSPACE_COLORSPACE_H_
#define ZIMG_COLORSPACE_COLORSPACE_H_

namespace zimg {;

enum class CPUClass;

namespace graph {;

class ImageFilter;

} // namespace graph


namespace colorspace {;

struct ColorspaceDefinition;

graph::ImageFilter *create_colorspace(unsigned width, unsigned height, const ColorspaceDefinition &in, const ColorspaceDefinition &out, CPUClass cpu);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_COLORSPACE2_H_
