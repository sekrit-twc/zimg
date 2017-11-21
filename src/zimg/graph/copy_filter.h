#pragma once

#ifndef ZIMG_GRAPH_COPY_FILTER_H_
#define ZIMG_GRAPH_COPY_FILTER_H_

#include "image_filter.h"

namespace zimg {

enum class PixelType;

namespace graph {

// Copies a greyscale image.
class CopyFilter : public ImageFilterBase {
	image_attributes m_attr;
	bool m_color;
public:
	CopyFilter(unsigned width, unsigned height, PixelType type, bool color = false);

	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	void process(void *ctx, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_COPY_FILTER_H_
