#pragma once

#ifndef ZIMG_GRAPH_COPY_FILTER_H_
#define ZIMG_GRAPH_COPY_FILTER_H_

#include <algorithm>
#include <cstdint>
#include "common/pixel.h"
#include "image_filter.h"

namespace zimg {;
namespace graph {;

class CopyFilter : public ImageFilterBase {
	image_attributes m_attr;
public:
	CopyFilter(unsigned width, unsigned height, PixelType type) :
		m_attr{ width, height, type }
	{
	}

	filter_flags get_flags() const override
	{
		filter_flags flags{};

		flags.same_row = 1;
		flags.in_place = 1;

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return m_attr;
	}

	void process(void *, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *, unsigned i, unsigned left, unsigned right) const override
	{
		const uint8_t *src_p = static_buffer_cast<const uint8_t>(*src)[i];
		uint8_t *dst_p = static_buffer_cast<uint8_t>(*dst)[i];

		left *= pixel_size(m_attr.type);
		right *= pixel_size(m_attr.type);

		std::copy(src_p + left, src_p + right, dst_p + left);
	}
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_COPY_FILTER_H_
