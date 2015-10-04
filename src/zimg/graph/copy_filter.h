#pragma once

#ifndef ZIMG_GRAPH_COPY_FILTER_H_
#define ZIMG_GRAPH_COPY_FILTER_H_

#include "common/linebuffer.h"
#include "common/pixel.h"
#include "zfilter.h"

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

	void process(void *, const ImageBufferConst &src, const ImageBuffer &dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		LineBuffer<const void> src_buf{ src };
		LineBuffer<void> dst_buf{ dst };

		unsigned byte_left = left * pixel_size(m_attr.type);
		unsigned byte_right = right * pixel_size(m_attr.type);

		copy_buffer_lines(src_buf, dst_buf, i, i + 1, byte_left, byte_right);
	}
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_COPY_FILTER_H_
