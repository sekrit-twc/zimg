#pragma once

#ifndef ZIMG_GRAPH_ZFILTER_H_
#define ZIMG_GRAPH_ZFILTER_H_

#include <cstddef>
#include <limits>
#include <utility>
#include "common/align.h"
#include "buffer.h"

namespace zimg {;

enum class PixelType;

namespace graph {;

class ImageFilter {
public:
	struct filter_flags {
		bool has_state : 1;
		bool same_row : 1;
		bool in_place : 1;
		bool entire_row : 1;
		bool entire_plane : 1;
		bool color : 1;
	};

	struct image_attributes {
		unsigned width;
		unsigned height;
		PixelType type;
	};

	typedef std::pair<unsigned, unsigned> pair_unsigned;

	virtual inline ~ImageFilter() = 0;

	virtual filter_flags get_flags() const = 0;

	virtual image_attributes get_image_attributes() const = 0;

	virtual pair_unsigned get_required_row_range(unsigned i) const = 0;

	virtual pair_unsigned get_required_col_range(unsigned left, unsigned right) const = 0;

	virtual unsigned get_simultaneous_lines() const = 0;

	virtual unsigned get_max_buffering() const = 0;

	virtual size_t get_context_size() const = 0;

	virtual size_t get_tmp_size(unsigned left, unsigned right) const = 0;

	virtual void init_context(void *ctx) const = 0;

	virtual void process(void *ctx, const ImageBufferConst &src, const ImageBuffer &dst, void *tmp, unsigned i, unsigned left, unsigned right) const = 0;
};

class ImageFilterBase : public ImageFilter {
public:
	virtual inline ~ImageFilterBase() = 0;

	pair_unsigned get_required_row_range(unsigned i) const override
	{
		return{ i, i + 1 };
	}

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override
	{
		return{ left, right };
	}

	unsigned get_simultaneous_lines() const override
	{
		return 1;
	}

	unsigned get_max_buffering() const override
	{
		return 1;
	}

	size_t get_context_size() const override
	{
		return 0;
	}

	size_t get_tmp_size(unsigned, unsigned) const override
	{
		return 0;
	}

	void init_context(void *ctx) const override
	{
	}
};

ImageFilter::~ImageFilter()
{
}

ImageFilterBase::~ImageFilterBase()
{
}

inline bool operator==(const ImageFilter::image_attributes &a, const ImageFilter::image_attributes &b)
{
	return a.width == b.width && a.height == b.height && a.type == b.type;
}

inline bool operator!=(const ImageFilter::image_attributes &a, const ImageFilter::image_attributes &b)
{
	return !operator==(a, b);
}

inline unsigned select_zimg_buffer_mask(unsigned count)
{
	const unsigned UINT_BITS = std::numeric_limits<unsigned>::digits;

	if (count != 0 && ((count - 1) & (1 << (UINT_BITS - 1))))
		return -1;

	for (unsigned i = UINT_BITS - 1; i != 0; --i) {
		if ((count - 1) & (1 << (i - 1)))
			return (1 << i) - 1;
	}

	return 0;
}

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_ZFILTER_H_
