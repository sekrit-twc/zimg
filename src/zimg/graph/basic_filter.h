#pragma once

#ifndef ZIMG_GRAPH_BASIC_FILTER_H_
#define ZIMG_GRAPH_BASIC_FILTER_H_

#include <memory>
#include "image_filter.h"

namespace zimg {

enum class PixelType;

namespace graph {

// Copies a greyscale image.
class CopyFilter : public ImageFilterBase {
	image_attributes m_attr;
public:
	CopyFilter(unsigned width, unsigned height, PixelType type);

	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	void process(void *ctx, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

// Applies a greyscale filter to a color image.
class MuxFilter : public ImageFilter {
	std::unique_ptr<ImageFilter> m_filter;
public:
	MuxFilter(std::unique_ptr<ImageFilter> &&filter);

	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override;

	unsigned get_simultaneous_lines() const override;

	unsigned get_max_buffering() const override;

	size_t get_context_size() const override;

	size_t get_tmp_size(unsigned left, unsigned right) const override;

	void init_context(void *ctx) const override;

	void process(void *ctx, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_BASIC_FILTER_H_
