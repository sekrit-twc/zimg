#pragma once

#ifndef ZIMG_RESIZE_RESIZE_IMPL_H_
#define ZIMG_RESIZE_RESIZE_IMPL_H_

#include <memory>
#include "graph/image_filter.h"
#include "filter.h"

namespace zimg {

enum class CPUClass;
enum class PixelType;

namespace resize {

class ResizeImplH : public graph::ImageFilterBase {
protected:
	FilterContext m_filter;
	image_attributes m_attr;
	bool m_is_sorted;

	ResizeImplH(const FilterContext &filter, const image_attributes &attr);
public:
	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override;

	unsigned get_max_buffering() const override;
};

class ResizeImplV : public graph::ImageFilterBase {
protected:
	FilterContext m_filter;
	image_attributes m_attr;
	bool m_is_sorted;

	ResizeImplV(const FilterContext &filter, const image_attributes &attr);
public:
	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	unsigned get_max_buffering() const override;
};

struct ResizeImplBuilder {
	unsigned src_width;
	unsigned src_height;
	PixelType type;

#include "common/builder.h"
	BUILDER_MEMBER(bool, horizontal)
	BUILDER_MEMBER(unsigned, dst_dim)
	BUILDER_MEMBER(unsigned, depth)
	BUILDER_MEMBER(const Filter *, filter)
	BUILDER_MEMBER(double, shift)
	BUILDER_MEMBER(double, subwidth)
	BUILDER_MEMBER(CPUClass, cpu)
#undef BUILDER_MEMBER

	ResizeImplBuilder(unsigned src_width, unsigned src_height, PixelType type);

	std::unique_ptr<graph::ImageFilter> create() const;
};

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_IMPL_H_
