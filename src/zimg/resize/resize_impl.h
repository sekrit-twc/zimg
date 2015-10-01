#pragma once

#ifndef ZIMG_RESIZE_RESIZE_IMPL2_H_
#define ZIMG_RESIZE_RESIZE_IMPL2_H_

#include "common/zfilter.h"
#include "filter.h"

namespace zimg {;

enum class CPUClass;
enum class PixelType;

namespace resize {;

class ResizeImplH : public ZimgFilter {
protected:
	FilterContext m_filter;
	image_attributes m_attr;
	bool m_is_sorted;

	ResizeImplH(const FilterContext &filter, const image_attributes &attr);
public:
	ZimgFilterFlags get_flags() const override;

	image_attributes get_image_attributes() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override;

	unsigned get_max_buffering() const override;
};

class ResizeImplV : public ZimgFilter {
protected:
	FilterContext m_filter;
	image_attributes m_attr;
	bool m_is_sorted;

	ResizeImplV(const FilterContext &filter, const image_attributes &attr);
public:
	ZimgFilterFlags get_flags() const override;

	image_attributes get_image_attributes() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	unsigned get_max_buffering() const override;
};

IZimgFilter *create_resize_impl(const Filter &f, PixelType type, bool horizontal, unsigned depth, unsigned src_width, unsigned src_height, unsigned dst_width, unsigned dst_height,
                                double shift, double subwidth, CPUClass cpu);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_IMPL2_H_
