#pragma once

#ifndef ZIMG_RESIZE_RESIZE_IMPL_H_
#define ZIMG_RESIZE_RESIZE_IMPL_H_

#include <memory>
#include <utility>
#include "graph/image_filter.h"
#include "graphengine/filter.h"
#include "filter.h"

namespace zimg {

enum class CPUClass;
enum class PixelType;

namespace resize {

class ResizeImplH_GE : public graphengine::Filter {
protected:
	graphengine::FilterDescriptor m_desc;
	FilterContext m_filter;

	ResizeImplH_GE(const FilterContext &filter, unsigned height, PixelType type);
public:
	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	std::pair<unsigned, unsigned> get_row_deps(unsigned i) const noexcept override;

	std::pair<unsigned, unsigned> get_col_deps(unsigned left, unsigned right) const noexcept override;

	void init_context(void *) const noexcept override {}
};

class ResizeImplV_GE : public graphengine::Filter {
protected:
	graphengine::FilterDescriptor m_desc;
	FilterContext m_filter;
	bool m_unsorted;

	ResizeImplV_GE(const FilterContext &filter, unsigned width, PixelType type);
public:
	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	std::pair<unsigned, unsigned> get_row_deps(unsigned i) const noexcept override;

	std::pair<unsigned, unsigned> get_col_deps(unsigned left, unsigned right) const noexcept override;

	void init_context(void *) const noexcept override {}
};

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

	std::unique_ptr<graphengine::Filter> create_ge() const;
};

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_IMPL_H_
