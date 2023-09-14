#pragma once

#ifndef ZIMG_RESIZE_RESIZE_IMPL_H_
#define ZIMG_RESIZE_RESIZE_IMPL_H_

#include <memory>
#include <utility>
#include "graph/filter_base.h"
#include "filter.h"

namespace zimg {
enum class CPUClass;
enum class PixelType;
}

namespace zimg::resize {

class ResizeImplH : public graph::FilterBase {
protected:
	FilterContext m_filter;

	ResizeImplH(const FilterContext &filter, unsigned height, PixelType type);
public:
	pair_unsigned get_row_deps(unsigned i) const noexcept override;

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override;

	void init_context(void *) const noexcept override {}
};

class ResizeImplV : public graph::FilterBase {
protected:
	FilterContext m_filter;
	bool m_unsorted;

	ResizeImplV(const FilterContext &filter, unsigned width, PixelType type);
public:
	pair_unsigned get_row_deps(unsigned i) const noexcept override;

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override;

	void init_context(void *) const noexcept override {}
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

	std::unique_ptr<graphengine::Filter> create() const;
};

} // namespace zimg::resize

#endif // ZIMG_RESIZE_RESIZE_IMPL_H_
