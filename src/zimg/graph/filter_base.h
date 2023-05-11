#pragma once

#ifndef ZIMG_GRAPH_FILTER_BASE_H_
#define ZIMG_GRAPH_FILTER_BASE_H_

#include "graphengine/filter.h"

namespace zimg {
enum class PixelType;
}

namespace zimg::graph {

class FilterBase : public graphengine::Filter {
protected:
	graphengine::FilterDescriptor m_desc{};
public:
	int version() const noexcept override final { return VERSION; }

	const graphengine::FilterDescriptor &descriptor() const noexcept override final { return m_desc; }

	void init_context(void *) const noexcept override {}
};

class PointFilter : public FilterBase {
protected:
	// Initializes |format| and |step| fields of descriptor.
	PointFilter(unsigned width, unsigned height, PixelType type);

	pair_unsigned get_row_deps(unsigned i) const noexcept override final { return{ i, i + 1 }; }

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override final { return{ left, right }; }
};

} // namespace zimg::graph

#endif // ZIMG_GRAPH_FILTER_BASE_H_
