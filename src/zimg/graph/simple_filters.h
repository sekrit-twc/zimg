#pragma once

#ifndef ZIMG_GRAPH_SIMPLE_FILTERS_H_
#define ZIMG_GRAPH_SIMPLE_FILTERS_H_

#include <cstdint>
#include "graphengine/filter.h"
#include "filter_base.h"

namespace zimg {
enum class PixelType;
}

namespace zimg::graph{

// Copies a subrectangle.
class CopyRectFilter : public graph::FilterBase {
	unsigned m_left;
	unsigned m_top;
public:
	CopyRectFilter(unsigned left, unsigned top, unsigned width, unsigned height, PixelType type);

	pair_unsigned get_row_deps(unsigned i) const noexcept override { return{ m_top + i, m_top + i + 1 }; }

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override { return{ m_left + left, m_left + right }; }

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override;
};

// Initializes a plane to a constant value.
class ValueInitializeFilter : public PointFilter {
public:
	union value_type {
		uint8_t b;
		uint16_t w;
		float f;
	};
private:
	value_type m_value;

	void fill_b(void *ptr, size_t n) const;
	void fill_w(void *ptr, size_t n) const;
	void fill_f(void *ptr, size_t n) const;
public:
	ValueInitializeFilter(unsigned width, unsigned height, PixelType type, value_type val);

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override;
};

// Premultiplies an image.
class PremultiplyFilter : public PointFilter {
public:
	PremultiplyFilter(unsigned width, unsigned height);

	void process(const graphengine::BufferDescriptor in[2], const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override;
};

// Unpremultiplies an image.
class UnpremultiplyFilter : public PointFilter {
public:
	UnpremultiplyFilter(unsigned width, unsigned height);

	void process(const graphengine::BufferDescriptor in[2], const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override;
};

} // namespace zimg::graph

#endif // ZIMG_GRAPH_SIMPLE_FILTERS_H_
