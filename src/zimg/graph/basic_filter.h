#pragma once

#ifndef ZIMG_GRAPH_BASIC_FILTER_H_
#define ZIMG_GRAPH_BASIC_FILTER_H_

#include <cstdint>
#include "graphengine/filter.h"

namespace zimg {

enum class PixelType;

namespace graph {

// Copies an image buffer.
class CopyFilter : public graphengine::Filter {
	graphengine::FilterDescriptor m_desc;
public:
	CopyFilter(unsigned width, unsigned height, PixelType type);

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	std::pair<unsigned, unsigned> get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	std::pair<unsigned, unsigned> get_col_deps(unsigned left, unsigned right) const noexcept override { return{ left, right }; }

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override;
};

// Initializes a plane to a constant value.
class ValueInitializeFilter : public graphengine::Filter {
public:
	union value_type {
		uint8_t b;
		uint16_t w;
		float f;
	};
private:
	graphengine::FilterDescriptor m_desc;
	value_type m_value;

	void fill_b(void *ptr, size_t n) const;
	void fill_w(void *ptr, size_t n) const;
	void fill_f(void *ptr, size_t n) const;
public:
	ValueInitializeFilter(unsigned width, unsigned height, PixelType type, value_type val);

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	std::pair<unsigned, unsigned> get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	std::pair<unsigned, unsigned> get_col_deps(unsigned left, unsigned right) const noexcept override { return{ left, right }; }

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override;
};

// Premultiplies an image.
class PremultiplyFilter : public graphengine::Filter {
	graphengine::FilterDescriptor m_desc;
public:
	PremultiplyFilter(unsigned width, unsigned height);

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	std::pair<unsigned, unsigned> get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	std::pair<unsigned, unsigned> get_col_deps(unsigned left, unsigned right) const noexcept override { return{ left, right }; }

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor in[2], const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override;
};

// Unpremultiplies an image.
class UnpremultiplyFilter : public graphengine::Filter {
	graphengine::FilterDescriptor m_desc;
public:
	UnpremultiplyFilter(unsigned width, unsigned height);

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	std::pair<unsigned, unsigned> get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	std::pair<unsigned, unsigned> get_col_deps(unsigned left, unsigned right) const noexcept override { return{ left, right }; }

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor in[2], const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override;
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_BASIC_FILTER_H_
