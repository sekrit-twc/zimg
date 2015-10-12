#include <algorithm>
#include <utility>
#include "common/align.h"
#include "common/alloc.h"
#include "common/except.h"
#include "mux_filter.h"

namespace zimg{;
namespace graph {;

MuxFilter::MuxFilter(std::unique_ptr<ImageFilter> &&filter)
{
	if (filter->get_flags().color)
		throw error::InternalError{ "can not mux color filter" };

	m_filter = std::move(filter);
}

ImageFilter::filter_flags MuxFilter::get_flags() const
{
	filter_flags flags = m_filter->get_flags();

	flags.color = true;

	return flags;
}

ImageFilter::image_attributes MuxFilter::get_image_attributes() const
{
	return m_filter->get_image_attributes();
}

ImageFilter::pair_unsigned MuxFilter::get_required_row_range(unsigned i) const
{
	return m_filter->get_required_row_range(i);
}

ImageFilter::pair_unsigned MuxFilter::get_required_col_range(unsigned left, unsigned right) const
{
	return m_filter->get_required_col_range(left, right);
}

unsigned MuxFilter::get_simultaneous_lines() const
{
	return m_filter->get_simultaneous_lines();
}

unsigned MuxFilter::get_max_buffering() const
{
	return m_filter->get_max_buffering();
}

size_t MuxFilter::get_context_size() const
{
	return zimg::ceil_n(m_filter->get_context_size(), ALIGNMENT) * 3;
}

size_t MuxFilter::get_tmp_size(unsigned left, unsigned right) const
{
	return m_filter->get_tmp_size(left, right);
}

void MuxFilter::init_context(void *ctx) const
{
	LinearAllocator alloc{ ctx };
	size_t context_size = m_filter->get_context_size();

	for (unsigned p = 0; p < 3; ++p) {
		void *ptr = alloc.allocate(context_size);
		m_filter->init_context(ptr);
	}
}

void MuxFilter::process(void *ctx, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *tmp, unsigned i, unsigned left, unsigned right) const
{
	LinearAllocator alloc{ ctx };
	size_t context_size = m_filter->get_context_size();

	for (unsigned p = 0; p < 3; ++p) {
		void *ptr = alloc.allocate(context_size);
		m_filter->process(ptr, src + p, dst + p, tmp, i, left, right);
	}
}

} // namespace graph
} // namespace zimg
