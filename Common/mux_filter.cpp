#include <algorithm>
#include "align.h"
#include "alloc.h"
#include "except.h"
#include "mux_filter.h"

namespace zimg{;

MuxFilter::MuxFilter(IZimgFilter *filter, IZimgFilter *filter_uv) :
	m_flags{}
{
	ZimgFilterFlags filter_flags = filter->get_flags();
	ZimgFilterFlags filter_flags_uv = filter_uv ? filter_uv->get_flags() : filter_flags;
	ZimgFilterFlags flags{};

	if (filter_flags.color || filter_flags_uv.color)
		throw zimg::error::InternalError{ "can not mux color filters" };

	if (filter_uv) {
		unsigned simultaneous_lines = filter->get_simultaneous_lines();
		image_attributes attr = filter->get_image_attributes();

		if (filter_uv->get_image_attributes() != attr)
			throw zimg::error::InternalError{ "can not mux filters with differing output formats" };

		if (filter_uv->get_simultaneous_lines() != simultaneous_lines)
			throw zimg::error::InternalError{ "UV filter must produce the same number of lines" };

		for (unsigned i = 0; i < attr.height; i += simultaneous_lines) {
			auto range = filter->get_required_row_range(i);
			auto range_uv = filter_uv->get_required_row_range(i);

			if (range != range_uv)
				throw zimg::error::InternalError{ "UV filter must operate on the same row range" };
		}
		for (unsigned j = 0; j < attr.width; ++j) {
			auto range = filter->get_required_col_range(j, j + 1);
			auto range_uv = filter->get_required_col_range(j, j + 1);

			if (range != range_uv)
				throw zimg::error::InternalError{ "UV filter must operate on the same column range" };
		}
	}

	m_flags.has_state = filter_flags.has_state || filter_flags_uv.has_state;
	m_flags.same_row = filter_flags.same_row && filter_flags_uv.same_row;
	m_flags.in_place = filter_flags.in_place && filter_flags_uv.in_place;
	m_flags.entire_row = filter_flags.entire_row || filter_flags_uv.entire_row;
	m_flags.entire_plane = filter_flags.entire_plane || filter_flags_uv.entire_plane;
	m_flags.color = true;

	m_filter.reset(filter);
	m_filter_uv.reset(filter_uv);
}

ZimgFilterFlags MuxFilter::get_flags() const
{
	return m_flags;
}

IZimgFilter::image_attributes MuxFilter::get_image_attributes() const
{
	return m_filter->get_image_attributes();
}

IZimgFilter::pair_unsigned MuxFilter::get_required_row_range(unsigned i) const
{
	return m_filter->get_required_row_range(i);
}

IZimgFilter::pair_unsigned MuxFilter::get_required_col_range(unsigned left, unsigned right) const
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
	size_t context_size = m_filter->get_context_size();
	size_t context_size_uv = m_filter_uv ? m_filter_uv->get_context_size() : context_size;

	return align(context_size, ALIGNMENT) + 2 * align(context_size_uv, ALIGNMENT);
}

size_t MuxFilter::get_tmp_size(unsigned left, unsigned right) const
{
	size_t tmp_size = m_filter->get_tmp_size(left, right);
	size_t tmp_size_uv = m_filter_uv ? m_filter_uv->get_tmp_size(left, right) : tmp_size;

	return std::max(tmp_size, tmp_size_uv);
}

void MuxFilter::init_context(void *ctx) const
{
	LinearAllocator alloc{ ctx };
	size_t context_size = m_filter->get_context_size();
	size_t context_size_uv = m_filter_uv ? m_filter_uv->get_context_size() : context_size;
	void *ptr;

	ptr = alloc.allocate(context_size);
	m_filter->init_context(ptr);

	ptr = alloc.allocate(context_size_uv);
	(m_filter_uv ? m_filter_uv : m_filter)->init_context(ptr);

	ptr = alloc.allocate(context_size_uv);
	(m_filter_uv ? m_filter_uv : m_filter)->init_context(ptr);
}

void MuxFilter::process(void *ctx, const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *tmp, unsigned i, unsigned left, unsigned right) const
{
	LinearAllocator alloc{ ctx };
	size_t context_size = m_filter->get_context_size();
	size_t context_size_uv = m_filter_uv ? m_filter_uv->get_context_size() : context_size;
	void *context[3];

	context[0] = alloc.allocate(context_size);
	context[1] = alloc.allocate(context_size_uv);
	context[2] = alloc.allocate(context_size_uv);

	for (unsigned p = 0; p < 3; ++p) {
		ZimgImageBufferConst src_p{};
		ZimgImageBuffer dst_p{};
		IZimgFilter *filter;

		src_p.data[0] = src.data[p];
		src_p.stride[0] = src.stride[p];
		src_p.mask[0] = src.mask[p];

		dst_p.data[0] = dst.data[p];
		dst_p.stride[0] = dst.stride[p];
		dst_p.mask[0] = dst.mask[p];

		filter = ((p == 1 || p == 2) && m_filter_uv) ? m_filter_uv.get() : m_filter.get();
		filter->process(context[p], src_p, dst_p, tmp, i, left, right);
	}
}

} // namespace zimg
