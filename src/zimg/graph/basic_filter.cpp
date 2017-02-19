#include <memory>
#include <stdexcept>
#include "common/alloc.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/pixel.h"
#include "basic_filter.h"

namespace zimg {
namespace graph {

CopyFilter::CopyFilter(unsigned width, unsigned height, PixelType type) :
	m_attr{ width, height, type }
{
}

auto CopyFilter::get_flags() const -> filter_flags
{
	filter_flags flags{};

	flags.same_row = true;
	flags.in_place = true;

	return flags;
}

auto CopyFilter::get_image_attributes() const -> image_attributes
{
	return m_attr;
}

void CopyFilter::process(void *, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *, unsigned i, unsigned left, unsigned right) const
{
	const uint8_t *src_p = static_buffer_cast<const uint8_t>(*src)[i];
	uint8_t *dst_p = static_buffer_cast<uint8_t>(*dst)[i];

	unsigned left_bytes = left * pixel_size(m_attr.type);
	unsigned right_bytes = right * pixel_size(m_attr.type);

	std::copy(src_p + left_bytes, src_p + right_bytes, dst_p + left_bytes);
}

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
	try {
		checked_size_t filter_ctx_size = m_filter->get_context_size();
		checked_size_t ctx_size = zimg::ceil_n(filter_ctx_size, ALIGNMENT) * 3;
		return ctx_size.get();
	} catch (const std::overflow_error &) {
		throw error::OutOfMemory{};
	}
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
