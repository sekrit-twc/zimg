#include "Common/except.h"
#include "Common/pixel.h"
#include "depth2.h"
#include "depth_convert2.h"
#include "dither2.h"

namespace zimg {;
namespace depth {;

Depth2::Depth2(DitherType type, unsigned width, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
try
{
	if (pixel_out.type == PixelType::HALF || pixel_out.type == PixelType::FLOAT)
		m_impl.reset(new DepthConvert2{ pixel_in, pixel_out, cpu });
	else
		m_impl.reset(create_dither_convert2(type, width, pixel_in, pixel_out, cpu));
} catch (const std::bad_alloc &) {
	throw ZimgOutOfMemory{};
}

zimg_filter_flags Depth2::get_flags() const
{
	return m_impl->get_flags();
}

ZimgFilter::pair_unsigned Depth2::get_required_row_range(unsigned i) const
{
	return m_impl->get_required_row_range(i);
}

ZimgFilter::pair_unsigned Depth2::get_required_col_range(unsigned left, unsigned right) const
{
	return m_impl->get_required_col_range(left, right);
}

unsigned Depth2::get_simultaneous_lines() const
{
	return m_impl->get_simultaneous_lines();
}

unsigned Depth2::get_max_buffering() const
{
	return m_impl->get_max_buffering();
}

size_t Depth2::get_context_size() const
{
	return m_impl->get_context_size();
}

size_t Depth2::get_tmp_size(unsigned left, unsigned right) const
{
	return m_impl->get_tmp_size(left, right);
}

void Depth2::init_context(void *ctx) const
{
	m_impl->init_context(ctx);
}

void Depth2::process(void *ctx, const zimg_image_buffer src[3], const zimg_image_buffer dst[3], void *tmp, unsigned i, unsigned left, unsigned right) const
{
	m_impl->process(ctx, src, dst, tmp, i, left, right);
}

} // namespace depth
} // namespace zimg
