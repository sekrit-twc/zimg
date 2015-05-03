#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "Common/alloc.h"
#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "Common/linebuffer.h"
#include "Common/pixel.h"
#include "Common/plane.h"
#include "resize.h"
#include "resize_impl.h"

namespace zimg {;
namespace resize {;

namespace {;

bool resize_h_first(double xscale, double yscale)
{
	double h_first_cost = std::max(xscale, 1.0) * 2.0 + xscale * std::max(yscale, 1.0);
	double v_first_cost = std::max(yscale, 1.0) + yscale * std::max(xscale, 1.0) * 2.0;

	return h_first_cost < v_first_cost;
}

unsigned ceillog2(unsigned x)
{
	unsigned n = std::numeric_limits<unsigned>::digits;

	for (; n != 0; --n) {
		if (x & (1U << (n - 1)))
			break;
	}

	return (1U << (n - 1)) < x ? n : n - 1;
}

template <class Alloc>
LineBuffer<void> alloc_line_buffer(Alloc &alloc, unsigned count, unsigned width, PixelType type)
{
	unsigned log2count = ceillog2(count);
	unsigned stride = align(width * pixel_size(type), ALIGNMENT);

	unsigned pow2count = log2count >= std::numeric_limits<unsigned>::digits ? UINT_MAX : 1U << log2count;
	unsigned mask = pow2count == UINT_MAX ? UINT_MAX : pow2count - 1;
	void *ptr = alloc.allocate((size_t)pow2count * stride);

	return{ ptr, width, stride, mask };
}

void invoke_impl(const ResizeImpl *impl, PixelType type, const LineBuffer<void> &src, LineBuffer<void> &dst, unsigned n, void *tmp)
{
	switch (type) {
	case PixelType::BYTE:
		throw ZimgUnsupportedError{ "BYTE not supported for resize" };
	case PixelType::WORD:
		impl->process_u16(buffer_cast<uint16_t>(src), buffer_cast<uint16_t>(dst), n, tmp);
		break;
	case PixelType::HALF:
		impl->process_f16(buffer_cast<uint16_t>(src), buffer_cast<uint16_t>(dst), n, tmp);
		break;
	case PixelType::FLOAT:
		impl->process_f32(buffer_cast<float>(src), buffer_cast<float>(dst), n, tmp);
		break;
	default:
		break;
	}
}

} // namespace


struct ResizeContext {
	LineBuffer<void> src_border_buf;
	LineBuffer<void> dst_border_buf;
	LineBuffer<void> tmp_buf;
	void *tmp_data;

	ResizeImpl *impl1;
	ResizeImpl *impl2;

	unsigned tmp_width;
	unsigned tmp_height;

	unsigned in_buffering1;
	unsigned in_buffering2;

	unsigned out_buffering1;
	unsigned out_buffering2;
};

Resize::Resize(const Filter &f, int src_width, int src_height, int dst_width, int dst_height,
               double shift_w, double shift_h, double subwidth, double subheight, CPUClass cpu)
try :
	m_src_width{ (unsigned)src_width },
	m_src_height{ (unsigned)src_height },
	m_dst_width{ (unsigned)dst_width },
	m_dst_height{ (unsigned)dst_height },
	m_skip_h{ src_width == dst_width && shift_w == 0.0 && subwidth == src_width },
	m_skip_v{ src_height == dst_height && shift_h == 0.0 && subheight == src_height },
	m_impl_h{ create_resize_impl(f, true, src_width, dst_width, shift_w, subwidth, cpu) },
	m_impl_v{ create_resize_impl(f, false, src_height, dst_height, shift_h, subheight, cpu) }
{
} catch (const std::bad_alloc &) {
	throw ZimgOutOfMemory{};
}

template <class Alloc>
ResizeContext Resize::get_context(Alloc &alloc, PixelType type) const
{
	ResizeContext ctx{};

	if (!m_skip_h && !m_skip_v) {
		double xscale = (double)m_dst_width / (double)m_src_width;
		double yscale = (double)m_dst_height / (double)m_src_height;
		bool h_first = resize_h_first(xscale, yscale);

		ctx.impl1 = h_first ? m_impl_h.get() : m_impl_v.get();
		ctx.impl2 = h_first ? m_impl_v.get() : m_impl_h.get();

		ctx.tmp_width = h_first ? m_dst_width : m_src_width;
		ctx.tmp_height = h_first ? m_src_height : m_dst_height;

		ctx.in_buffering1 = std::min(ctx.impl1->input_buffering(type), m_src_height);
		ctx.in_buffering2 = std::min(ctx.impl2->input_buffering(type), ctx.tmp_height);

		ctx.out_buffering1 = ctx.impl1->output_buffering(type);
		ctx.out_buffering2 = ctx.impl2->output_buffering(type);

		ctx.src_border_buf = alloc_line_buffer(alloc, ctx.in_buffering1, m_src_width, type);
		ctx.dst_border_buf = alloc_line_buffer(alloc, ctx.out_buffering2, m_dst_width, type);
		ctx.tmp_buf = alloc_line_buffer(alloc, ctx.out_buffering1 + ctx.in_buffering2 - 1, ctx.tmp_width, type);

		ctx.tmp_data = alloc.allocate(std::max(ctx.impl1->tmp_size(type, ctx.tmp_width), ctx.impl2->tmp_size(type, m_dst_width)));
	} else if (!(m_skip_h && m_skip_v)) {
		ctx.impl1 = m_impl_h ? m_impl_h.get() : m_impl_v.get();

		ctx.in_buffering1 = std::min(ctx.impl1->input_buffering(type), m_src_height);
		ctx.out_buffering1 = ctx.impl1->output_buffering(type);

		ctx.src_border_buf = alloc_line_buffer(alloc, ctx.in_buffering1, m_src_width, type);
		ctx.dst_border_buf = alloc_line_buffer(alloc, ctx.out_buffering1, m_dst_width, type);
		ctx.tmp_data = alloc.allocate(ctx.impl1->tmp_size(type, m_dst_width));
	}

	return ctx;
}

size_t Resize::tmp_size(PixelType type) const
{
	FakeAllocator alloc;
	get_context(alloc, type);
	return alloc.count();
}

void Resize::process1d(const ImagePlane<const void> &src, const ImagePlane<void> &dst, void *tmp) const
{
	PixelType type = src.format().type;
	LinearAllocator alloc{ tmp };
	ResizeContext ctx = get_context(alloc, type);

	LineBuffer<void> src_buf{ (void *)src.data(), (unsigned)src.width(), (unsigned)src.stride() * pixel_size(type), UINT_MAX };
	LineBuffer<void> dst_buf{ dst.data(), (unsigned)dst.width(), (unsigned)dst.stride() * pixel_size(type), UINT_MAX };

	bool overflow_flag = false;
	unsigned src_linesize = m_src_width * pixel_size(type);
	unsigned dst_linesize = m_dst_width * pixel_size(type);

	for (unsigned i = 0; i < m_dst_height; i += ctx.out_buffering1) {
		const LineBuffer<void> *in_buf = &src_buf;
		LineBuffer<void> *out_buf = &dst_buf;

		unsigned dep_first = ctx.impl1->dependent_line(i);
		unsigned dep_last = dep_first + ctx.in_buffering1;

		if (dep_last > m_src_height) {
			if (!overflow_flag) {
				copy_buffer_lines(src_buf, ctx.src_border_buf, src_linesize, dep_first, m_src_height);
				overflow_flag = true;
			}
			in_buf = &ctx.src_border_buf;
		}

		if (i + ctx.out_buffering1 > m_dst_height)
			out_buf = &ctx.dst_border_buf;

		invoke_impl(ctx.impl1, type, *in_buf, *out_buf, i, ctx.tmp_data);

		if (i + ctx.out_buffering1 > m_dst_height)
			copy_buffer_lines(ctx.dst_border_buf, dst_buf, dst_linesize, i, m_dst_height);
	}
}

void Resize::process2d(const ImagePlane<const void> &src, const ImagePlane<void> &dst, void *tmp) const
{
	PixelType type = src.format().type;
	LinearAllocator alloc{ tmp };
	ResizeContext ctx = get_context(alloc, type);

	LineBuffer<void> src_buf{ (void *)src.data(), (unsigned)src.width(), (unsigned)src.stride() * pixel_size(type), UINT_MAX };
	LineBuffer<void> dst_buf{ dst.data(), (unsigned)dst.width(), (unsigned)dst.stride() * pixel_size(type), UINT_MAX };

	bool overflow_flag = false;
	unsigned buffer_pos = 0;
	unsigned src_linesize = m_src_width * pixel_size(type);
	unsigned dst_linesize = m_dst_width * pixel_size(type);

	for (unsigned i = 0; i < m_dst_height; i += ctx.out_buffering2) {
		const LineBuffer<void> *in_buf = &src_buf;
		LineBuffer<void> *out_buf = &dst_buf;

		unsigned dep2_first = ctx.impl2->dependent_line(i);
		unsigned dep2_last = std::min(dep2_first + ctx.in_buffering2, ctx.tmp_height);

		for (; buffer_pos < dep2_last; buffer_pos += ctx.out_buffering1) {
			unsigned dep1_first = ctx.impl1->dependent_line(buffer_pos);
			unsigned dep1_last = dep1_first + ctx.in_buffering1;

			if (dep1_last > m_src_height) {
				if (!overflow_flag) {
					copy_buffer_lines(src_buf, ctx.src_border_buf, src_linesize, dep1_first, m_src_height);
					overflow_flag = true;
				}
				in_buf = &ctx.src_border_buf;
			}

			invoke_impl(ctx.impl1, type, *in_buf, ctx.tmp_buf, buffer_pos, ctx.tmp_data);
		}

		if (i + ctx.out_buffering2 > m_dst_height)
			out_buf = &ctx.dst_border_buf;

		invoke_impl(ctx.impl2, type, ctx.tmp_buf, *out_buf, i, ctx.tmp_data);

		if (i + ctx.out_buffering2 > m_dst_height)
			copy_buffer_lines(ctx.dst_border_buf, dst_buf, dst_linesize, i, m_dst_height);
	}
}

void Resize::process(const ImagePlane<const void> &src, const ImagePlane<void> &dst, void *tmp) const
{
	if (m_skip_h && m_skip_v)
		copy_image_plane(src, dst);
	else if (m_skip_h || m_skip_v)
		process1d(src, dst, tmp);
	else
		process2d(src, dst, tmp);
}

} // namespace resize
} // namespace zimg
