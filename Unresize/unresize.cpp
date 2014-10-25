#include <algorithm>
#include "Common/align.h"
#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "Common/pixel.h"
#include "unresize.h"
#include "unresize_impl.h"

namespace zimg {;
namespace unresize {;

Unresize::Unresize(int src_width, int src_height, int dst_width, int dst_height, float shift_w, float shift_h, CPUClass cpu) try :
	m_src_width(src_width),
	m_src_height(src_height),
	m_dst_width(dst_width),
	m_dst_height(dst_height),
	m_impl(create_unresize_impl(src_width, src_height, dst_width, dst_height, shift_w, shift_h, cpu))
{
}
catch (const std::bad_alloc &) {
	throw ZimgOutOfMemory{};
}

Unresize::~Unresize() {}

size_t Unresize::max_frame_size(PixelType type) const
{
	int alignment = type == PixelType::FLOAT ? AlignmentOf<float>::value : AlignmentOf<uint16_t>::value;

	size_t aligned_src_width = align(m_src_width, alignment);
	size_t aligned_src_height = align(m_src_height, alignment);
	size_t aligned_dst_width = align(m_dst_width, alignment);
	size_t aligned_dst_height = align(m_dst_height, alignment);

	return std::max(aligned_src_width, aligned_dst_width) * std::max(aligned_src_height, aligned_dst_height);
}

size_t Unresize::tmp_size(PixelType type) const
{
	size_t size = 0;
	
	// Temporary image.
	if (m_src_width != m_dst_width && m_src_height != m_dst_height)
		size += max_frame_size(type);

	// Line buffer for horizontal pass.
	if (m_src_width != m_dst_width)
		size += m_dst_width * 4;

	return size;
}

void Unresize::process(PixelType type, const void *src, void *dst, void *tmp, int src_stride, int dst_stride) const
{
	int pxsize = pixel_size(type);

	if (type != PixelType::FLOAT)
		throw ZimgUnsupportedError{ "only f32 supported" };

	if (m_src_width == m_dst_width) {
		m_impl->process_f32_v((const float *)src, (float *)dst, (float *)tmp, m_src_width, m_src_height, src_stride, dst_stride);
	} else if (m_src_height == m_dst_height) {
		m_impl->process_f32_h((const float *)src, (float *)dst, (float *)tmp, m_src_width, m_src_height, src_stride, dst_stride);
	} else {
		double xscale = (double)m_dst_width / (double)m_src_width;
		double yscale = (double)m_dst_height / (double)m_src_height;

		// Downscaling cost is proportional to input size, whereas upscaling cost is proportional to output size.
		// Horizontal operation is roughly twice as costly as vertical operation for SIMD cores.
		double h_first_cost = std::max(xscale, 1.0) * 2.0 + xscale * std::max(yscale, 1.0);
		double v_first_cost = std::max(yscale, 1.0)       + yscale * std::max(xscale, 1.0) * 2.0;

		char *tmp1 = (char *)tmp;
		char *tmp2 = tmp1 + max_frame_size(type) * pxsize;

		if (h_first_cost < v_first_cost) {
			int tmp_stride = align(m_dst_width, ALIGNMENT / pxsize);

			m_impl->process_f32_h((const float *)src, (float *)tmp1, (float *)tmp2, m_src_width, m_src_height, src_stride, tmp_stride);
			m_impl->process_f32_v((const float *)tmp1, (float *)dst, (float *)tmp2, m_dst_width, m_src_height, tmp_stride, dst_stride);
		} else {
			int tmp_stride = align(m_src_width, ALIGNMENT / pxsize);

			m_impl->process_f32_v((const float *)src, (float *)tmp1, (float *)tmp2, m_src_width, m_src_height, src_stride, tmp_stride);
			m_impl->process_f32_h((const float *)tmp1, (float *)dst, (float *)tmp2, m_src_width, m_dst_height, tmp_stride, dst_stride);
		}
	}
}

} // namespace unresize
} // namespace zimg
