#include <algorithm>
#include "resize.h"
#include "resize_impl.h"

namespace resize {;

Resize::Resize(const Filter &f, int src_width, int src_height, int dst_width, int dst_height,
               double shift_w, double shift_h, double subwidth, double subheight, bool x86) :
	m_src_width{ src_width },
	m_src_height{ src_height },
	m_dst_width{ dst_width },
	m_dst_height{ dst_height },
	m_skip_h{ src_width == dst_width && shift_w == 0.0 && subwidth == src_width },
	m_skip_v{ src_height == dst_height && shift_h == 0.0 && subheight == src_height },
	m_impl{ create_resize_impl(f, src_width, src_height, dst_width, dst_height, shift_w, shift_h, subwidth, subheight, x86) }
{
}

size_t Resize::max_frame_size(PixelType type) const
{
	int alignment = type == PixelType::FLOAT ? AlignmentOf<float>::value : AlignmentOf<uint16_t>::value;

	size_t aligned_src_width = align(m_src_width, alignment);
	size_t aligned_src_height = align(m_src_height, alignment);
	size_t aligned_dst_width = align(m_dst_width, alignment);
	size_t aligned_dst_height = align(m_dst_height, alignment);

	return std::max(aligned_src_width, aligned_dst_width) * std::max(aligned_src_height, aligned_dst_height);
}

size_t Resize::tmp_size(PixelType type) const
{
	size_t size = 0;

	// Need temporary buffer to hold the partially scaled image.
	if (!m_skip_h && !m_skip_v)
		size += max_frame_size(type);

	// Need a line buffer to store cached accumulators.
	if (type == PixelType::WORD && !m_skip_v)
		size += m_src_width * 2;

	return size;
}

void Resize::copy_plane(const void *src, void *dst, int src_stride_bytes, int dst_stride_bytes) const
{
	const char *src_byteptr = (const char *)src;
	char *dst_byteptr = (char *)dst;

	for (int i = 0; i < m_dst_height; ++i) {
		std::copy(src_byteptr + i * src_stride_bytes, src_byteptr + i * src_stride_bytes + m_src_width, dst_byteptr + i * dst_stride_bytes);
	}
}

void Resize::process_u16(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp, int src_stride, int dst_stride) const
{
	if (m_skip_h && m_skip_v) {
		copy_plane(src, dst, src_stride * sizeof(uint16_t), dst_stride * sizeof(uint16_t));
	} else if (m_skip_h) {
		m_impl->process_u16_v(src, dst, tmp, m_src_width, m_src_height, src_stride, dst_stride);
	} else if (m_skip_v) {
		m_impl->process_u16_h(src, dst, tmp, m_src_width, m_src_height, src_stride, dst_stride);
	} else {
		double xscale = (double)m_dst_width / (double)m_src_width;
		double yscale = (double)m_dst_height / (double)m_src_height;

		uint16_t *tmp1 = tmp;
		uint16_t *tmp2 = tmp + max_frame_size(PixelType::WORD);

		// First execute the pass that results in the fewest pixels.
		if ((xscale < 1.0 && xscale * 2.0 < yscale) || (xscale >= 1.0 && xscale < yscale)) {
			int tmp_stride = align(m_dst_width, AlignmentOf<uint16_t>::value);

			m_impl->process_u16_h(src, tmp1, tmp2, m_src_width, m_src_height, src_stride, tmp_stride);
			m_impl->process_u16_v(tmp1, dst, tmp2, m_dst_width, m_src_height, tmp_stride, dst_stride);
		} else {
			int tmp_stride = align(m_src_width, AlignmentOf<uint16_t>::value);

			m_impl->process_u16_v(src, tmp1, tmp2, m_src_width, m_src_height, src_stride, tmp_stride);
			m_impl->process_u16_h(tmp1, dst, tmp2, m_src_width, m_dst_height, tmp_stride, dst_stride);
		}
	}
}

void Resize::process_f16(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp, int src_stride, int dst_stride) const
{
	if (m_skip_h && m_skip_v) {
		copy_plane(src, dst, src_stride * sizeof(uint16_t), dst_stride * sizeof(uint16_t));
	} else if (m_skip_h) {
		m_impl->process_f16_v(src, dst, tmp, m_src_width, m_src_height, src_stride, dst_stride);
	} else if (m_skip_v) {
		m_impl->process_f16_h(src, dst, tmp, m_src_width, m_src_height, src_stride, dst_stride);
	} else {
		double xscale = (double)m_dst_width / (double)m_src_width;
		double yscale = (double)m_dst_height / (double)m_src_height;

		uint16_t *tmp1 = tmp;
		uint16_t *tmp2 = tmp + max_frame_size(PixelType::WORD);

		// First execute the pass that results in the fewest pixels.
		if ((xscale < 1.0 && xscale * 2.0 < yscale) || (xscale >= 1.0 && xscale < yscale)) {
			int tmp_stride = align(m_dst_width, AlignmentOf<uint16_t>::value);

			m_impl->process_f16_h(src, tmp1, tmp2, m_src_width, m_src_height, src_stride, tmp_stride);
			m_impl->process_f16_v(tmp1, dst, tmp2, m_dst_width, m_src_height, tmp_stride, dst_stride);
		} else {
			int tmp_stride = align(m_src_width, AlignmentOf<uint16_t>::value);

			m_impl->process_f16_v(src, tmp1, tmp2, m_src_width, m_src_height, src_stride, tmp_stride);
			m_impl->process_f16_h(tmp1, dst, tmp2, m_src_width, m_dst_height, tmp_stride, dst_stride);
		}
	}
}

void Resize::process_f32(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp, int src_stride, int dst_stride) const
{
	if (m_skip_h && m_skip_v) {
		copy_plane(src, dst, src_stride * sizeof(float), dst_stride * sizeof(float));
	} else if (m_skip_h) {
		m_impl->process_f32_v(src, dst, tmp, m_src_width, m_src_height, src_stride, dst_stride);
	} else if (m_skip_v) {
		m_impl->process_f32_h(src, dst, tmp, m_src_width, m_src_height, src_stride, dst_stride);
	} else {
		double xscale = (double)m_dst_width / (double)m_src_width;
		double yscale = (double)m_dst_height / (double)m_src_height;

		// First execute the pass that results in the fewest pixels.
		if ((xscale < 1.0 && xscale * 2.0 < yscale) || (xscale >= 1.0 && xscale < yscale)) {
			int tmp_stride = align(m_dst_width, AlignmentOf<float>::value);

			m_impl->process_f32_h(src, tmp, nullptr, m_src_width, m_src_height, src_stride, tmp_stride);
			m_impl->process_f32_v(tmp, dst, nullptr, m_dst_width, m_src_height, tmp_stride, dst_stride);
		} else {
			int tmp_stride = align(m_src_width, AlignmentOf<float>::value);

			m_impl->process_f32_v(src, tmp, nullptr, m_src_width, m_src_height, src_stride, tmp_stride);
			m_impl->process_f32_h(tmp, dst, nullptr, m_src_width, m_dst_height, tmp_stride, dst_stride);
		}
	}
}

} // namespace resize
