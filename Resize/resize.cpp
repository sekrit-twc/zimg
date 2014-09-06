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
	m_skip_v{ src_height == dst_height && shift_h == 0.0 && subheight == src_height }
{
	if (m_skip_h && m_skip_v)
		throw std::domain_error{ "no-op filter" };

	m_impl.reset(create_resize_impl(f, src_width, src_height, dst_width, dst_height, shift_w, shift_h, subwidth, subheight, x86));
}

Resize::~Resize()
{
}

size_t Resize::tmp_size() const
{
	size_t size = 0;

	size_t aligned_src_width = align(m_src_width, 8);
	size_t aligned_src_height = align(m_src_height, 8);
	size_t aligned_dst_width = align(m_dst_width, 8);
	size_t aligned_dst_height = align(m_dst_height, 8);

	// Need one temporary buffer to hold the partially resized image.
	size += std::max(aligned_src_width, aligned_dst_width) * std::max(aligned_src_height, aligned_dst_height);

	// Need a second buffer to hold the transpose.
	if (!m_skip_h && !m_skip_v)
		size *= 3;
	else
		size *= 2;

	return size;
}

void Resize::process(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp, int src_stride, int dst_stride) const
{
	if (m_skip_h) {
		m_impl->process_v(src, dst, tmp, m_src_width, m_src_height, src_stride, dst_stride);
	} else if (m_skip_v) {
		m_impl->process_h(src, dst, tmp, m_src_width, m_src_height, src_stride, dst_stride);
	} else {
		double xscale = (double)m_dst_width / (double)m_src_width;
		double yscale = (double)m_dst_height / (double)m_src_height;

		float *tmp1 = tmp;
		float *tmp2 = tmp + tmp_size() / 3;

		// First execute the pass that results in the fewest pixels.
		if (xscale < yscale) {
			int tmp_stride = align(m_dst_width, 8);

			m_impl->process_h(src, tmp1, tmp2, m_src_width, m_src_height, src_stride, tmp_stride);
			m_impl->process_v(tmp1, dst, tmp2, m_dst_width, m_src_height, tmp_stride, dst_stride);
		} else {
			int tmp_stride = align(m_src_width, 8);

			m_impl->process_v(src, tmp1, tmp2, m_src_width, m_src_height, src_stride, tmp_stride);
			m_impl->process_h(tmp1, dst, tmp2, m_src_width, m_dst_height, tmp_stride, dst_stride);
		}
	}
}

} // namespace resize
