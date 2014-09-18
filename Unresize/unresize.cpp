#include <algorithm>
#include "Common/align.h"
#include "Common/except.h"
#include "unresize.h"
#include "unresize_impl.h"

namespace zimg {;
namespace unresize {;

Unresize::Unresize(int src_width, int src_height, int dst_width, int dst_height, float shift_w, float shift_h, bool x86) try :
	m_src_width(src_width), m_src_height(src_height), m_dst_width(dst_width), m_dst_height(dst_height),
	m_impl(create_unresize_impl(src_width, src_height, dst_width, dst_height, shift_w, shift_h, x86))
{
}
catch (const std::bad_alloc &) {
	throw ZimgOutOfMemory{};
}

Unresize::~Unresize() {}

size_t Unresize::tmp_size(PixelType src, PixelType dst) const
{
	int aligned_src_width = align(m_src_width, 8);
	int aligned_src_height = align(m_src_height, 8);

	int aligned_dst_width = align(m_dst_width, 8);
	int aligned_dst_height = align(m_dst_height, 8);

	size_t size = 0;

	if (src != PixelType::FLOAT)
		size += aligned_src_width * m_src_height;
	if (dst != PixelType::FLOAT)
		size += aligned_dst_width * m_dst_height;
	
	// Temporary image.
	size += 2 * std::max(aligned_dst_width * m_src_height, aligned_src_height * m_dst_width);
	// Line buffer for scanline_4.
	size += 4 * std::max(aligned_dst_height, aligned_dst_width);

	return size;
}

void Unresize::process(const uint8_t *src, uint8_t *dst, float *tmp, int src_stride, int dst_stride, PixelType src_type, PixelType dst_type) const
{
	float *dst_buf;
	float *src_buf;
	int dst_buf_stride;
	int src_buf_stride;

	if (src_type == PixelType::FLOAT) {
		src_buf = (float *)src;
		src_buf_stride = src_stride;
	} else {
		src_buf = (float *)tmp;
		src_buf_stride = align(m_src_width, 8);
		tmp += src_buf_stride * m_src_height;
	}

	if (dst_type == PixelType::FLOAT) {
		dst_buf = (float *)dst;
		dst_buf_stride = dst_stride;
	} else {
		dst_buf = (float *)tmp;
		dst_buf_stride = align(m_dst_width, 8);
		tmp += dst_buf_stride * m_dst_height;
	}

	// Load if necessary.
	if (src_type == PixelType::BYTE) {
		for (int i = 0; i < m_src_height; ++i) {
			m_impl->load_scanline_u8(&src[i * src_stride], &src_buf[i * src_buf_stride], m_src_width);
		}
	} else if (src_type == PixelType::WORD) {
		for (int i = 0; i < m_src_height; ++i) {
			m_impl->load_scanline_u16(&((uint16_t *)src)[i * src_stride], &src_buf[i * src_buf_stride], m_src_width);
		}
	}

	process(src_buf, dst_buf, tmp, src_buf_stride, dst_buf_stride);

	// Store if necessary.
	if (dst_type == PixelType::BYTE) {
		for (int i = 0; i < m_dst_height; ++i) {
			m_impl->store_scanline_u8(&dst_buf[i * dst_buf_stride], &dst[i * dst_stride], m_dst_width);
		}
	} else if (dst_type == PixelType::WORD) {
		for (int i = 0; i < m_dst_height; ++i) {
			m_impl->store_scanline_u16(&dst_buf[i * dst_buf_stride], &((uint16_t *)dst)[i * dst_stride], m_dst_width);
		}
	}
}

void Unresize::process(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const
{
	int aligned_src_width = align(m_src_width, 8);
	int aligned_src_height = align(m_src_height, 8);

	int aligned_dst_width = align(m_dst_width, 8);
	int aligned_dst_height = align(m_dst_height, 8);

	float *tmp_buf_1;
	float *tmp_buf_2;
	int tmp_buf_size = std::max(aligned_dst_width * m_src_height, aligned_src_height * m_dst_width);

	tmp_buf_1 = tmp;
	tmp += tmp_buf_size;
	tmp_buf_2 = tmp;
	tmp += tmp_buf_size;

	if (m_dst_width != m_src_width) {
		float *dst_p = tmp_buf_1;
		const float *src_p = src;
		
		int dst_p_stride = aligned_dst_width;
		int src_p_stride = src_stride;

		// Horizontal transform.
		for (int i = 0; i < mod(m_src_height, 4); i += 4) {
			m_impl->unresize_scanline4_h(src_p, dst_p, tmp, src_p_stride, dst_p_stride);

			dst_p += dst_p_stride * 4;
			src_p += src_p_stride * 4;
		}
		for (int i = mod(m_src_height, 4); i < m_src_height; ++i) {
			m_impl->unresize_scanline(src_p, dst_p, tmp, true);

			dst_p += dst_p_stride;
			src_p += src_p_stride;
		}

		// Transpose.
		m_impl->transpose_plane(tmp_buf_1, tmp_buf_2, m_dst_width, m_src_height, aligned_dst_width, aligned_src_height);
	} else {
		// Transpose the source to tmp_buf_2, where the vertical transform will look.
		m_impl->transpose_plane(src, tmp_buf_2, m_src_width, m_src_height, src_stride, aligned_src_height);
	}

	if (m_dst_height != m_src_height) {
		float *dst_p = tmp_buf_1;
		const float *src_p = tmp_buf_2;

		int dst_p_stride = aligned_dst_height;
		int src_p_stride = aligned_src_height;

		// Vertical transform.
		for (int i = 0; i < mod(m_dst_width, 4); i += 4) {
			m_impl->unresize_scanline4_v(src_p, dst_p, tmp, src_p_stride, dst_p_stride);

			dst_p += dst_p_stride * 4;
			src_p += src_p_stride * 4;
		}
		for (int i = mod(m_dst_width, 4); i < m_dst_width; ++i) {
			m_impl->unresize_scanline(src_p, dst_p, tmp, false);

			dst_p += dst_p_stride;
			src_p += src_p_stride;
		}

		// Transpose.
		m_impl->transpose_plane(tmp_buf_1, dst, m_dst_height, m_dst_width, aligned_dst_height, dst_stride);
	} else {
		// Transpose tmp_buf_2 where the horizontal transform stored its result.
		m_impl->transpose_plane(tmp_buf_2, dst, m_src_height, m_dst_width, aligned_src_height, dst_stride);
	}
}

} // namespace unresize
} // namespace zimg
