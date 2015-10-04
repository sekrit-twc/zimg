#include <algorithm>
#include <memory>
#include "graph/copy_filter.h"
#include "unresize.h"
#include "unresize_impl.h"

namespace zimg {;
namespace unresize {;

namespace {;

bool unresize_h_first(double xscale, double yscale)
{
	double h_first_cost = std::max(xscale, 1.0) * 2.0 + xscale * std::max(yscale, 1.0);
	double v_first_cost = std::max(yscale, 1.0) + yscale * std::max(xscale, 1.0) * 2.0;

	return h_first_cost < v_first_cost;
}

} // namespace


std::pair<graph::ImageFilter *, graph::ImageFilter *> create_unresize(
	PixelType type, unsigned src_width, unsigned src_height, unsigned dst_width, unsigned dst_height,
	double shift_w, double shift_h, CPUClass cpu)
{
	bool skip_h = (src_width == dst_width && shift_w == 0);
	bool skip_v = (src_height == dst_height && shift_h == 0);

	if (skip_h && skip_v) {
		return{ new graph::CopyFilter{ src_width, src_height, type }, nullptr };
	} else if (skip_h) {
		return{ create_unresize_impl(type, false, src_width, src_height, dst_width, dst_height, shift_w, cpu), nullptr };
	} else if (skip_v) {
		return{ create_unresize_impl(type, true, src_width, src_height, dst_width, dst_height, shift_h, cpu), nullptr };
	} else {
		bool h_first = unresize_h_first((double)dst_width / src_width, (double)dst_height / src_height);
		std::unique_ptr<graph::ImageFilter> stage1;
		std::unique_ptr<graph::ImageFilter> stage2;

		if (h_first) {
			stage1.reset(create_unresize_impl(type, true, src_width, src_height, dst_width, src_height, shift_w, cpu));
			stage2.reset(create_unresize_impl(type, false, dst_width, src_height, dst_width, dst_height, shift_h, cpu));
		} else {
			stage1.reset(create_unresize_impl(type, false, src_width, src_height, src_width, dst_height, shift_h, cpu));
			stage2.reset(create_unresize_impl(type, true, src_width, dst_height, dst_width, dst_height, shift_w, cpu));
		}

		return{ stage1.release(), stage2.release() };
	}
}

} // namespace unresize
} // namespace zimg

#if 0
#include <algorithm>
#include "common/align.h"
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/pixel.h"
#include "plane.h"
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
} catch (const std::bad_alloc &) {
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

void Unresize::invoke_impl_h(const ImagePlane<const void> &src, const ImagePlane<void> &dst, void *tmp) const
{
	switch (src.format().type) {
	case PixelType::HALF:
		m_impl->process_f16_h(plane_cast<const uint16_t>(src), plane_cast<uint16_t>(dst), (uint16_t *)tmp);
		break;
	case PixelType::FLOAT:
		m_impl->process_f32_h(plane_cast<const float>(src), plane_cast<float>(dst), (float *)tmp);
		break;
	default:
		throw error::UnsupportedOperation{ "only HALF and FLOAT supported for unresize" };
	}
}

void Unresize::invoke_impl_v(const ImagePlane<const void> &src, const ImagePlane<void> &dst, void *tmp) const
{
	switch (src.format().type) {
	case PixelType::HALF:
		m_impl->process_f16_v(plane_cast<const uint16_t>(src), plane_cast<uint16_t>(dst), (uint16_t *)tmp);
		break;
	case PixelType::FLOAT:
		m_impl->process_f32_v(plane_cast<const float>(src), plane_cast<float>(dst), (float *)tmp);
		break;
	default:
		throw error::UnsupportedOperation{ "only HALF and FLOAT supported for unresize" };
	}
}

size_t Unresize::tmp_size(PixelType type) const
{
	size_t size = 0;

	// Temporary image.
	if (m_src_width != m_dst_width && m_src_height != m_dst_height)
		size += max_frame_size(type);

	// Line buffer for horizontal pass.
	if (m_src_width != m_dst_width)
		size += (size_t)m_dst_width * 8;

	return size;
}

void Unresize::process(const ImagePlane<const void> &src, const ImagePlane<void> &dst, void *tmp) const
{
	PixelType type = src.format().type;
	int pxsize = pixel_size(type);

	if (m_src_width == m_dst_width) {
		invoke_impl_v(src, dst, tmp);
	} else if (m_src_height == m_dst_height) {
		invoke_impl_h(src, dst, tmp);
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
			ImagePlane<void> tmp_plane{ tmp1, m_dst_width, m_src_height, tmp_stride, type };

			invoke_impl_h(src, tmp_plane, tmp2);
			invoke_impl_v(tmp_plane, dst, tmp2);
		} else {
			int tmp_stride = align(m_src_width, ALIGNMENT / pxsize);
			ImagePlane<void> tmp_plane{ tmp1, m_src_width, m_dst_height, tmp_stride, type };

			invoke_impl_v(src, tmp_plane, tmp2);
			invoke_impl_h(tmp_plane, dst, tmp2);
		}
	}
}

} // namespace unresize
} // namespace zimg
#endif