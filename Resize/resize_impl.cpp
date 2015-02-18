#include <algorithm>
#include <climits>
#include "Common/align.h"
#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "Common/pixel.h"
#include "Common/plane.h"
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

class ResizeImplC_H : public ResizeImpl {
public:
	ResizeImplC_H(const FilterContext &filter) : ResizeImpl(filter, true)
	{}

	void process_u16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
		filter_line_h_scalar(m_filter, src, dst, n, n + 1, dst.left(), dst.right(), ScalarPolicy_U16{});
	}

	void process_f16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in C impl" };
	}

	void process_f32(const LineBuffer<float> &src, LineBuffer<float> &dst, unsigned n, void *tmp) const override
	{
		filter_line_h_scalar(m_filter, src, dst, n, n + 1, dst.left(), dst.right(), ScalarPolicy_F32{});
	}
};

class ResizeImplC_V : public ResizeImpl {
public:
	ResizeImplC_V(const FilterContext &filter) : ResizeImpl(filter, false)
	{}

	void process_u16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
		filter_line_v_scalar(m_filter, src, dst, n, n + 1, dst.left(), dst.right(), ScalarPolicy_U16{});
	}

	void process_f16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in C impl" };
	}

	void process_f32(const LineBuffer<float> &src, LineBuffer<float> &dst, unsigned n, void *tmp) const override
	{
		filter_line_v_scalar(m_filter, src, dst, n, n + 1, dst.left(), dst.right(), ScalarPolicy_F32{});
	}
};

ResizeImpl *create_resize_impl_c(const FilterContext &filter, bool horizontal)
{
	if (horizontal)
		return new ResizeImplC_H{ filter };
	else
		return new ResizeImplC_V{ filter };
}

} // namespace


ResizeImpl::ResizeImpl(const FilterContext &filter, bool horizontal) : m_horizontal{ horizontal }, m_filter(filter)
{
}

ResizeImpl::~ResizeImpl()
{
}

bool ResizeImpl::pixel_supported(PixelType type) const
{
	return type != PixelType::HALF;
}

size_t ResizeImpl::tmp_size(PixelType type, unsigned width) const
{
	return 0;
}

unsigned ResizeImpl::input_buffering(PixelType type) const
{
	if (m_horizontal)
		return output_buffering(type);
	else if (std::is_sorted(m_filter.left.begin(), m_filter.left.end()))
		return m_filter.filter_width;
	else
		return UINT_MAX;
}

unsigned ResizeImpl::output_buffering(PixelType type) const
{
	return 1;
}

unsigned ResizeImpl::dependent_line(unsigned n) const
{
	return m_horizontal ? n : m_filter.left[n];
}

ResizeImpl *create_resize_impl(const Filter &f, bool horizontal, int src_dim, int dst_dim, double shift, double subwidth, CPUClass cpu)
{
	ResizeImpl *ret = nullptr;

	if (src_dim != dst_dim || shift != 0.0 || subwidth != src_dim) {
		FilterContext filter = compute_filter(f, src_dim, dst_dim, shift, subwidth);
#ifdef ZIMG_X86
		ret = create_resize_impl_x86(filter, horizontal, cpu);
#endif
		if (!ret)
			ret = create_resize_impl_c(filter, horizontal);
	}

	return ret;
}

} // namespace resize
} // namespace zimg
