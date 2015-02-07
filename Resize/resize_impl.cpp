#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "Common/plane.h"
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

class ResizeImplC_H final : public ResizeImpl {
public:
	ResizeImplC_H(const EvaluatedFilter &filter) : ResizeImpl(filter)
	{}

	void process_u16(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		filter_plane_h_scalar(m_filter, src, dst, 0, src.height(), 0, m_filter.height(), ScalarPolicy_U16{});
	}

	void process_f16(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in C impl" };
	}

	void process_f32(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const override
	{
		filter_plane_h_scalar(m_filter, src, dst, 0, src.height(), 0, m_filter.height(), ScalarPolicy_F32{});
	}
};

class ResizeImplC_V final : public ResizeImpl {
public:
	ResizeImplC_V(const EvaluatedFilter &filter) : ResizeImpl(filter)
	{}

	void process_u16(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		filter_plane_v_scalar(m_filter, src, dst, 0, m_filter.height(), 0, src.width(), ScalarPolicy_U16{});
	}

	void process_f16(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in C impl" };
	}

	void process_f32(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const override
	{
		filter_plane_v_scalar(m_filter, src, dst, 0, m_filter.height(), 0, src.width(), ScalarPolicy_F32{});
	}
};

ResizeImpl *create_resize_impl_c(const EvaluatedFilter &filter, bool horizontal)
{
	if (horizontal)
		return new ResizeImplC_H{ filter };
	else
		return new ResizeImplC_V{ filter };
}

} // namespace


ResizeImpl::ResizeImpl(const EvaluatedFilter &filter) : m_filter{ filter }
{
}

ResizeImpl::~ResizeImpl()
{
}

ResizeImpl *create_resize_impl(const Filter &f, bool horizontal, int src_dim, int dst_dim, double shift, double subwidth, CPUClass cpu)
{
	ResizeImpl *ret = nullptr;

	if (src_dim != dst_dim || shift != 0.0 || subwidth != src_dim) {
		EvaluatedFilter filter = compute_filter(f, src_dim, dst_dim, shift, subwidth);

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
