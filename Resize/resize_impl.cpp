#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "Common/plane.h"
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

class ResizeImplC final : public ResizeImpl {
public:
	ResizeImplC(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) : ResizeImpl(filter_h, filter_v)
	{}

	void process_u16_h(const ImagePlane<uint16_t> &src, ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		const EvaluatedFilter &filter = m_filter_h;
		filter_plane_h_scalar(filter, src, dst, 0, src.height(), 0, filter.height(), ScalarPolicy_U16{});
	}

	void process_u16_v(const ImagePlane<uint16_t> &src, ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		const EvaluatedFilter &filter = m_filter_v;
		filter_plane_v_scalar(filter, src, dst, 0, filter.height(), 0, src.width(), ScalarPolicy_U16{});
	}

	void process_f16_h(const ImagePlane<uint16_t> &src, ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in C impl" };
	}

	void process_f16_v(const ImagePlane<uint16_t> &src, ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in C impl" };
	}

	void process_f32_h(const ImagePlane<float> &src, ImagePlane<float> &dst, float *tmp) const override
	{
		const EvaluatedFilter &filter = m_filter_h;
		filter_plane_h_scalar(filter, src, dst, 0, src.height(), 0, filter.height(), ScalarPolicy_F32{});
	}

	void process_f32_v(const ImagePlane<float> &src, ImagePlane<float> &dst, float *tmp) const override
	{
		const EvaluatedFilter &filter = m_filter_v;
		filter_plane_v_scalar(filter, src, dst, 0, filter.height(), 0, src.width(), ScalarPolicy_F32{});
	}
};

} // namespace


ResizeImpl::ResizeImpl(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v)
try :
	m_filter_h{ filter_h },
	m_filter_v{ filter_v }
{
} catch (const std::bad_alloc &) {
	throw ZimgOutOfMemory{};
}

ResizeImpl::~ResizeImpl()
{
}

ResizeImpl *create_resize_impl(const Filter &f, int src_width, int src_height, int dst_width, int dst_height,
                               double shift_w, double shift_h, double subwidth, double subheight, CPUClass cpu)
{
	try {
		ResizeImpl *ret = nullptr;

		EvaluatedFilter filter_h;
		EvaluatedFilter filter_v;

		if (src_width != dst_width || shift_w != 0.0 || subwidth != src_width)
			filter_h = compute_filter(f, src_width, dst_width, shift_w, subwidth);
		if (src_height != dst_height || shift_h != 0.0 || subheight != src_height)
			filter_v = compute_filter(f, src_height, dst_height, shift_h, subheight);

#ifdef ZIMG_X86
		ret = create_resize_impl_x86(filter_h, filter_v, cpu);
#endif
		if (!ret)
			ret = new ResizeImplC(filter_h, filter_v);

		return ret;
	} catch (const std::bad_alloc &) {
		throw ZimgOutOfMemory{};
	}
}

} // namespace resize
} // namespace zimg
