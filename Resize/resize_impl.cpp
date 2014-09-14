#include "except.h"
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

class ResizeImplC final : public ResizeImpl {
public:
	ResizeImplC(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) : ResizeImpl(filter_h, filter_v)
	{}

	void process_u16_h(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		const EvaluatedFilter &filter = m_filter_h;
		filter_plane_h_scalar(filter, src, dst, 0, src_height, 0, filter.height(), src_stride, dst_stride, ScalarPolicy_U16{});
	}

	void process_u16_v(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		const EvaluatedFilter &filter = m_filter_v;
		filter_plane_v_scalar(filter, src, dst, 0, filter.height(), 0, src_width, src_stride, dst_stride, ScalarPolicy_U16{});
	}

	void process_f16_h(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in C impl" };
	}

	void process_f16_v(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in C impl" };
	}

	void process_f32_h(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		const EvaluatedFilter &filter = m_filter_h;
		filter_plane_h_scalar(filter, src, dst, 0, src_height, 0, filter.height(), src_stride, dst_stride, ScalarPolicy_F32{});
	}

	void process_f32_v(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		const EvaluatedFilter &filter = m_filter_v;
		filter_plane_v_scalar(filter, src, dst, 0, filter.height(), 0, src_width, src_stride, dst_stride, ScalarPolicy_F32{});
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
                               double shift_w, double shift_h, double subwidth, double subheight, bool x86)
{
	try {
		EvaluatedFilter filter_h;
		EvaluatedFilter filter_v;

		if (src_width != dst_width || shift_w != 0.0 || subwidth != src_width)
			filter_h = compute_filter(f, src_width, dst_width, shift_w, subwidth);
		if (src_height != dst_height || shift_h != 0.0 || subheight != src_height)
			filter_v = compute_filter(f, src_height, dst_height, shift_h, subheight);

		if (x86) {
#ifdef ZIMG_X86
			ResizeImpl *ret = create_resize_impl_x86(filter_h, filter_v);

			if (ret)
				return ret;
			else
				return new ResizeImplC(filter_h, filter_v);
#else
			return new ResizeImplC(filter_h, filter_v);
#endif
		} else {
			return new ResizeImplC(filter_h, filter_v);
		}
	} catch (const std::bad_alloc &) {
		throw ZimgOutOfMemory{};
	}
}

} // namespace resize
} // namespace zimg
