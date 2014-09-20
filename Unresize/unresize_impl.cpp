#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "Common/osdep.h"
#include "unresize_impl.h"
#include "unresize_impl_x86.h"

namespace zimg {;
namespace unresize {;

namespace {;

class UnresizeImplC : public UnresizeImpl {
public:
	UnresizeImplC(const BilinearContext &hcontext, const BilinearContext &vcontext) : UnresizeImpl(hcontext, vcontext)
	{}

	void process_f32_h(const float *src, float *dst, float *tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		for (int i = 0; i < src_height; ++i) {
			filter_scanline_h_forward(m_hcontext, src, tmp, src_stride, i, 0, m_hcontext.dst_width);
			filter_scanline_h_back(m_hcontext, tmp, dst, dst_stride, i, m_hcontext.dst_width, 0);
		}
	}

	void process_f32_v(const float *src, float *dst, float *tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		for (int i = 0; i < m_vcontext.dst_width; ++i) {
			filter_scanline_v_forward(m_vcontext, src, dst, src_stride, dst_stride, i, 0, src_width);
		}
		for (int i = m_vcontext.dst_width; i > 0; --i) {
			filter_scanline_v_back(m_vcontext, dst, dst_stride, i, 0, src_width);
		}
	}
};

} // namespace


UnresizeImpl::UnresizeImpl(const BilinearContext &hcontext, const BilinearContext &vcontext) :
	m_hcontext(hcontext),
	m_vcontext(vcontext)
{}

UnresizeImpl::~UnresizeImpl()
{}

UnresizeImpl *create_unresize_impl(int src_width, int src_height, int dst_width, int dst_height, float shift_w, float shift_h, CPUClass cpu)
{
	BilinearContext hcontext;
	BilinearContext vcontext;
	UnresizeImpl *ret = nullptr;

	if (dst_width == src_width && dst_height == src_height)
		throw ZimgIllegalArgument("input dimensions must differ from output");
	if (dst_width > src_width || dst_height > src_height)
		throw ZimgIllegalArgument("input dimension must be greater than output");

	if (dst_width != src_width)
		hcontext = create_bilinear_context(dst_width, src_width, shift_w);
	else
		hcontext.matrix_row_size = 0;

	if (dst_height != src_height)
		vcontext = create_bilinear_context(dst_height, src_height, shift_h);
	else
		vcontext.matrix_row_size = 0;

#ifdef ZIMG_X86
	ret = create_unresize_impl_x86(hcontext, vcontext, cpu);
#endif

	if (!ret)
		ret = new UnresizeImplC(hcontext, vcontext);

	return ret;
}

} // namespace unresize
} // namespace zimg
