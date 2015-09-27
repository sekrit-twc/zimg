#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "Common/osdep.h"
#include "bilinear.h"
#include "unresize_impl.h"
#include "unresize_impl_x86.h"

namespace zimg {;
namespace unresize {;

namespace {;

class UnresizeImplC : public UnresizeImpl {
public:
	UnresizeImplC(const BilinearContext &hcontext, const BilinearContext &vcontext) : UnresizeImpl(hcontext, vcontext)
	{}

	void process_f16_h(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		throw error::UnsupportedOperation{ "f16 not supported in C impl" };
	}

	void process_f16_v(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, uint16_t *tmp) const override
	{
		throw error::UnsupportedOperation{ "f16 not supported in C impl" };
	}

	void process_f32_h(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const override
	{
		for (int i = 0; i < src.height(); ++i) {
			filter_scanline_h_forward(m_hcontext, src, tmp, i, 0, m_hcontext.dst_width, ScalarPolicy_F32{});
			filter_scanline_h_back(m_hcontext, tmp, dst, i, m_hcontext.dst_width, 0, ScalarPolicy_F32{});
		}
	}

	void process_f32_v(const ImagePlane<const float> &src, const ImagePlane<float> &dst, float *tmp) const override
	{
		for (int i = 0; i < m_vcontext.dst_width; ++i) {
			filter_scanline_v_forward(m_vcontext, src, dst, i, 0, src.width(), ScalarPolicy_F32{});
		}
		for (int i = m_vcontext.dst_width; i > 0; --i) {
			filter_scanline_v_back(m_vcontext, dst, i, 0, src.width(), ScalarPolicy_F32{});
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
		throw error::IllegalArgument("input dimensions must differ from output");
	if (dst_width > src_width || dst_height > src_height)
		throw error::IllegalArgument("input dimension must be greater than output");

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
