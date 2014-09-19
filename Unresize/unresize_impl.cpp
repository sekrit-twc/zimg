#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "Common/osdep.h"
#include "unresize_impl.h"

namespace zimg {;
namespace unresize {;

UnresizeImpl::UnresizeImpl(const BilinearContext &hcontext, const BilinearContext &vcontext) :
	m_hcontext(hcontext), m_vcontext(vcontext)
{}

UnresizeImpl::~UnresizeImpl()
{}

void UnresizeImpl::unresize_scanline(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp, bool horizontal) const
{
	const BilinearContext &ctx = horizontal ? m_hcontext : m_vcontext;

	const float *c = ctx.lu_c.data();
	const float *l = ctx.lu_l.data();
	const float *u = ctx.lu_u.data();

	float z = 0;
	float w = 0;

	// Matrix-vector product, and forward substitution loop.
	for (int i = 0; i < ctx.dst_width; ++i) {
		const float *row = ctx.matrix_coefficients.data() + i * ctx.matrix_row_size;
		int left = ctx.matrix_row_offsets[i];

		float accum = 0;
		for (int j = 0; j < ctx.matrix_row_size; ++j) {
			accum += row[j] * src[left + j];
		}

		z = (accum - c[i] * z) * l[i];
		tmp[i] = z;
	}

	// Backward substitution loop.
	for (int i = ctx.dst_width - 1; i >= 0; --i) {
		w = tmp[i] - u[i] * w;
		dst[i] = w;
	}
}


UnresizeImplC::UnresizeImplC(const BilinearContext &hcontext, const BilinearContext &vcontext)
: UnresizeImpl(hcontext, vcontext)
{}

void UnresizeImplC::unresize_scanline4_h(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const
{
	for (int i = 0; i < 4; ++i) {
		unresize_scanline(src, dst, tmp, true);

		dst += dst_stride;
		src += src_stride;
	}
}

void UnresizeImplC::unresize_scanline4_v(const float *src, float *dst, float *tmp, int src_stride, int dst_stride) const
{
	for (int i = 0; i < 4; ++i) {
		unresize_scanline(src, dst, tmp, false);

		dst += dst_stride;
		src += src_stride;
	}
}

void UnresizeImplC::transpose_plane(const float *src, float *dst, int src_width, int src_height, int src_stride, int dst_stride) const
{
	for (int i = 0; i < mod(src_height, 16); i += 16) {
		for (int j = 0; j < mod(src_width, 16); j += 16) {
			for (int k = 0; k < 16; ++k) {
				for (int kk = 0; kk < 16; ++kk) {
					dst[(j + kk) * dst_stride + i + k] = src[(i + k) * src_stride + j + kk];
				}
			}
		}
	}
	for (int i = mod(src_height, 16); i < src_height; ++i) {
		for (int j = 0; j < src_width; ++j) {
			dst[j * dst_stride + i] = src[i * src_stride + j];
		}
	}
	for (int j = mod(src_width, 16); j < src_width; ++j) {
		for (int i = 0; i < src_height; ++i) {
			dst[j * dst_stride + i] = src[i * src_stride + j];
		}
	}
}

void UnresizeImplC::load_scanline_u8(const uint8_t *src, float *dst, int width) const
{
	std::transform(src, src + width, dst, [](uint8_t x){ return (float)x / UINT8_MAX; });
}

void UnresizeImplC::load_scanline_u16(const uint16_t *src, float *dst, int width) const
{
	std::transform(src, src + width, dst, [](uint16_t x){ return (float)x / UINT16_MAX; });
}

void UnresizeImplC::store_scanline_u8(const float *src, uint8_t *dst, int width) const
{
	std::transform(src, src + width, dst, clamp_float<uint8_t>);
}

void UnresizeImplC::store_scanline_u16(const float *src, uint16_t *dst, int width) const
{
	std::transform(src, src + width, dst, clamp_float<uint16_t>);
}


UnresizeImpl *create_unresize_impl(int src_width, int src_height, int dst_width, int dst_height, float shift_w, float shift_h, bool x86)
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
	if (x86) {
		int hwidth = hcontext.matrix_row_size;
		int vwidth = vcontext.matrix_row_size;
		X86Capabilities caps = query_x86_capabilities();

		if (caps.sse2) {
			switch (hwidth) {
			case 4:
				if (vwidth == 4)
					ret = new UnresizeImplX86<4, 4>(hcontext, vcontext);
				else if (vwidth == 8)
					ret = new UnresizeImplX86<4, 8>(hcontext, vcontext);
				else if (vwidth == 12)
					ret = new UnresizeImplX86<4, 12>(hcontext, vcontext);
				else
					ret = new UnresizeImplX86<4, 0>(hcontext, vcontext);
				break;
			case 8:
				if (vwidth == 4)
					ret = new UnresizeImplX86<8, 4>(hcontext, vcontext);
				else if (vwidth == 8)
					ret = new UnresizeImplX86<8, 8>(hcontext, vcontext);
				else if (vwidth == 12)
					ret = new UnresizeImplX86<8, 12>(hcontext, vcontext);
				else
					ret = new UnresizeImplX86<8, 0>(hcontext, vcontext);
				break;
			case 12:
				if (vwidth == 4)
					ret = new UnresizeImplX86<12, 4>(hcontext, vcontext);
				else if (vwidth == 8)
					ret = new UnresizeImplX86<12, 8>(hcontext, vcontext);
				else if (vwidth == 12)
					ret = new UnresizeImplX86<12, 12>(hcontext, vcontext);
				else
					ret = new UnresizeImplX86<12, 0>(hcontext, vcontext);
				break;
			default:
				ret = new UnresizeImplX86<0, 0>(hcontext, vcontext);
			}
		}
	}
#endif
	if (!ret)
		ret = new UnresizeImplC(hcontext, vcontext);

	return ret;
}

} // namespace unresize
} // namespace zimg
