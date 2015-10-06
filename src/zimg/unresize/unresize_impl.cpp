#include <algorithm>
#include "common/except.h"
#include "common/linebuffer.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "unresize_impl.h"

namespace zimg {;
namespace unresize {;

namespace {;

void unresize_line_h_f32_c(const BilinearContext &ctx, const float *src, float *dst)
{
	const float *c = ctx.lu_c.data();
	const float *l = ctx.lu_l.data();
	const float *u = ctx.lu_u.data();

	float z = 0.0f;
	float w = 0.0f;

	for (unsigned j = 0; j < ctx.dst_width; ++j) {
		float accum = 0.0f;
		unsigned left = ctx.matrix_row_offsets[j];

		for (unsigned k = 0; k < ctx.matrix_row_size; ++k) {
			float coeff = ctx.matrix_coefficients[j * ctx.matrix_row_stride + k];
			float x = src[left + k];

			accum += coeff * x;
		}

		z = (accum - c[j] * z) * l[j];
		dst[j] = z;
	}

	for (unsigned j = ctx.dst_width; j != 0; --j) {
		w = dst[j - 1] - u[j - 1] * w;
		dst[j - 1] = w;
	}
}

void unresize_line_forward_v_f32_c(const BilinearContext &ctx, const LineBuffer<const float> &src, const LineBuffer<float> &dst, unsigned i, unsigned width)
{
	const float *c = ctx.lu_c.data();
	const float *l = ctx.lu_l.data();

	const float *coeffs = &ctx.matrix_coefficients[i * ctx.matrix_row_stride];
	unsigned top = ctx.matrix_row_offsets[i];

	for (unsigned j = 0; j < width; ++j) {
		float z = i ? dst[i - 1][j] : 0.0f;
		float accum = 0.0f;

		for (unsigned k = 0; k < ctx.matrix_row_size; ++k) {
			float coeff = coeffs[k];
			float x = src[top + k][j];

			accum += coeff * x;
		}

		z = (accum - c[i] * z) * l[i];
		dst[i][j] = z;
	}
}

void unresize_line_back_v_f32_c(const BilinearContext &ctx, const LineBuffer<float> &dst, unsigned i, unsigned width)
{
	const float *u = ctx.lu_u.data();

	for (unsigned j = 0; j < width; ++j) {
		float w = i < ctx.dst_width ? dst[i][j] : 0.0f;

		w = dst[i - 1][j] - u[i - 1] * w;
		dst[i - 1][j] = w;
	}
}


class UnresizeImplH_C final : public UnresizeImplH {
public:
	UnresizeImplH_C(const BilinearContext &context, unsigned height, PixelType type) :
		UnresizeImplH(context, image_attributes{ context.dst_width, height, type })
	{
		if (type != PixelType::FLOAT)
			throw error::InternalError{ "pixel type not supported" };
	}

	void process(void *, const graph::ImageBufferConst &src, const graph::ImageBuffer &dst, void *, unsigned i, unsigned, unsigned) const override
	{
		unresize_line_h_f32_c(m_context, LineBuffer<const float>{ src }[i], LineBuffer<float>{ dst }[i]);
	}
};

class UnresizeImplV_C final : public UnresizeImplV {
public:
	UnresizeImplV_C(const BilinearContext &context, unsigned width, PixelType type) :
		UnresizeImplV(context, image_attributes{ width, context.dst_width, type })
	{
		if (type != PixelType::FLOAT)
			throw error::InternalError{ "pixel type not supported" };
	}

	void process(void *, const graph::ImageBufferConst &src, const graph::ImageBuffer &dst, void *, unsigned, unsigned, unsigned) const override
	{
		LineBuffer<const float> src_buf{ src };
		LineBuffer<float> dst_buf{ dst };

		unsigned width = get_image_attributes().width;
		unsigned height = get_image_attributes().height;

		for (unsigned i = 0; i < height; ++i) {
			unresize_line_forward_v_f32_c(m_context, src_buf, dst_buf, i, width);
		}
		for (unsigned i = height; i != 0; --i) {
			unresize_line_back_v_f32_c(m_context, dst_buf, i, width);
		}
	}
};

} // namespace


UnresizeImplH::UnresizeImplH(const BilinearContext &context, const image_attributes &attr) :
	m_context(context),
	m_attr(attr)
{
}

auto UnresizeImplH::get_flags() const -> filter_flags
{
	filter_flags flags{};

	flags.same_row = true;
	flags.entire_row = true;

	return flags;
}

auto UnresizeImplH::get_image_attributes() const -> image_attributes
{
	return m_attr;
}

auto UnresizeImplH::get_required_row_range(unsigned i) const -> pair_unsigned
{
	return{ i, std::min(i + get_simultaneous_lines(), get_image_attributes().height) };
}

auto UnresizeImplH::get_required_col_range(unsigned left, unsigned right) const -> pair_unsigned
{
	return{ 0, get_image_attributes().width };
}

unsigned UnresizeImplH::get_max_buffering() const
{
	return get_simultaneous_lines();
}


UnresizeImplV::UnresizeImplV(const BilinearContext &context, const image_attributes &attr) :
	m_context(context),
	m_attr(attr)
{
}

auto UnresizeImplV::get_flags() const -> filter_flags
{
	filter_flags flags{};

	flags.has_state = true;
	flags.entire_row = true;
	flags.entire_plane = true;

	return flags;
}

auto UnresizeImplV::get_image_attributes() const -> image_attributes
{
	return m_attr;
}

auto UnresizeImplV::get_required_row_range(unsigned) const -> pair_unsigned
{
	return{ 0, get_image_attributes().height };
}

auto UnresizeImplV::get_required_col_range(unsigned, unsigned) const -> pair_unsigned
{
	return{ 0, get_image_attributes().width };
}

unsigned UnresizeImplV::get_simultaneous_lines() const
{
	return -1;
}

unsigned UnresizeImplV::get_max_buffering() const
{
	return -1;
}


std::unique_ptr<graph::ImageFilter> create_unresize_impl(PixelType type, bool horizontal, unsigned src_width, unsigned src_height,
                                                         unsigned dst_width, unsigned dst_height, double shift, CPUClass cpu)
{
	if (src_width != dst_width && src_height != dst_height)
		throw error::InternalError{ "cannot unresize both width and height" };

	std::unique_ptr<graph::ImageFilter> ret;

	unsigned src_dim = horizontal ? src_width : src_height;
	unsigned dst_dim = horizontal ? dst_width : dst_height;
	BilinearContext context = create_bilinear_context(dst_dim, src_dim, shift);

	if (!ret && horizontal)
		ret = ztd::make_unique<UnresizeImplH_C>(context, dst_height, type);
	if (!ret && !horizontal)
		ret = ztd::make_unique<UnresizeImplV_C>(context, dst_width, type);

	return ret;
}

} // namespace unresize
} // namespace zimg
