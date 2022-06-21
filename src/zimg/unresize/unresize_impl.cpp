#include <algorithm>
#include <climits>
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "unresize_impl.h"

#if defined(ZIMG_X86)
  #include "x86/unresize_impl_x86.h"
#endif

namespace zimg {
namespace unresize {

namespace {

template <class T>
struct Buffer {
	const graphengine::BufferDescriptor &buffer;

	Buffer(const graphengine::BufferDescriptor &buffer) : buffer{ buffer } {}

	T *operator[](unsigned i) const { return buffer.get_line<T>(i); }
};


void unresize_line_h_f32_c(const BilinearContext &ctx, const float *src, float *dst)
{
	const float *c = ctx.lu_c.data();
	const float *l = ctx.lu_l.data();
	const float *u = ctx.lu_u.data();

	float z = 0.0f;
	float w = 0.0f;

	for (unsigned j = 0; j < ctx.output_width; ++j) {
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

	for (unsigned j = ctx.output_width; j != 0; --j) {
		w = dst[j - 1] - u[j - 1] * w;
		dst[j - 1] = w;
	}
}

void unresize_line_forward_v_f32_c(const BilinearContext &ctx, const Buffer<const float> &src, const Buffer<float> &dst, unsigned i, unsigned left, unsigned right)
{
	const float *c = ctx.lu_c.data();
	const float *l = ctx.lu_l.data();

	const float *coeffs = &ctx.matrix_coefficients[i * ctx.matrix_row_stride];
	unsigned top = ctx.matrix_row_offsets[i];

	for (unsigned j = left; j < right; ++j) {
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

void unresize_line_back_v_f32_c(const BilinearContext &ctx, const Buffer<float> &dst, unsigned i, unsigned left, unsigned right)
{
	const float *u = ctx.lu_u.data();

	for (unsigned j = left; j < right; ++j) {
		float w = i < ctx.output_width ? dst[i][j] : 0.0f;

		w = dst[i - 1][j] - u[i - 1] * w;
		dst[i - 1][j] = w;
	}
}


class UnresizeImplH_C : public UnresizeImplH {
public:
	UnresizeImplH_C(const BilinearContext &context, unsigned height, PixelType type) :
		UnresizeImplH(context, context.output_width, height, type)
	{
		zassert_d(context.input_width <= pixel_max_width(type), "overflow");
		zassert_d(context.output_width <= pixel_max_width(type), "overflow");

		if (type != PixelType::FLOAT)
			error::throw_<error::InternalError>("pixel type not supported");
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		unresize_line_h_f32_c(m_context, in->get_line<float>(i), out->get_line<float>(i));
	}
};

class UnresizeImplV_C : public UnresizeImplV {
public:
	UnresizeImplV_C(const BilinearContext &context, unsigned width, PixelType type) :
		UnresizeImplV(context, width, context.output_width, type)
	{
		if (type != PixelType::FLOAT)
			error::throw_<error::InternalError>("pixel type not supported");
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		unsigned height = m_desc.format.height;

		for (unsigned i = 0; i < height; ++i) {
			unresize_line_forward_v_f32_c(m_context, *in, *out, i, left, right);
		}
		for (unsigned i = height; i != 0; --i) {
			unresize_line_back_v_f32_c(m_context, *out, i, left, right);
		}
	}
};

} // namespace


UnresizeImplH::UnresizeImplH(const BilinearContext &context, unsigned width, unsigned height, PixelType type) :
	m_desc{},
	m_context(context)
{
	zassert_d(m_context.input_width <= pixel_max_width(type), "overflow");
	zassert_d(width <= pixel_max_width(type), "overflow");

	m_desc.format = { width, height, pixel_size(type) };
	m_desc.num_deps = 1;
	m_desc.num_planes = 1;
	m_desc.step = 1;
	m_desc.flags.entire_row = 1;
}

std::pair<unsigned, unsigned> UnresizeImplH::get_row_deps(unsigned i) const noexcept
{
	unsigned step = m_desc.step;
	unsigned last = std::min(i, UINT_MAX - step) + step;
	return{ i, std::min(last, m_desc.format.height) };
}

std::pair<unsigned, unsigned> UnresizeImplH::get_col_deps(unsigned, unsigned) const noexcept
{
	return{ 0, m_context.input_width  };
}


UnresizeImplV::UnresizeImplV(const BilinearContext &context, unsigned width, unsigned height, PixelType type) :
	m_desc{},
	m_context(context)
{
	zassert_d(m_context.input_width <= pixel_max_width(type), "overflow");
	zassert_d(width <= pixel_max_width(type), "overflow");

	m_desc.format = { width, height, pixel_size(type) };
	m_desc.num_deps = 1;
	m_desc.num_planes = 1;
	m_desc.step = UINT_MAX;
	m_desc.flags.entire_col = 1;
}

std::pair<unsigned, unsigned> UnresizeImplV::get_row_deps(unsigned i) const noexcept
{
	return{ 0, m_context.input_width };
}

std::pair<unsigned, unsigned> UnresizeImplV::get_col_deps(unsigned left, unsigned right) const noexcept
{
	return{ left, right };
}


UnresizeImplBuilder::UnresizeImplBuilder(unsigned up_width, unsigned up_height, PixelType type) :
	up_width{ up_width },
	up_height{ up_height },
	type{ type },
	horizontal{},
	orig_dim{},
	shift{},
	cpu{ CPUClass::NONE }
{}

std::unique_ptr<graphengine::Filter> UnresizeImplBuilder::create() const
{
	std::unique_ptr<graphengine::Filter> ret;

	unsigned up_dim = horizontal ? up_width : up_height;
	BilinearContext context = create_bilinear_context(orig_dim, up_dim, shift);

#if defined(ZIMG_X86)
	ret = horizontal ?
		create_unresize_impl_h_x86(context, up_height, type, cpu) :
		create_unresize_impl_v_x86(context, up_width, type, cpu);
#endif

	if (!ret && horizontal)
		ret = std::make_unique<UnresizeImplH_C>(context, up_height, type);
	if (!ret && !horizontal)
		ret = std::make_unique<UnresizeImplV_C>(context, up_width, type);

	return ret;
}

} // namespace unresize
} // namespace zimg
