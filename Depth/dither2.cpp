#include <algorithm>
#include <limits>
#include <random>
#include <utility>
#include "Common/except.h"
#include "Common/linebuffer.h"
#include "Common/pixel.h"
#include "depth2.h"
#include "dither2.h"
#include "quantize.h"

namespace zimg {;
namespace depth {;

namespace {;

static const int ORDERED_DITHER_SIZE = 8;
static const int ORDERED_DITHER_NUM = ORDERED_DITHER_SIZE * ORDERED_DITHER_SIZE;
static const int ORDERED_DITHERS_SCALE = 65;

static const unsigned short ORDERED_DITHERS[ORDERED_DITHER_NUM] = {
	 1, 49, 13, 61,  4, 52, 16, 64,
	33, 17, 45, 29, 36, 20, 48, 32,
	 9, 57,  5, 53, 12, 60,  8, 56,
	41, 25, 37, 21, 44, 28, 40, 24,
	 3, 51, 15, 63,  2, 50, 14, 62,
	35, 19, 47, 31, 34, 18, 46, 30,
	11, 59,  7, 55, 10, 58,  6, 54,
	43, 27, 39, 23, 42, 26, 38, 22	
};

template <class T, class U>
void dither_ordered(const float *dither, unsigned dither_offset, unsigned dither_len, const void *src, void *dst, float scale, float offset, unsigned bits, unsigned width)
{
	const T *src_p = reinterpret_cast<const T *>(src);
	U *dst_p = reinterpret_cast<U *>(dst);

	for (unsigned i = 0; i < width; ++i) {
		float x = static_cast<float>(src_p[i]) * scale + offset;
		float d = dither[(dither_offset + i) % dither_len];

		x += d;
		x = std::min(std::max(x, 0.0f), static_cast<float>(((uint32_t)1 << bits) - 1));

		dst_p[i] = static_cast<U>(x + d + 0.5f);
	}
}

template <class T, class U>
void dither_ed(const void *src, void *dst, void *error_top, void *error_cur, float scale, float offset, unsigned bits, unsigned width)
{
	const float *error_top_p = reinterpret_cast<const float *>(error_top) + AlignmentOf<float>::value;
	float *error_cur_p = reinterpret_cast<float *>(error_cur) + AlignmentOf<float>::value;

	const T *src_p = reinterpret_cast<const T *>(src);
	U *dst_p = reinterpret_cast<U *>(dst);

	for (unsigned i = 0; i < width; ++i) {
		float x = static_cast<float>(src_p[i]) * scale + offset;
		float err = 0;

		err += error_cur_p[(int)i - 1] * (7.0f / 16.0f);
		err += error_top_p[(int)i + 1] * (3.0f / 16.0f);
		err += error_top_p[(int)i + 0] * (5.0f / 16.0f);
		err += error_top_p[(int)i - 1] * (1.0f / 16.0f);

		x += err;
		x = std::min(std::max(x, 0.0f), static_cast<float>(((uint32_t)1 << bits) - 1));

		U q = static_cast<U>(x + 0.5f);
		
		dst_p[i] = q;
		error_cur_p[i] = x - static_cast<float>(q);
	}
}

void half_to_float_n(const void *src, void *dst, unsigned width)
{
	const uint16_t *src_p = reinterpret_cast<const uint16_t *>(src);
	float *dst_p = reinterpret_cast<float *>(dst);

	std::transform(src_p, src_p + width, dst_p, half_to_float);
}

std::pair<float, float> get_scale_offset(const PixelFormat &pixel_in, const PixelFormat &pixel_out)
{
	uint32_t range_in = pixel_in.type < PixelType::HALF ? integer_range(pixel_in.depth, pixel_in.fullrange, pixel_in.chroma) : 1;
	uint32_t offset_in = pixel_in.type < PixelType::HALF ? integer_offset(pixel_in.depth, pixel_in.fullrange, pixel_in.chroma) : 0;
	uint32_t range_out = pixel_out.type < PixelType::HALF ? integer_range(pixel_out.depth, pixel_out.fullrange, pixel_out.chroma) : 1;
	uint32_t offset_out = pixel_out.type < PixelType::HALF ? integer_offset(pixel_out.depth, pixel_out.fullrange, pixel_out.chroma) : 0;

	float scale = (float)((double)range_out / range_in);
	float offset = (float)(-(double)offset_in * range_out / range_in + (double)offset_out);

	return{ scale, offset };
}

std::pair<OrderedDitherBase::func_type, OrderedDitherBase::f16c_func_type> select_func(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	OrderedDitherBase::func_type func = nullptr;
	OrderedDitherBase::f16c_func_type f16c = nullptr;

	if (pixel_in.type == PixelType::BYTE && pixel_out.type == PixelType::BYTE)
		func = dither_ordered<uint8_t, uint8_t>;
	else if (pixel_in.type == PixelType::BYTE && pixel_out.type == PixelType::WORD)
		func = dither_ordered<uint8_t, uint16_t>;
	else if (pixel_in.type == PixelType::WORD && pixel_out.type == PixelType::BYTE)
		func = dither_ordered<uint16_t, uint8_t>;
	else if (pixel_in.type == PixelType::WORD && pixel_out.type == PixelType::WORD)
		func = dither_ordered<uint16_t, uint16_t>;
	else if ((pixel_in.type == PixelType::HALF || pixel_in.type == PixelType::FLOAT) && pixel_out.type == PixelType::BYTE)
		func = dither_ordered<float, uint8_t>;
	else if ((pixel_in.type == PixelType::HALF || pixel_in.type == PixelType::FLOAT) && pixel_out.type == PixelType::WORD)
		func = dither_ordered<float, uint8_t>;

	if (pixel_in.type == PixelType::HALF)
		f16c = half_to_float_n;

	if (pixel_in == pixel_out) {
		func = nullptr;
		f16c = nullptr;
	}

	return{ func, f16c };
}

std::pair<ErrorDiffusion::func_type, ErrorDiffusion::f16c_func_type> select_func_ed(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	ErrorDiffusion::func_type func = nullptr;
	ErrorDiffusion::f16c_func_type f16c = nullptr;

	if (pixel_in.type == PixelType::BYTE && pixel_out.type == PixelType::BYTE)
		func = dither_ed<uint8_t, uint8_t>;
	else if (pixel_in.type == PixelType::BYTE && pixel_out.type == PixelType::WORD)
		func = dither_ed<uint8_t, uint16_t>;
	else if (pixel_in.type == PixelType::WORD && pixel_out.type == PixelType::BYTE)
		func = dither_ed<uint16_t, uint8_t>;
	else if (pixel_in.type == PixelType::WORD && pixel_out.type == PixelType::WORD)
		func = dither_ed<uint16_t, uint16_t>;
	else if ((pixel_in.type == PixelType::HALF || pixel_in.type == PixelType::FLOAT) && pixel_out.type == PixelType::BYTE)
		func = dither_ed<float, uint8_t>;
	else if ((pixel_in.type == PixelType::HALF || pixel_in.type == PixelType::FLOAT) && pixel_out.type == PixelType::WORD)
		func = dither_ed<float, uint8_t>;

	if (pixel_in.type == PixelType::HALF)
		f16c = half_to_float_n;

	if (pixel_in == pixel_out) {
		func = nullptr;
		f16c = nullptr;
	}

	return{ func, f16c };
}

} // namespace


OrderedDitherBase::OrderedDitherBase(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu) :
	m_func{},
	m_f16c{},
	m_pixel_in{ pixel_in.type },
	m_pixel_out{ pixel_out.type },
	m_scale{},
	m_offset{},
	m_depth{ (unsigned)pixel_out.depth }
{
	auto impl = select_func(pixel_in, pixel_out, cpu);
	m_func = impl.first;
	m_f16c = impl.second;

	auto scale_offset = get_scale_offset(pixel_in, pixel_out);
	m_scale = scale_offset.first;
	m_offset = scale_offset.second;

}

ZimgFilterFlags OrderedDitherBase::get_flags() const
{
	ZimgFilterFlags flags{};

	flags.same_row = true;
	flags.in_place = pixel_size(m_pixel_in) >= pixel_size(m_pixel_out);

	return flags;
}

size_t OrderedDitherBase::get_tmp_size(unsigned left, unsigned right) const
{
	return (m_func && m_f16c) ? align(right - left, AlignmentOf<float>::value) * sizeof(float) : 0;
}

void OrderedDitherBase::process(void *ctx, const ZimgImageBuffer *src, const ZimgImageBuffer *dst, void *tmp, unsigned i, unsigned left, unsigned right) const
{
	LineBuffer<void> src_buf{ src->data[0], right, (unsigned)src->stride[0], src->mask[0] };
	LineBuffer<void> dst_buf{ dst->data[0], right, (unsigned)dst->stride[0], dst->mask[0] };

	const void *src_p = reinterpret_cast<const char *>(src_buf[i]) + left * pixel_size(m_pixel_in);
	void *dst_p = reinterpret_cast<char *>(dst_buf[i]) + left * pixel_size(m_pixel_out);

	if (!m_func && !m_f16c) {
		if (src_p != dst_p)
			std::copy_n(reinterpret_cast<const char *>(src_p), (right - left) * pixel_size(m_pixel_out), reinterpret_cast<char *>(dst_p));
	} else {
		auto dither_params = get_dither_params(i, left);
		const float *dither = m_dither.data() + std::get<0>(dither_params);
		unsigned dither_offset = std::get<1>(dither_params);
		unsigned dither_len = std::get<2>(dither_params);

		if (m_f16c) {
			m_f16c(src_p, dst_p, right - left);
			src_p = dst_p;
			dst_p = reinterpret_cast<char *>(dst_buf[i]) + left * pixel_size(m_pixel_out);
		}

		if (m_func)
			m_func(dither, dither_offset, dither_len, src_p, dst_p, m_scale, m_offset, m_depth, right - left);
	}
}


NoneDither::NoneDither(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu) :
	OrderedDitherBase(pixel_in, pixel_out, cpu)
{
	m_dither.assign(16, 0.0f);
}

std::tuple<unsigned, unsigned, unsigned> NoneDither::get_dither_params(unsigned i, unsigned left) const
{
	return std::make_tuple(0U, 0U, 16U);
}

BayerDither::BayerDither(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu) :
	OrderedDitherBase(pixel_in, pixel_out, cpu)
{
	m_dither.reserve(std::end(ORDERED_DITHERS) - std::begin(ORDERED_DITHERS));

	for (unsigned d : ORDERED_DITHERS) {
		m_dither.push_back((float)d / ORDERED_DITHERS_SCALE - 0.5f);
	}
}

std::tuple<unsigned, unsigned, unsigned> BayerDither::get_dither_params(unsigned i, unsigned left) const
{
	return std::make_tuple((i % ORDERED_DITHER_SIZE) * ORDERED_DITHER_SIZE, left % ORDERED_DITHER_SIZE, ORDERED_DITHER_SIZE);
}

RandomDither::RandomDither(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu) :
	OrderedDitherBase(pixel_in, pixel_out, cpu)
{
	std::mt19937 mt;
	auto mt_min = std::mt19937::min();
	auto mt_max = std::mt19937::max();

	m_dither.resize(RAND_NUM);

	std::generate(m_dither.begin(), m_dither.end(), [&](){ return (float)((double)(mt() - mt_min) / (double)(mt_max - mt_min) - 0.5) * 0.5f; });
}

std::tuple<unsigned, unsigned, unsigned> RandomDither::get_dither_params(unsigned i, unsigned left) const
{
	std::mt19937 mt{ (uint32_t)i + left };
	unsigned offset = mt() % (RAND_NUM / AlignmentOf<float>::value);

//	return std::make_tuple((i % 64) * 64, left % 64, 64);
	return std::make_tuple(0U, offset, RAND_NUM);
}


ErrorDiffusion::ErrorDiffusion(unsigned width, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu) :
	m_func{},
	m_f16c{},
	m_pixel_in{ pixel_in.type },
	m_pixel_out{ pixel_out.type },
	m_scale{},
	m_offset{},
	m_depth{ (unsigned)pixel_out.depth },
	m_width{ width }
{
	auto impl = select_func_ed(pixel_in, pixel_out, cpu);
	m_func = impl.first;
	m_f16c = impl.second;

	auto scale_offset = get_scale_offset(pixel_in, pixel_out);
	m_scale = scale_offset.first;
	m_offset = scale_offset.second;
}

ZimgFilterFlags ErrorDiffusion::get_flags() const
{
	ZimgFilterFlags flags{};

	flags.has_state = true;
	flags.same_row = true;
	flags.in_place = pixel_size(m_pixel_in) >= pixel_size(m_pixel_out);
	flags.entire_row = true;

	return flags;
}

size_t ErrorDiffusion::get_context_width() const
{
	return 2 * (align(m_width, AlignmentOf<float>::value) + 2 * AlignmentOf<float>::value);
}

size_t ErrorDiffusion::get_context_size() const
{
	return get_context_width() * sizeof(float);
}

void ErrorDiffusion::init_context(void *ctx) const
{
	float *ctx_p = reinterpret_cast<float *>(ctx);
	std::fill_n(ctx_p, get_context_width(), 0.0f);
}

void ErrorDiffusion::process(void *ctx, const ZimgImageBuffer *src, const ZimgImageBuffer *dst, void *tmp, unsigned i, unsigned, unsigned) const
{
	LineBuffer<void> src_buf{ src->data[0], m_width, (unsigned)src->stride[0], src->mask[0] };
	LineBuffer<void> dst_buf{ dst->data[0], m_width, (unsigned)dst->stride[0], dst->mask[0] };

	const void *src_p = reinterpret_cast<const char *>(src_buf[i]);
	void *dst_p = reinterpret_cast<char *>(dst_buf[i]);

	void *error_a = ctx;
	void *error_b = reinterpret_cast<float *>(ctx) + get_context_width() / 2;

	void *error_cur = i % 2 ? error_b : error_a;
	void *error_top = i % 2 ? error_a : error_b;

	if (!m_func && !m_f16c) {
		if (src_p != dst_p)
			std::copy_n(reinterpret_cast<const char *>(src_p), m_width * pixel_size(m_pixel_out), reinterpret_cast<char *>(dst_p));
	} else {
		if (m_f16c) {
			m_f16c(src_p, dst_p, m_width);
			src_p = dst_p;
			dst_p = reinterpret_cast<char *>(dst_buf[i]);
		}

		if (m_func)
			m_func(src_p, dst_p, error_cur, error_top, m_scale, m_offset, m_depth, m_width);
	}
}


IZimgFilter *create_dither_convert2(DitherType type, unsigned width, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	if (pixel_out.type != PixelType::BYTE && pixel_out.type != PixelType::WORD)
		throw ZimgLogicError{ "dither only converts to floating-point types" };

	switch (type) {
	case DitherType::DITHER_NONE:
		return new NoneDither{ pixel_in, pixel_out, cpu };
	case DitherType::DITHER_ORDERED:
		return new BayerDither{ pixel_in, pixel_out, cpu };
	case DitherType::DITHER_RANDOM:
		return new RandomDither{ pixel_in, pixel_out, cpu };
	case DitherType::DITHER_ERROR_DIFFUSION:
		return new ErrorDiffusion{ width, pixel_in, pixel_out, cpu };
	default:
		throw ZimgLogicError{ "unknown dither type" };
	}
}

} // namespace depth
} // namespace zimg
