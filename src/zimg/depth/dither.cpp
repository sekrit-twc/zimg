#include <algorithm>
#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include "common/alloc.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "graph/image_filter.h"
#include "depth.h"
#include "dither.h"
#include "dither_x86.h"
#include "hexfloat.h"
#include "quantize.h"

namespace zimg {
namespace depth {

namespace {

template <class T, class U>
void dither_ordered(const float *dither, unsigned dither_offset, unsigned dither_mask,
                    const void *src, void *dst, float scale, float offset, unsigned bits, unsigned left, unsigned right)
{
	const T *src_p = static_cast<const T *>(src);
	U *dst_p = static_cast<U *>(dst);

	for (unsigned j = left; j < right; ++j) {
		float x = static_cast<float>(src_p[j]) * scale + offset;
		float d = dither[(dither_offset + j) & dither_mask];

		x += d;
		x = std::min(std::max(x, 0.0f), static_cast<float>(1UL << bits) - 1);

		dst_p[j] = static_cast<U>(std::lrint(x));
	}
}

template <class T, class U>
void dither_ed(const void *src, void *dst, void *error_top, void *error_cur, float scale, float offset, unsigned bits, unsigned width)
{
	const float *error_top_p = static_cast<const float *>(error_top);
	float *error_cur_p = static_cast<float *>(error_cur);

	const T *src_p = static_cast<const T *>(src);
	U *dst_p = static_cast<U *>(dst);

	for (unsigned j = 0; j < width; ++j) {
		// Error array is padded by one on each side.
		unsigned j_err = j + 1;

		float x = static_cast<float>(src_p[j]) * scale + offset;
		float err = 0;

		err += error_cur_p[j_err - 1] * (7.0f / 16.0f);
		err += error_top_p[j_err + 1] * (3.0f / 16.0f);
		err += error_top_p[j_err + 0] * (5.0f / 16.0f);
		err += error_top_p[j_err - 1] * (1.0f / 16.0f);

		x += err;
		x = std::min(std::max(x, 0.0f), static_cast<float>(1UL << bits) - 1);

		U q = static_cast<U>(std::lrint(x));

		dst_p[j] = q;
		error_cur_p[j_err] = x - static_cast<float>(q);
	}
}

void half_to_float_n(const void *src, void *dst, unsigned left, unsigned right)
{
	const uint16_t *src_p = static_cast<const uint16_t *>(src);
	float *dst_p = static_cast<float *>(dst);

	std::transform(src_p + left, src_p + right, dst_p + left, half_to_float);
}


dither_convert_func select_ordered_dither_func(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::HALF)
		pixel_in = PixelType::FLOAT;

	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return dither_ordered<uint8_t, uint8_t>;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return dither_ordered<uint8_t, uint16_t>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return dither_ordered<uint16_t, uint8_t>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return dither_ordered<uint16_t, uint16_t>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::BYTE)
		return dither_ordered<float, uint8_t>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return dither_ordered<float, uint16_t>;
	else
		throw error::InternalError{ "no conversion between pixel types" };
}

decltype(&dither_ed<uint8_t, uint8_t>) select_error_diffusion_func(PixelType pixel_in, PixelType pixel_out)
{
	if (pixel_in == PixelType::HALF)
		pixel_in = PixelType::FLOAT;

	if (pixel_in == PixelType::BYTE && pixel_out == PixelType::BYTE)
		return dither_ed<uint8_t, uint8_t>;
	else if (pixel_in == PixelType::BYTE && pixel_out == PixelType::WORD)
		return dither_ed<uint8_t, uint16_t>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::BYTE)
		return dither_ed<uint16_t, uint8_t>;
	else if (pixel_in == PixelType::WORD && pixel_out == PixelType::WORD)
		return dither_ed<uint16_t, uint16_t>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::BYTE)
		return dither_ed<float, uint8_t>;
	else if (pixel_in == PixelType::FLOAT && pixel_out == PixelType::WORD)
		return dither_ed<float, uint16_t>;
	else
		throw error::InternalError{ "no conversion between pixel types" };
}


constexpr unsigned BAYER_TABLE_LEN = 8;

constexpr uint8_t BAYER_TABLE[BAYER_TABLE_LEN * BAYER_TABLE_LEN] = {
	 1, 49, 13, 61,  4, 52, 16, 64,
	33, 17, 45, 29, 36, 20, 48, 32,
	 9, 57,  5, 53, 12, 60,  8, 56,
	41, 25, 37, 21, 44, 28, 40, 24,
	 3, 51, 15, 63,  2, 50, 14, 62,
	35, 19, 47, 31, 34, 18, 46, 30,
	11, 59,  7, 55, 10, 58,  6, 54,
	43, 27, 39, 23, 42, 26, 38, 22
};

constexpr unsigned BAYER_TABLE_SCALE = 65;

class OrderedDitherTable {
public:
	virtual ~OrderedDitherTable() = default;

	virtual std::tuple<const float *, unsigned, unsigned> get_dither_coeffs(unsigned i, unsigned left) const = 0;
};

class NoneDitherTable final : public OrderedDitherTable {
	AlignedVector<float> m_table;
public:
	NoneDitherTable()
	{
		m_table.resize(8);
	}

	std::tuple<const float *, unsigned, unsigned> get_dither_coeffs(unsigned i, unsigned left) const override
	{
		return std::make_tuple(m_table.data(), 0, 7);
	}
};

class BayerDitherTable final : public OrderedDitherTable {
	AlignedVector<float> m_table;
public:
	BayerDitherTable()
	{
		m_table.resize(BAYER_TABLE_LEN * BAYER_TABLE_LEN);

		for (unsigned i = 0; i < BAYER_TABLE_LEN * BAYER_TABLE_LEN; ++i) {
			m_table[i] = static_cast<float>(BAYER_TABLE[i]) / BAYER_TABLE_SCALE - 0.5f;
		}
	}

	std::tuple<const float *, unsigned, unsigned> get_dither_coeffs(unsigned i, unsigned left) const override
	{
		const float *data = m_table.data() + (i % BAYER_TABLE_LEN) * BAYER_TABLE_LEN;

		return std::make_tuple(data, left % BAYER_TABLE_LEN, 7);
	}
};

class RandomDitherTable final : public OrderedDitherTable {
	static constexpr size_t RAND_NUM = 1 << 14;

	AlignedVector<float> m_table;
	std::vector<unsigned> m_row_offset;
public:
	RandomDitherTable(unsigned, unsigned height)
	{
		// The greatest value such that rint(65535.0f + x) yields 65535.0f unchanged.
		float safe_min = HEX_LF_C(-0x1.fdfffep-2);
		float safe_max = HEX_LF_C(0x1.fdfffep-2);

		std::mt19937 mt;
		double mt_min = std::mt19937::min();
		double mt_max = std::mt19937::max();

		m_table.resize(RAND_NUM);

		std::generate(m_table.begin(), m_table.end(), [&]()
		{
			double x = mt();
			float f = static_cast<float>((x - mt_min) / (mt_max - mt_min) - 0.5);
			return std::min(std::max(f, safe_min), safe_max);
		});

		m_row_offset.resize(height);

		for (unsigned i = 0; i < height; ++i) {
			std::mt19937 mt{ i };
			m_row_offset[i] = floor_n(mt(), 8);
		}
	}

	std::tuple<const float *, unsigned, unsigned> get_dither_coeffs(unsigned i, unsigned left) const override
	{
		unsigned offset = (m_row_offset[i] + left) % RAND_NUM;
		unsigned mask = RAND_NUM - 1;

		return std::make_tuple(m_table.data(), offset, mask);
	}
};

class OrderedDither final : public graph::ImageFilterBase {
	std::unique_ptr<OrderedDitherTable> m_dither_table;
	dither_convert_func m_func;
	dither_f16c_func m_f16c;

	PixelType m_pixel_in;
	PixelType m_pixel_out;

	float m_scale;
	float m_offset;
	unsigned m_depth;

	unsigned m_width;
	unsigned m_height;
public:
	OrderedDither(std::unique_ptr<OrderedDitherTable> &&table, dither_convert_func func, dither_f16c_func f16c, unsigned width, unsigned height,
	              const PixelFormat &format_in, const PixelFormat &format_out) :
		m_func{ func },
		m_f16c{ f16c },
		m_pixel_in{ format_in.type },
		m_pixel_out{ format_out.type },
		m_scale{},
		m_offset{},
		m_depth{ format_out.depth },
		m_width{ width },
		m_height{ height }
	{
		zassert_d(width <= pixel_max_width(format_in.type), "overflow");
		zassert_d(width <= pixel_max_width(format_out.type), "overflow");

		if (!pixel_is_integer(format_out.type))
			throw error::InternalError{ "cannot dither to non-integer format" };

		std::tie(m_scale, m_offset) = get_scale_offset(format_in, format_out);
		m_dither_table = std::move(table);
	}

	filter_flags get_flags() const override
	{
		filter_flags flags{};

		flags.same_row = true;
		flags.in_place = (pixel_size(m_pixel_in) == pixel_size(m_pixel_out));

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_width, m_height, m_pixel_out };
	}

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		size_t size = 0;

		if (m_f16c) {
			unsigned pixel_align = std::max(pixel_alignment(m_pixel_in), pixel_alignment(m_pixel_out));

			left = floor_n(left, pixel_align);
			right = ceil_n(right, pixel_align);

			size = (right - left) * sizeof(float);
		}

		return size;
	}

	void process(void *ctx, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		const char *src_line = graph::static_buffer_cast<const char>(*src)[i];
		char *dst_line = graph::static_buffer_cast<char>(*dst)[i];

		unsigned pixel_align = std::max(pixel_alignment(m_pixel_in), pixel_alignment(m_pixel_out));
		unsigned line_base = floor_n(left, pixel_align);

		const float *dither_table;
		unsigned dither_offset;
		unsigned dither_mask;

		src_line += pixel_size(m_pixel_in) * line_base;
		dst_line += pixel_size(m_pixel_out) * line_base;

		std::tie(dither_table, dither_offset, dither_mask) = m_dither_table->get_dither_coeffs(i, line_base);

		left -= line_base;
		right -= line_base;

		if (m_f16c) {
			m_f16c(src_line, tmp, left, right);
			src_line = static_cast<char *>(tmp);
		}

		m_func(dither_table, dither_offset, dither_mask, src_line, dst_line, m_scale, m_offset, m_depth, left, right);
	}
};

class ErrorDiffusion final : public graph::ImageFilterBase {
public:
	typedef void(*ed_func)(const void *src, void *dst, void *error_top, void *error_cur, float scale, float offset, unsigned bits, unsigned width);
private:
	ed_func m_func;
	dither_f16c_func m_f16c;

	PixelType m_pixel_in;
	PixelType m_pixel_out;

	float m_scale;
	float m_offset;
	unsigned m_depth;

	unsigned m_width;
	unsigned m_height;
public:
	ErrorDiffusion(ed_func func, dither_f16c_func f16c, unsigned width, unsigned height, const PixelFormat &format_in, const PixelFormat &format_out) :
		m_func{ func },
		m_f16c{ f16c },
		m_pixel_in{ format_in.type },
		m_pixel_out{ format_out.type },
		m_scale{},
		m_offset{},
		m_depth{ format_out.depth },
		m_width{ width },
		m_height{ height }
	{
		zassert_d(width <= pixel_max_width(format_in.type), "overflow");
		zassert_d(width <= pixel_max_width(format_out.type), "overflow");

		if (!pixel_is_integer(format_out.type))
			throw error::InternalError{ "cannot dither to non-integer format" };

		std::tie(m_scale, m_offset) = get_scale_offset(format_in, format_out);
	}

	filter_flags get_flags() const override
	{
		filter_flags flags{};

		flags.has_state = true;
		flags.same_row = true;
		flags.in_place = pixel_size(m_pixel_in) == pixel_size(m_pixel_out);
		flags.entire_row = true;

		return flags;
	}

	pair_unsigned get_required_col_range(unsigned, unsigned) const override
	{
		return{ 0, get_image_attributes().width };
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_width, m_height, m_pixel_out };
	}

	size_t get_context_size() const override
	{
		return (m_width + 2) * sizeof(float) * 2;
	}

	size_t get_tmp_size(unsigned, unsigned) const override
	{
		return m_f16c ? ceil_n(m_width, AlignmentOf<float>::value) * sizeof(float) : 0;
	}

	void init_context(void *ctx) const override
	{
		std::fill_n(static_cast<float *>(ctx), get_context_size() / sizeof(float), 0.0f);
	}

	void process(void *ctx, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned, unsigned) const override
	{
		const void *src_p = (*src)[i];
		void *dst_p = (*dst)[i];

		void *error_a = ctx;
		void *error_b = static_cast<unsigned char *>(ctx) + get_context_size() / 2;

		void *error_top = i % 2 ? error_a : error_b;
		void *error_cur = i % 2 ? error_b : error_a;

		if (m_f16c) {
			m_f16c(src_p, tmp, 0, m_width);
			src_p = tmp;
		}

		m_func(src_p, dst_p, error_top, error_cur, m_scale, m_offset, m_depth, m_width);
	}
};


std::unique_ptr<OrderedDitherTable> create_dither_table(DitherType type, unsigned width, unsigned height)
{
	switch (type) {
	case DitherType::NONE:
		return ztd::make_unique<NoneDitherTable>();
	case DitherType::ORDERED:
		return ztd::make_unique<BayerDitherTable>();
	case DitherType::RANDOM:
		return ztd::make_unique<RandomDitherTable>(width, height);
	default:
		throw error::InternalError{ "unrecognized dither type" };
	}
}

std::unique_ptr<graph::ImageFilter> create_error_diffusion(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
#ifdef ZIMG_X86
	if (auto ret = create_error_diffusion_x86(width, height, pixel_in, pixel_out, cpu))
		return ret;
#endif

	ErrorDiffusion::ed_func func = nullptr;
	dither_f16c_func f16c = nullptr;
	bool needs_f16c = (pixel_in.type == PixelType::HALF);

	if (!func)
		func = select_error_diffusion_func(pixel_in.type, pixel_out.type);
	if (needs_f16c && !f16c)
		f16c = half_to_float_n;

	return ztd::make_unique<ErrorDiffusion>(func, f16c, width, height, pixel_in, pixel_out);
}

} // namespace


std::unique_ptr<graph::ImageFilter> create_dither(DitherType type, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	if (type == DitherType::ERROR_DIFFUSION)
		return create_error_diffusion(width, height, pixel_in, pixel_out, cpu);

	auto table = create_dither_table(type, width, height);
	dither_convert_func func = nullptr;
	dither_f16c_func f16c = nullptr;
	bool needs_f16c = (pixel_in.type == PixelType::HALF);

#ifdef ZIMG_X86
	func = select_ordered_dither_func_x86(pixel_in, pixel_out, cpu);
	needs_f16c = needs_f16c && needs_dither_f16c_func_x86(cpu);

	if (needs_f16c)
		f16c = select_dither_f16c_func_x86(cpu);
#endif

	if (!func)
		func = select_ordered_dither_func(pixel_in.type, pixel_out.type);

	if (needs_f16c && !f16c)
		f16c = half_to_float_n;

	return ztd::make_unique<OrderedDither>(std::move(table), func, f16c, width, height, pixel_in, pixel_out);
}

} // namespace depth
} // namespace zimg
