#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include "common/alloc.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "graph/filter_base.h"
#include "blue.h"
#include "depth.h"
#include "dither.h"
#include "quantize.h"

#if defined(ZIMG_X86)
  #include "x86/dither_x86.h"
#elif defined(ZIMG_ARM)
  #include "arm/dither_arm.h"
#endif

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
		x = std::clamp(x, 0.0f, static_cast<float>(1UL << bits) - 1);

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
		x = std::clamp(x, 0.0f, static_cast<float>(1UL << bits) - 1);

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
		error::throw_<error::InternalError>("no conversion between pixel types");
}

auto select_error_diffusion_func(PixelType pixel_in, PixelType pixel_out)
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
		error::throw_<error::InternalError>("no conversion between pixel types");
}


constexpr unsigned BAYER_TABLE_LEN = 16;
constexpr unsigned BAYER_TABLE_SCALE = 255;

constexpr uint8_t BAYER_TABLE[BAYER_TABLE_LEN][BAYER_TABLE_LEN] = {
	{   0, 192,  48, 240,  12, 204,  60, 252,   3, 195,  51, 243,  15, 207,  63, 255, },
	{ 128,  64, 176, 112, 140,  76, 188, 124, 131,  67, 179, 115, 143,  79, 191, 127, },
	{  32, 224,  16, 208,  44, 236,  28, 220,  35, 227,  19, 211,  47, 239,  31, 223, },
	{ 160,  96, 144,  80, 172, 108, 156,  92, 163,  99, 147,  83, 175, 111, 159,  95, },
	{   8, 200,  56, 248,   4, 196,  52, 244,  11, 203,  59, 251,   7, 199,  55, 247, },
	{ 136,  72, 184, 120, 132,  68, 180, 116, 139,  75, 187, 123, 135,  71, 183, 119, },
	{  40, 232,  24, 216,  36, 228,  20, 212,  43, 235,  27, 219,  39, 231,  23, 215, },
	{ 168, 104, 152,  88, 164, 100, 148,  84, 171, 107, 155,  91, 167, 103, 151,  87, },
	{   2, 194,  50, 242,  14, 206,  62, 254,   1, 193,  49, 241,  13, 205,  61, 253, },
	{ 130,  66, 178, 114, 142,  78, 190, 126, 129,  65, 177, 113, 141,  77, 189, 125, },
	{  34, 226,  18, 210,  46, 238,  30, 222,  33, 225,  17, 209,  45, 237,  29, 221, },
	{ 162,  98, 146,  82, 174, 110, 158,  94, 161,  97, 145,  81, 173, 109, 157,  93, },
	{  10, 202,  58, 250,   6, 198,  54, 246,   9, 201,  57, 249,   5, 197,  53, 245, },
	{ 138,  74, 186, 122, 134,  70, 182, 118, 137,  73, 185, 121, 133,  69, 181, 117, },
	{  42, 234,  26, 218,  38, 230,  22, 214,  41, 233,  25, 217,  37, 229,  21, 213, },
	{ 170, 106, 154,  90, 166, 102, 150,  86, 169, 105, 153,  89, 165, 101, 149,  85, },
};

AlignedVector<float> load_dither_table(const uint8_t *data, unsigned len, unsigned scale)
{
	zassert_d(len >= 16 && len % 16 == 0, "table length must be multiple of 16");

	AlignedVector<float> table(len * len);

	for (unsigned i = 0; i < len * len; ++i) {
		table[i] = static_cast<float>(data[i] + 1) / (scale + 2) - 0.5f;
	}

	return table;
}


class OrderedDitherTable {
public:
	virtual ~OrderedDitherTable() = default;

	virtual std::tuple<const float *, unsigned, unsigned> get_dither_coeffs(unsigned i, unsigned seq) const = 0;
};

class NoneDitherTable final : public OrderedDitherTable {
public:
	std::tuple<const float *, unsigned, unsigned> get_dither_coeffs(unsigned i, unsigned seq) const override
	{
		static constexpr float table alignas(ALIGNMENT)[AlignmentOf<float>] = {};
		return std::make_tuple(table, 0, AlignmentOf<float> - 1);
	}
};

class BayerDitherTable final : public OrderedDitherTable {
	AlignedVector<float> m_table;
public:
	BayerDitherTable() : m_table(load_dither_table(&BAYER_TABLE[0][0], BAYER_TABLE_LEN, BAYER_TABLE_SCALE))
	{
		m_table.resize(m_table.size() * 4);

		float *alternate1 = m_table.data() + BAYER_TABLE_LEN * BAYER_TABLE_LEN * 1;
		float *alternate2 = m_table.data() + BAYER_TABLE_LEN * BAYER_TABLE_LEN * 2;
		float *alternate3 = m_table.data() + BAYER_TABLE_LEN * BAYER_TABLE_LEN * 3;

		for (unsigned i = 0; i < BAYER_TABLE_LEN; ++i) {
			for (unsigned j = 0; j < BAYER_TABLE_LEN; ++j) {
				// Horizontal flip.
				alternate1[i * BAYER_TABLE_LEN + j] = m_table[i * BAYER_TABLE_LEN + (BAYER_TABLE_LEN - j - 1)];
				// Vertical flip.
				alternate2[i * BAYER_TABLE_LEN + j] = m_table[(BAYER_TABLE_LEN - i - 1) * BAYER_TABLE_LEN + j];
				// Transposed.
				alternate3[i * BAYER_TABLE_LEN + j] = m_table[j * BAYER_TABLE_LEN + i];
			}
		}
	}

	std::tuple<const float *, unsigned, unsigned> get_dither_coeffs(unsigned i, unsigned seq) const override
	{
		const float *data = m_table.data() + BAYER_TABLE_LEN * BAYER_TABLE_LEN * (seq % 4) + (i % BAYER_TABLE_LEN) * BAYER_TABLE_LEN;
		return std::make_tuple(data, 0, BAYER_TABLE_LEN - 1);
	}
};

class RandomDitherTable final : public OrderedDitherTable {
	AlignedVector<float> m_table;
public:
	RandomDitherTable() : m_table(load_dither_table(&blue_noise_table[0][0], BLUE_NOISE_LEN, BLUE_NOISE_SCALE))
	{}

	std::tuple<const float *, unsigned, unsigned> get_dither_coeffs(unsigned i, unsigned seq) const override
	{
		const unsigned offset[] = { (0 << 8) | 0, (32 << 8) | 12, (16 << 8) | 55, (48 << 8) | 26 };
		unsigned hoff = offset[seq % 4] >> 8;
		unsigned voff = offset[seq % 4] & 0xFF;

		const float *data = m_table.data() + ((i + voff) % BLUE_NOISE_LEN) * BLUE_NOISE_LEN;
		return std::make_tuple(data, hoff, BLUE_NOISE_LEN - 1);
	}
};


class OrderedDither : public graph::PointFilter {
	std::shared_ptr<OrderedDitherTable> m_dither_table;
	dither_convert_func m_func;
	dither_f16c_func m_f16c;
	float m_scale;
	float m_offset;
	unsigned m_depth;
	unsigned m_plane;

	void check_preconditions(unsigned width, const PixelFormat &pixel_in, const PixelFormat &pixel_out)
	{
		zassert_d(width <= pixel_max_width(pixel_in.type), "overflow");
		zassert_d(width <= pixel_max_width(pixel_out.type), "overflow");

		if (!pixel_is_integer(pixel_out.type))
			error::throw_<error::InternalError>("cannot dither to non-integer format");
	}
public:
	OrderedDither(std::shared_ptr<OrderedDitherTable> table, dither_convert_func func, dither_f16c_func f16c, unsigned width, unsigned height,
	              const PixelFormat &pixel_in, const PixelFormat &pixel_out, unsigned plane) :
		PointFilter(width, height, pixel_out.type),
		m_dither_table{ std::move(table) },
		m_func{ func },
		m_f16c{ f16c },
		m_scale{},
		m_offset{},
		m_depth{ pixel_out.depth },
		m_plane{ plane }
	{
		check_preconditions(width, pixel_in, pixel_out);

		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.scratchpad_size = m_f16c ? (static_cast<checked_size_t>(width) * sizeof(float)).get() : 0;
		m_desc.flags.in_place = pixel_size(pixel_in.type) == pixel_size(pixel_out.type);

		std::tie(m_scale, m_offset) = get_scale_offset(pixel_in, pixel_out);
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *, void *tmp) const noexcept override
	{
		auto dither = m_dither_table->get_dither_coeffs(i, m_plane);

		const void *src_line = in->get_line(i);
		void *dst_line = out->get_line(i);

		if (m_f16c) {
			m_f16c(src_line, tmp, left, right);
			src_line = tmp;
		}

		m_func(std::get<0>(dither), std::get<1>(dither), std::get<2>(dither), src_line, dst_line, m_scale, m_offset, m_depth, left, right);
	}
};


class ErrorDiffusion : public graph::FilterBase {
public:
	typedef void (*ed_func)(const void *src, void *dst, void *error_top, void *error_cur, float scale, float offset, unsigned bits, unsigned width);
private:
	ed_func m_func;
	dither_f16c_func m_f16c;
	float m_scale;
	float m_offset;
	unsigned m_depth;

	void check_preconditions(unsigned width, const PixelFormat &pixel_in, const PixelFormat &pixel_out)
	{
		zassert_d(width <= pixel_max_width(pixel_in.type), "overflow");
		zassert_d(width <= pixel_max_width(pixel_out.type), "overflow");

		if (!pixel_is_integer(pixel_out.type))
			error::throw_<error::InternalError>("cannot dither to non-integer format");
	}
public:
	ErrorDiffusion(ed_func func, dither_f16c_func f16c, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out) :
		m_func{ func },
		m_f16c{ f16c },
		m_scale{},
		m_offset{},
		m_depth{ pixel_out.depth }
	{
		check_preconditions(width, pixel_in, pixel_out);

		m_desc.format = { width, height, pixel_size(pixel_out.type) };
		m_desc.num_deps = 1;
		m_desc.num_planes = 1;
		m_desc.step = 1;

		m_desc.context_size = ((static_cast<checked_size_t>(width) + 2) * sizeof(float) * 2).get();
		m_desc.scratchpad_size = m_f16c ? (static_cast<checked_size_t>(width) * sizeof(float)).get() : 0;

		m_desc.flags.stateful = 1;
		m_desc.flags.in_place = pixel_size(pixel_in.type) == pixel_size(pixel_out.type);
		m_desc.flags.entire_row = 1;

		std::tie(m_scale, m_offset) = get_scale_offset(pixel_in, pixel_out);
	}

	pair_unsigned get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	pair_unsigned get_col_deps(unsigned, unsigned) const noexcept override { return{ 0, m_desc.format.width }; }

	void init_context(void *context) const noexcept override
	{
		std::fill_n(static_cast<float *>(context), m_desc.context_size / sizeof(float), 0.0f);
	}

	void process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	             unsigned i, unsigned left, unsigned right, void *context, void *tmp) const noexcept override
	{
		const void *src_p = in->get_line(i);
		void *dst_p = out->get_line(i);

		void *error_a = context;
		void *error_b = static_cast<uint8_t *>(context) + m_desc.context_size / 2;

		void *error_top = i % 2 ? error_a : error_b;
		void *error_cur = i % 2 ? error_b : error_a;

		if (m_f16c) {
			m_f16c(src_p, tmp, 0, m_desc.format.width);
			src_p = tmp;
		}

		m_func(src_p, dst_p, error_top, error_cur, m_scale, m_offset, m_depth, m_desc.format.width);
	}
};


std::unique_ptr<OrderedDitherTable> create_dither_table(DitherType type, unsigned width, unsigned height)
{
	switch (type) {
	case DitherType::NONE:
		return std::make_unique<NoneDitherTable>();
	case DitherType::ORDERED:
		return std::make_unique<BayerDitherTable>();
	case DitherType::RANDOM:
		return std::make_unique<RandomDitherTable>();
	default:
		error::throw_<error::InternalError>("unrecognized dither type");
	}
}

std::unique_ptr<graphengine::Filter> create_error_diffusion(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
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

	return std::make_unique<ErrorDiffusion>(func, f16c, width, height, pixel_in, pixel_out);
}

} // namespace


DepthConversion::result create_dither(DitherType type, unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, const bool planes[4], CPUClass cpu)
{
	if (type == DitherType::ERROR_DIFFUSION)
		return{ create_error_diffusion(width, height, pixel_in, pixel_out, cpu), planes };

	dither_convert_func func = nullptr;
	dither_f16c_func f16c = nullptr;
	bool needs_f16c = (pixel_in.type == PixelType::HALF);

#if defined(ZIMG_X86)
	func = select_ordered_dither_func_x86(pixel_in, pixel_out, cpu);
	needs_f16c = needs_f16c && needs_dither_f16c_func_x86(cpu);
#elif defined(ZIMG_ARM)
	func = select_ordered_dither_func_arm(pixel_in, pixel_out, cpu);
	needs_f16c = needs_f16c && needs_dither_f16c_func_arm(cpu);
#endif
	if (!func)
		func = select_ordered_dither_func(pixel_in.type, pixel_out.type);

	if (needs_f16c) {
#if defined(ZIMG_X86)
		f16c = select_dither_f16c_func_x86(cpu);
#elif defined(ZIMG_ARM)
		f16c = select_dither_f16c_func_arm(cpu);
#endif
		if (!f16c)
			f16c = half_to_float_n;
	}

	std::shared_ptr<OrderedDitherTable> table = create_dither_table(type, width, height);
	DepthConversion::result res{};
	for (unsigned p = 0; p < 4; ++p) {
		if (!planes[p])
			continue;

		res.filters[p] = std::make_unique<OrderedDither>(table, func, f16c, width, height, pixel_in, pixel_out, p);
		res.filter_refs[p] = res.filters[p].get();
	}
	return res;
}

} // namespace depth
} // namespace zimg
