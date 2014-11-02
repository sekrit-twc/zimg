#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>
#include "Common/align.h"
#include "Common/except.h"
#include "Common/plane.h"
#include "depth.h"
#include "dither_impl.h"
#include "quantize.h"

namespace zimg {;
namespace depth {;

namespace {;

static const int RANDOM_DITHER_NUMBER = 128 * 128;
static const int RANDOM_DITHER_PERIOD = 128;

// Divide by 65.
static const unsigned short ORDERED_DITHERS[64] = {
	 1, 49, 13, 61,  4, 52, 16, 64,
	33, 17, 45, 29, 36, 20, 48, 32,
	 9, 57,  5, 53, 12, 60,  8, 56,
	41, 25, 37, 21, 44, 28, 40, 24,
	 3, 51, 15, 63,  2, 50, 14, 62,
	35, 19, 47, 31, 34, 18, 46, 30,
	11, 59,  7, 55, 10, 58,  6, 54,
	43, 27, 39, 23, 42, 26, 38, 22	
};

template <class T>
float normalize_dither(T x, T min, T max)
{
	return static_cast<float>(x - min) / static_cast<float>(max) - 0.5f;
}

std::pair<int, int> get_none_dithers(float *p)
{
	std::fill_n(p, 8, 0.0f);
	return{ 8, 8 };
}

std::pair<int, int> get_ordered_dithers(float *p)
{
	for (unsigned i : ORDERED_DITHERS) {
		*p++ = normalize_dither(i, 0U, 65U);
	}

	return{ 64, 8 };
}

std::pair<int, int> get_random_dithers(float *p)
{
	std::mt19937 mt;
	uint_fast32_t mt_min = std::mt19937::min();
	uint_fast32_t mt_max = std::mt19937::max();

	// Dividing the random numbers by 4 chosen arbitrarily to limit noisiness.
	std::generate_n(p, RANDOM_DITHER_NUMBER, [&]() { return normalize_dither(mt(), mt_min, mt_max) * 0.25; });

	return{ RANDOM_DITHER_NUMBER, RANDOM_DITHER_PERIOD };
}

template <class T>
IntegerToFloat<T> make_integer_to_float(const PixelFormat &fmt)
{
	return{ fmt.depth, fmt.fullrange, fmt.chroma };
}

template <class T>
FloatToInteger<T> make_float_to_integer(const PixelFormat &fmt)
{
	return{ fmt.depth, fmt.fullrange, fmt.chroma };
}

class OrderedDitherC : public OrderedDither {
	template <class T, class U, class ToFloat, class FromFloat>
	void dither(const ImagePlane<const T> &src, const ImagePlane<U> &dst, float *tmp, int depth, ToFloat to_float, FromFloat from_float) const
	{
		const T *src_p = src.data();
		U *dst_p = dst.data();
		int width = src.width();
		int height = dst.height();
		int src_stride = src.stride();
		int dst_stride = dst.stride();

		int hperiod = m_period;
		int vperiod = (int)(m_dither.size() / hperiod);
		float scale = 1.0f / (float)(1 << (depth - 1));

		auto dither_pixel = [=](T x, float d) { return from_float(to_float(x) + d * scale); };

		for (int i = 0; i < height; ++i) {
			int loop_end = mod(width, hperiod);
			int m, j;

			const float *dith = m_dither.data() + (i % vperiod) * hperiod;

			for (j = 0; j < loop_end; j += hperiod) {
				m = 0;
				for (int jj = j; jj < j + hperiod; ++jj) {
					dst_p[i * dst_stride + jj] = dither_pixel(src_p[i * src_stride + jj], dith[m++]);
				}
			}

			m = 0;
			for (; j < width; ++j) {
				dst_p[i * dst_stride + j] = dither_pixel(src_p[i * src_stride + j], dith[m++]);
			}
		}
	}
public:
	OrderedDitherC(const float *first, const float *last, int period) : OrderedDither(first, last, period)
	{}

	void byte_to_byte(const ImagePlane<const uint8_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp, dst.format().depth,
		       make_integer_to_float<uint8_t>(src.format()), make_float_to_integer<uint8_t>(dst.format()));
	}

	void byte_to_word(const ImagePlane<const uint8_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp, dst.format().depth,
		       make_integer_to_float<uint8_t>(src.format()), make_float_to_integer<uint16_t>(dst.format()));
	}

	void word_to_byte(const ImagePlane<const uint16_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp, dst.format().depth,
		       make_integer_to_float<uint16_t>(src.format()), make_float_to_integer<uint8_t>(dst.format()));
	}

	void word_to_word(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp, dst.format().depth,
		       make_integer_to_float<uint16_t>(src.format()), make_float_to_integer<uint16_t>(dst.format()));
	}

	void half_to_byte(const ImagePlane<const uint16_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp, dst.format().depth, depth::half_to_float, make_float_to_integer<uint8_t>(dst.format()));
	}

	void half_to_word(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp, dst.format().depth, depth::half_to_float, make_float_to_integer<uint16_t>(dst.format()));
	}

	void float_to_byte(const ImagePlane<const float> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp, dst.format().depth, identity<float>, make_float_to_integer<uint8_t>(dst.format()));
	}

	void float_to_word(const ImagePlane<const float> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp, dst.format().depth, identity<float>, make_float_to_integer<uint16_t>(dst.format()));
	}
};

} // namespace


OrderedDither::OrderedDither(const float *first, const float *last, int period) :
	m_dither(first, last),
	m_period{ period }
{
}

DitherConvert *create_ordered_dither(DitherType type, CPUClass cpu)
{
	std::vector<float> dithers(RANDOM_DITHER_NUMBER);
	std::pair<int, int> dither_size{};

	switch (type) {
	case DitherType::DITHER_NONE:
		dither_size = get_none_dithers(dithers.data());
		break;
	case DitherType::DITHER_ORDERED:
		dither_size = get_ordered_dithers(dithers.data());
		break;
	case DitherType::DITHER_RANDOM:
		dither_size = get_random_dithers(dithers.data());
		break;
	default:
		throw ZimgIllegalArgument{ "unrecognized ordered dither type" };
	}

	return new OrderedDitherC{ dithers.data(), dithers.data() + dither_size.first, dither_size.second };
}

} // namespace depth
} // namespace zimg
