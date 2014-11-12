#include <algorithm>
#include <cstddef>
#include "Common/except.h"
#include "Common/plane.h"
#include "dither.h"
#include "error_diffusion.h"
#include "quantize.h"

namespace zimg {;
namespace depth {;

namespace {

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

class ErrorDiffusionC : public DitherConvert {
	template <class T, class U, class ToFloat, class FromFloat>
	void dither(const ImagePlane<const T> &src, const ImagePlane<U> &dst, float *tmp, ToFloat to_float, FromFloat from_float) const
	{		
		int width = src.width();
		int height = src.height();

		float quant_scale = (float)((1 << dst.format().depth) - 1);
		float dequant_scale = 1.0f / quant_scale;

		float *prev_line = tmp + 1;
		float *curr_line = prev_line + width + 2;

		std::fill_n(tmp, ((size_t)width + 2) * 2, 0.0f);

		for (ptrdiff_t i = 0; i < height; ++i) {
			for (ptrdiff_t j = 0; j < width; ++j) {
				float x = to_float(src[i][j]);
				float err = 0;

				err += curr_line[j - 1] * (7.0f / 16.0f);
				err += prev_line[j + 1] * (3.0f / 16.0f);
				err += prev_line[j + 0] * (5.0f / 16.0f);
				err += prev_line[j - 1] * (1.0f / 16.0f);

				x += err;

				float q = (float)(int)(x * quant_scale + (x < 0 ? -0.5 : 0.5)) * dequant_scale;

				dst[i][j] = from_float(x);
				curr_line[j] = x - q;
			}

			std::swap(prev_line, curr_line);
		}
	}
public:
	void byte_to_byte(const ImagePlane<const uint8_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp,
		       make_integer_to_float<uint8_t>(src.format()),
		       make_float_to_integer<uint8_t>(dst.format()));
	}

	void byte_to_word(const ImagePlane<const uint8_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp,
		       make_integer_to_float<uint8_t>(src.format()),
		       make_float_to_integer<uint16_t>(dst.format()));
	}

	void word_to_byte(const ImagePlane<const uint16_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp,
		       make_integer_to_float<uint16_t>(src.format()),
		       make_float_to_integer<uint8_t>(dst.format()));
	}

	void word_to_word(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp,
		       make_integer_to_float<uint16_t>(src.format()),
		       make_float_to_integer<uint16_t>(dst.format()));
	}

	void half_to_byte(const ImagePlane<const uint16_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp,
		       depth::half_to_float,
		       make_float_to_integer<uint8_t>(dst.format()));
	}

	void half_to_word(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp,
		       depth::half_to_float,
		       make_float_to_integer<uint16_t>(dst.format()));
	}

	void float_to_byte(const ImagePlane<const float> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp,
		       identity<float>,
		       make_float_to_integer<uint8_t>(dst.format()));
	}

	void float_to_word(const ImagePlane<const float> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		dither(src, dst, tmp,
		       identity<float>,
		       make_float_to_integer<uint16_t>(dst.format()));
	}
};

} // namespace


DitherConvert *create_error_diffusion(CPUClass cpu)
{
	return new ErrorDiffusionC{};
}

} // namespace depth
} // namespace zimg
