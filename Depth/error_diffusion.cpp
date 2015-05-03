#include <algorithm>
#include <cstddef>
#include "Common/except.h"
#include "Common/linebuffer.h"
#include "dither.h"
#include "error_diffusion.h"
#include "quantize.h"

namespace zimg {;
namespace depth {;

namespace {;

class ErrorDiffusionC : public DitherConvert {
	template <class T, class U, class Unpack, class Quant, class Dequant>
	void dither(const LineBuffer<T> &src, LineBuffer<U> &dst, int n, float *tmp, Unpack unpack, Quant quant, Dequant dequant) const
	{		
		unsigned left = dst.left();
		unsigned right = dst.right();

		float *prev_line = tmp + 1;
		float *curr_line = prev_line + right + 2;

		const T *src_row = src[n];
		U *dst_row = dst[n];

		if (n % 2)
			std::swap(prev_line, curr_line);

		for (ptrdiff_t j = left; j < right; ++j) {
			float x = unpack(src_row[j]);
			float err = 0;

			err += curr_line[j - 1] * (7.0f / 16.0f);
			err += prev_line[j + 1] * (3.0f / 16.0f);
			err += prev_line[j + 0] * (5.0f / 16.0f);
			err += prev_line[j - 1] * (1.0f / 16.0f);

			x += err;

			U q = quant(x);
			float y = dequant(q);

			dst_row[j] = q;
			curr_line[j] = x - y;
		}
	}
public:
	void byte_to_byte(const LineBuffer<uint8_t> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		dither(src, dst, n, (float *)tmp,
		       make_integer_to_float<uint8_t>(src_fmt),
		       make_float_to_integer<uint8_t>(dst_fmt),
		       make_integer_to_float<uint8_t>(dst_fmt));
	}

	void byte_to_word(const LineBuffer<uint8_t> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		dither(src, dst, n, (float *)tmp,
		       make_integer_to_float<uint8_t>(src_fmt),
		       make_float_to_integer<uint16_t>(dst_fmt),
		       make_integer_to_float<uint16_t>(dst_fmt));
	}

	void word_to_byte(const LineBuffer<uint16_t> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		dither(src, dst, n, (float *)tmp,
		       make_integer_to_float<uint16_t>(src_fmt),
		       make_float_to_integer<uint8_t>(dst_fmt),
		       make_integer_to_float<uint8_t>(dst_fmt));
	}

	void word_to_word(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		dither(src, dst, n, (float *)tmp,
		       make_integer_to_float<uint16_t>(src_fmt),
		       make_float_to_integer<uint16_t>(dst_fmt),
		       make_integer_to_float<uint16_t>(dst_fmt));
	}

	void half_to_byte(const LineBuffer<uint16_t> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		dither(src, dst, n, (float *)tmp,
		       depth::half_to_float,
		       make_float_to_integer<uint8_t>(dst_fmt),
		       make_integer_to_float<uint8_t>(dst_fmt));
	}

	void half_to_word(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		dither(src, dst, n, (float *)tmp,
		       depth::half_to_float,
		       make_float_to_integer<uint16_t>(dst_fmt),
		       make_integer_to_float<uint16_t>(dst_fmt));
	}

	void float_to_byte(const LineBuffer<float> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		dither(src, dst, n, (float *)tmp,
		       identity<float>,
		       make_float_to_integer<uint8_t>(dst_fmt),
		       make_integer_to_float<uint8_t>(dst_fmt));
	}

	void float_to_word(const LineBuffer<float> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		dither(src, dst, n, (float *)tmp,
		       identity<float>,
		       make_float_to_integer<uint16_t>(dst_fmt),
		       make_integer_to_float<uint16_t>(dst_fmt));
	}
};

} // namespace


DitherConvert *create_error_diffusion(CPUClass cpu)
{
	return new ErrorDiffusionC{};
}

} // namespace depth
} // namespace zimg
