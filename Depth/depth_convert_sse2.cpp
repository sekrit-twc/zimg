#ifdef ZIMG_X86

#include <algorithm>
#include <emmintrin.h>
#include "Common/align.h"
#include "Common/osdep.h"
#include "quantize.h"
#include "quantize_sse2.h"
#include "depth_convert.h"
#include "depth_convert_x86.h"

namespace zimg {;
namespace depth {;

namespace {;

class DepthConvertSSE2 : public DepthConvertX86 {
public:
	void byte_to_half(const uint8_t *src, uint16_t *dst, int width, const PixelFormat &src_fmt) const override
	{
		auto cvt_sse2 = make_integer_to_float_sse2(src_fmt);
		auto cvt = make_integer_to_float<uint8_t>(src_fmt);
		
		process(src, dst, width, UnpackByteSSE2{}, PackWordSSE2{},
		        [=](__m128i x) { return float_to_half_sse2(cvt_sse2(x)); },
		        [=](uint8_t x) { return depth::float_to_half(cvt(x)); });
	}

	void byte_to_float(const uint8_t *src, float *dst, int width, const PixelFormat &src_fmt) const override
	{
		auto cvt_sse2 = make_integer_to_float_sse2(src_fmt);
		auto cvt = make_integer_to_float<uint8_t>(src_fmt);
		
		process(src, dst, width, UnpackByteSSE2{}, PackFloatSSE2{}, cvt_sse2, cvt);
	}

	void word_to_half(const uint16_t *src, uint16_t *dst, int width, const PixelFormat &src_fmt) const override
	{
		auto cvt_sse2 = make_integer_to_float_sse2(src_fmt);
		auto cvt = make_integer_to_float<uint16_t>(src_fmt);
		
		process(src, dst, width, UnpackWordSSE2{}, PackWordSSE2{},
		        [=](__m128i x) { return float_to_half_sse2(cvt_sse2(x)); },
		        [=](uint16_t x) { return depth::float_to_half(cvt(x)); });
	}

	void word_to_float(const uint16_t *src, float *dst, int width, const PixelFormat &src_fmt) const override
	{
		auto cvt_sse2 = make_integer_to_float_sse2(src_fmt);
		auto cvt = make_integer_to_float<uint16_t>(src_fmt);
		
		process(src, dst, width, UnpackWordSSE2{}, PackFloatSSE2{}, cvt_sse2, cvt);
	}

	void half_to_float(const uint16_t *src, float *dst, int width) const override
	{
		process(src, dst, width, UnpackWordSSE2{}, PackFloatSSE2{}, half_to_float_sse2, depth::half_to_float);
	}

	void float_to_half(const float *src, uint16_t *dst, int width) const override
	{
		process(src, dst, width, UnpackFloatSSE2{}, PackWordSSE2{}, float_to_half_sse2, depth::float_to_half);
	}
};

} // namespace


DepthConvert *create_depth_convert_sse2()
{
	return new DepthConvertSSE2{};
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
