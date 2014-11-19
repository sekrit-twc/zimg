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

template <int N, int M>
struct Max {
	static const int value = N > M ? N : M;
};

template <int N, int M>
struct Div {
	static const int value = N / M;
};

class DepthConvertSSE2 : public DepthConvert {
	template <class T, class U, class Unpack, class Pack, class VectorOp, class ScalarOp>
	void process(const T *src, U *dst, int width, Unpack unpack, Pack pack, VectorOp op, ScalarOp scalar_op) const
	{
		typedef typename Unpack::type src_vector_type;
		typedef typename Pack::type dst_vector_type;

		typedef Max<Unpack::loop_step, Pack::loop_step> loop_step;
		typedef Div<loop_step::value, Unpack::loop_step> loop_unroll_unpack;
		typedef Div<loop_step::value, Pack::loop_step> loop_unroll_pack;

		src_vector_type src_unpacked[loop_unroll_unpack::value * Unpack::unpacked_count];
		dst_vector_type dst_unpacked[loop_unroll_pack::value * Pack::unpacked_count];

		for (int i = 0; i < mod(width, loop_step::value); i += loop_step::value) {
			for (int k = 0; k < loop_unroll_unpack::value; ++k) {
				unpack.unpack(&src_unpacked[k * Unpack::unpacked_count], &src[i + k * Unpack::loop_step]);
			}

			for (int k = 0; k < loop_unroll_pack::value * Pack::unpacked_count; ++k) {
				dst_unpacked[k] = op(src_unpacked[k]);
			}

			for (int k = 0; k < loop_unroll_pack::value; ++k) {
				pack.pack(&dst[i + k * Pack::loop_step], &dst_unpacked[k * Pack::unpacked_count]);
			}
		}
		for (int i = mod(width, loop_step::value); i < width; ++i) {
			dst[i] = scalar_op(src[i]);
		}
	}
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
