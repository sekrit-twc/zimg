#if 0
#ifdef ZIMG_X86

#include <immintrin.h>
#include "Common/linebuffer.h"
#include "dither_impl.h"
#include "dither_impl_x86.h"
#include "quantize_avx2.h"

namespace zimg {;
namespace depth {;

namespace {;

struct DitherPolicyAVX2 {
	typedef __m256 type;
	static const int vector_size = 8;

	__m256 set1(float x) { return _mm256_set1_ps(x); }

	__m256 load(const float *ptr) { return _mm256_load_ps(ptr); }

	__m256 add(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }

	__m256 mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
};

class OrderedDitherAVX2 : public OrderedDitherX86 {
public:
	explicit OrderedDitherAVX2(const float *dither) : OrderedDitherX86(dither)
	{}

	void byte_to_byte(const LineBuffer<uint8_t> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		process(src, dst, dst_fmt.depth, n, DitherPolicyAVX2{}, UnpackByteAVX2{}, PackByteAVX2{},
		        make_integer_to_float_avx2(src_fmt), make_float_to_integer_avx2(dst_fmt),
		        make_integer_to_float<uint8_t>(src_fmt), make_float_to_integer<uint8_t>(dst_fmt));
	}

	void byte_to_word(const LineBuffer<uint8_t> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		process(src, dst, dst_fmt.depth, n, DitherPolicyAVX2{}, UnpackByteAVX2{}, PackWordAVX2{},
		        make_integer_to_float_avx2(src_fmt), make_float_to_integer_avx2(dst_fmt),
		        make_integer_to_float<uint8_t>(src_fmt), make_float_to_integer<uint16_t>(dst_fmt));
	}

	void word_to_byte(const LineBuffer<uint16_t> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		process(src, dst, dst_fmt.depth, n, DitherPolicyAVX2{}, UnpackWordAVX2{}, PackByteAVX2{},
		        make_integer_to_float_avx2(src_fmt), make_float_to_integer_avx2(dst_fmt),
		        make_integer_to_float<uint16_t>(src_fmt), make_float_to_integer<uint8_t>(dst_fmt));
	}

	void word_to_word(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		process(src, dst, dst_fmt.depth, n, DitherPolicyAVX2{}, UnpackWordAVX2{}, PackWordAVX2{},
		        make_integer_to_float_avx2(src_fmt), make_float_to_integer_avx2(dst_fmt),
		        make_integer_to_float<uint16_t>(src_fmt), make_float_to_integer<uint16_t>(dst_fmt));
	}

	void half_to_byte(const LineBuffer<uint16_t> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		process(src, dst, dst_fmt.depth, n, DitherPolicyAVX2{}, UnpackHalfAVX2{}, PackByteAVX2{},
		        half_to_float_avx2, make_float_to_integer_avx2(dst_fmt),
		        depth::half_to_float, make_float_to_integer<uint8_t>(dst_fmt));
	}

	void half_to_word(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		process(src, dst, dst_fmt.depth, n, DitherPolicyAVX2{}, UnpackHalfAVX2{}, PackWordAVX2{},
		        half_to_float_avx2, make_float_to_integer_avx2(dst_fmt),
		        depth::half_to_float, make_float_to_integer<uint16_t>(dst_fmt));
	}

	void float_to_byte(const LineBuffer<float> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		process(src, dst, dst_fmt.depth, n, DitherPolicyAVX2{}, UnpackFloatAVX2{}, PackByteAVX2{},
		        identity<__m256>, make_float_to_integer_avx2(dst_fmt),
		        identity<float>, make_float_to_integer<uint8_t>(dst_fmt));
	}

	void float_to_word(const LineBuffer<float> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const override
	{
		process(src, dst, dst_fmt.depth, n, DitherPolicyAVX2{}, UnpackFloatAVX2{}, PackWordAVX2{},
		        identity<__m256>, make_float_to_integer_avx2(dst_fmt),
		        identity<float>, make_float_to_integer<uint16_t>(dst_fmt));
	}
};

} // namespace


DitherConvert *create_ordered_dither_avx2(const float *dither)
{
	return new OrderedDitherAVX2{ dither };
}

} // namespace depth
} // namespace zimg

#endif // ZIMG_X86
#endif
