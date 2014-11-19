#ifdef ZIMG_X86

#include <immintrin.h>
#include "Common/plane.h"
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

	void byte_to_byte(const ImagePlane<const uint8_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		process(src, dst, DitherPolicyAVX2{}, UnpackByteAVX2{}, PackByteAVX2{},
		        make_integer_to_float_avx2(src.format()), make_float_to_integer_avx2(dst.format()),
		        make_integer_to_float<uint8_t>(src.format()), make_float_to_integer<uint8_t>(dst.format()));
	}

	void byte_to_word(const ImagePlane<const uint8_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		process(src, dst, DitherPolicyAVX2{}, UnpackByteAVX2{}, PackWordAVX2{},
		        make_integer_to_float_avx2(src.format()), make_float_to_integer_avx2(dst.format()),
		        make_integer_to_float<uint8_t>(src.format()), make_float_to_integer<uint16_t>(dst.format()));
	}

	void word_to_byte(const ImagePlane<const uint16_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		process(src, dst, DitherPolicyAVX2{}, UnpackWordAVX2{}, PackByteAVX2{},
		        make_integer_to_float_avx2(src.format()), make_float_to_integer_avx2(dst.format()),
		        make_integer_to_float<uint16_t>(src.format()), make_float_to_integer<uint8_t>(dst.format()));
	}

	void word_to_word(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		process(src, dst, DitherPolicyAVX2{}, UnpackWordAVX2{}, PackWordAVX2{},
		        make_integer_to_float_avx2(src.format()), make_float_to_integer_avx2(dst.format()),
		        make_integer_to_float<uint16_t>(src.format()), make_float_to_integer<uint16_t>(dst.format()));
	}

	void half_to_byte(const ImagePlane<const uint16_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		process(src, dst, DitherPolicyAVX2{}, UnpackHalfAVX2{}, PackByteAVX2{},
		        half_to_float_avx2, make_float_to_integer_avx2(dst.format()),
		        depth::half_to_float, make_float_to_integer<uint8_t>(dst.format()));
	}

	void half_to_word(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		process(src, dst, DitherPolicyAVX2{}, UnpackHalfAVX2{}, PackWordAVX2{},
		        half_to_float_avx2, make_float_to_integer_avx2(dst.format()),
		        depth::half_to_float, make_float_to_integer<uint16_t>(dst.format()));
	}

	void float_to_byte(const ImagePlane<const float> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		process(src, dst, DitherPolicyAVX2{}, UnpackFloatAVX2{}, PackByteAVX2{},
		        identity<__m256>, make_float_to_integer_avx2(dst.format()),
		        identity<float>, make_float_to_integer<uint8_t>(dst.format()));
	}

	void float_to_word(const ImagePlane<const float> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		process(src, dst, DitherPolicyAVX2{}, UnpackFloatAVX2{}, PackWordAVX2{},
		        identity<__m256>, make_float_to_integer_avx2(dst.format()),
		        identity<float>, make_float_to_integer<uint16_t>(dst.format()));
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
