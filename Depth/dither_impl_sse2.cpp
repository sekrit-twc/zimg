#ifdef ZIMG_X86

#include "Common/plane.h"
#include "dither_impl.h"
#include "dither_impl_x86.h"
#include "quantize_sse2.h"

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

class OrderedDitherSSE2 : public OrderedDither {
	template <class T, class U, class Unpack, class Pack, class ToFloat, class FromFloat, class ToFloatScalar, class FromFloatScalar>
	void process(const ImagePlane<const T> &src, const ImagePlane<U> &dst, Unpack unpack, Pack pack,
	             ToFloat to_float, FromFloat from_float, ToFloatScalar to_float_scalar, FromFloatScalar from_float_scalar) const
	{
		typedef typename Unpack::type src_vector_type;
		typedef typename Pack::type dst_vector_type;

		typedef Max<Unpack::loop_step, Pack::loop_step> loop_step;
		typedef Div<loop_step::value, Unpack::loop_step> loop_unroll_unpack;
		typedef Div<loop_step::value, Pack::loop_step> loop_unroll_pack;

		int width = src.width();
		int height = src.height();

		const float *dither_data = m_dither.data();

		float scale = 1.0f / (float)(1 << (dst.format().depth - 1));
		__m128 scale_ps = _mm_set_ps1(scale);

		for (ptrdiff_t i = 0; i < height; ++i) {
			const T *src_row = src[i];
			U * dst_row = dst[i];

			const float *dither_row = &dither_data[(i % NUM_DITHERS_V) * NUM_DITHERS_H];
			int m = 0;

			src_vector_type src_unpacked[loop_unroll_unpack::value * Unpack::unpacked_count];
			dst_vector_type dst_unpacked[loop_unroll_pack::value * Pack::unpacked_count];

			for (ptrdiff_t j = 0; j < mod(width, loop_step::value); j += loop_step::value) {
				for (ptrdiff_t k = 0; k < loop_unroll_unpack::value; ++k) {
					unpack.unpack(&src_unpacked[k * Unpack::unpacked_count], &src_row[j + k * Unpack::loop_step]);
				}

				for (ptrdiff_t k = 0; k < loop_unroll_pack::value * Pack::unpacked_count; ++k) {
					__m128 x = to_float(src_unpacked[k]);
					__m128 d = _mm_load_ps(&dither_row[m]);

					d = _mm_mul_ps(d, scale_ps);
					x = _mm_add_ps(x, d);

					dst_unpacked[k] = from_float(x);

					m += 4;
				}

				for (ptrdiff_t k = 0; k < loop_unroll_pack::value; ++k) {
					pack.pack(&dst_row[j + k * Pack::loop_step], &dst_unpacked[k * Pack::unpacked_count]);
				}

				m %= NUM_DITHERS_H;
			}

			m = 0;
			for (ptrdiff_t j = mod(width, loop_step::value); j < width; ++j) {
				float x = to_float_scalar(src[i][j]);
				float d = dither_row[m++];

				dst[i][j] = from_float_scalar(x + d * scale);
				m %= NUM_DITHERS_H;
			}
		}
	}
public:
	explicit OrderedDitherSSE2(const float *dither) : OrderedDitherX86(dither)
	{}

	void byte_to_byte(const ImagePlane<const uint8_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		process(src, dst, UnpackByteSSE2{}, PackByteSSE2{},
		        make_integer_to_float_sse2(src.format()), make_float_to_integer_sse2(dst.format()),
		        make_integer_to_float<uint8_t>(src.format()), make_float_to_integer<uint8_t>(dst.format()));
	}

	void byte_to_word(const ImagePlane<const uint8_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		process(src, dst, UnpackByteSSE2{}, PackWordSSE2{},
		        make_integer_to_float_sse2(src.format()), make_float_to_integer_sse2(dst.format()),
		        make_integer_to_float<uint8_t>(src.format()), make_float_to_integer<uint16_t>(dst.format()));
	}

	void word_to_byte(const ImagePlane<const uint16_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		process(src, dst, UnpackWordSSE2{}, PackByteSSE2{},
		        make_integer_to_float_sse2(src.format()), make_float_to_integer_sse2(dst.format()),
		        make_integer_to_float<uint16_t>(src.format()), make_float_to_integer<uint8_t>(dst.format()));
	}

	void word_to_word(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		process(src, dst, UnpackWordSSE2{}, PackWordSSE2{},
		        make_integer_to_float_sse2(src.format()), make_float_to_integer_sse2(dst.format()),
		        make_integer_to_float<uint16_t>(src.format()), make_float_to_integer<uint16_t>(dst.format()));
	}

	void half_to_byte(const ImagePlane<const uint16_t> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		process(src, dst, UnpackWordSSE2{}, PackByteSSE2{},
		        half_to_float_sse2, make_float_to_integer_sse2(dst.format()),
		        depth::half_to_float, make_float_to_integer<uint8_t>(dst.format()));
	}

	void half_to_word(const ImagePlane<const uint16_t> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		process(src, dst, UnpackWordSSE2{}, PackWordSSE2{},
		        half_to_float_sse2, make_float_to_integer_sse2(dst.format()),
		        depth::half_to_float, make_float_to_integer<uint16_t>(dst.format()));
	}

	void float_to_byte(const ImagePlane<const float> &src, const ImagePlane<uint8_t> &dst, float *tmp) const override
	{
		process(src, dst, UnpackFloatSSE2{}, PackByteSSE2{},
		        identity<__m128>, make_float_to_integer_sse2(dst.format()),
		        identity<float>, make_float_to_integer<uint8_t>(dst.format()));
	}

	void float_to_word(const ImagePlane<const float> &src, const ImagePlane<uint16_t> &dst, float *tmp) const override
	{
		process(src, dst, UnpackFloatSSE2{}, PackWordSSE2{},
		        identity<__m128>, make_float_to_integer_sse2(dst.format()),
		        identity<float>, make_float_to_integer<uint16_t>(dst.format()));
	}
};

} // namespace


DitherConvert *create_ordered_dither_sse2(const float *dither)
{
	return new OrderedDitherSSE2{ dither };
}

}
}

#endif // ZIMG_X86
