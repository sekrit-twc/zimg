#include <algorithm>
#include "resize_impl.h"
#include "resize_impl_x86.h"

namespace resize {;

namespace {;

int32_t unpack_u16(uint16_t x)
{
	// Convert to 16-bit signed and store in 32-bits.
	return (int32_t)x + (int32_t)INT16_MIN;
}

int32_t round_shift(int32_t x, int32_t n)
{
	x += x < 0 ? -(1 << (n - 1)) : 1 << (n - 1);
	return x >> n;
}

uint16_t pack_i30(int32_t x)
{
	// Reduce 16.14 fixed point to 16.0 and convert to unsigned.
	x = round_shift(x, 14) - (int32_t)INT16_MIN;
	x = std::max(x, (int32_t)0);
	x = std::min(x, (int32_t)UINT16_MAX);
	return (uint16_t)x;
}

class ResizeImplC final : public ResizeImpl {
public:
	ResizeImplC(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) : ResizeImpl(filter_h, filter_v)
	{}

	void process_u16_h(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		const EvaluatedFilter &filter = m_filter_h;

		for (int i = 0; i < src_height; ++i) {
			for (int j = 0; j < filter.height(); ++j) {
				int left = filter.left()[j];
				int32_t accum = 0;

				for (int k = 0; k < filter.width(); ++k) {
					int32_t coeff = filter.data_i16()[j * filter.stride_i16() + k];
					int32_t x = unpack_u16(src[i * src_stride + left + k]);

					accum += coeff * x;
				}

				dst[i * dst_stride + j] = pack_i30(accum);
			}
		}
	}

	void process_u16_v(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		const EvaluatedFilter &filter = m_filter_v;

		for (int i = 0; i < filter.height(); ++i) {
			for (int j = 0; j < src_width; ++j) {
				int top = filter.left()[i];
				int32_t accum = 0;

				for (int k = 0; k < filter.width(); ++k) {
					int32_t coeff = filter.data_i16()[i * filter.stride_i16() + k];
					int32_t x = unpack_u16(src[(top + k) * src_stride + j]);

					accum += coeff * x;
				}

				dst[i * dst_stride + j] = pack_i30(accum);
			}
		}
	}

	void process_f32_h(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		const EvaluatedFilter &filter = m_filter_h;

		for (int i = 0; i < src_height; ++i) {
			for (int j = 0; j < filter.height(); ++j) {
				int left = filter.left()[j];
				float accum = 0.f;

				for (int k = 0; k < filter.width(); ++k) {
					float coeff = filter.data()[j * filter.stride() + k];
					float x = src[i * src_stride + left + k];
					accum += coeff * x;
				}
				dst[i * dst_stride + j] = accum;
			}
		};
	}

	void process_f32_v(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	                   int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		const EvaluatedFilter &filter = m_filter_v;

		for (int i = 0; i < filter.height(); ++i) {
			for (int j = 0; j < src_width; ++j) {
				int top = filter.left()[i];
				float accum = 0.f;

				for (int k = 0; k < filter.width(); ++k) {
					float coeff = filter.data()[i * filter.stride() + k];
					float x = src[(top + k) * src_stride + j];
					accum += coeff * x;
				}
				dst[i * dst_stride + j] = accum;
			}
		}
	}
};

} // namespace


ResizeImpl::ResizeImpl(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) :
	m_filter_h{ filter_h }, m_filter_v{ filter_v }
{
}

ResizeImpl::~ResizeImpl()
{
}

ResizeImpl *create_resize_impl(const Filter &f, int src_width, int src_height, int dst_width, int dst_height,
                               double shift_w, double shift_h, double subwidth, double subheight, bool x86)
{
	EvaluatedFilter filter_h; 
	EvaluatedFilter filter_v;

	if (src_width != dst_width || shift_w != 0.0 || subwidth != src_width)
		filter_h = compute_filter(f, src_width, dst_width, shift_w, subwidth);
	if (src_height != dst_height || shift_h != 0.0 || subheight != src_height)
		filter_v = compute_filter(f, src_height, dst_height, shift_h, subheight);

	if (x86) {
#ifdef RESIZE_X86
		return create_resize_impl_x86(filter_h, filter_v);
#else
		throw std::runtime_error{ "x86 support not enabled" };
#endif
	} else {
		return new ResizeImplC(filter_h, filter_v);
	}
}

} // namespace resize
