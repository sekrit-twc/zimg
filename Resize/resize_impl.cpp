#include "resize_impl.h"

namespace resize {;

ResizeImpl::ResizeImpl(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) :
	m_filter_h{ filter_h }, m_filter_v{ filter_v }
{
}

ResizeImpl::~ResizeImpl()
{
}

ResizeImplC::ResizeImplC(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) :
	ResizeImpl(filter_h, filter_v)
{
}

void ResizeImplC::process_h(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
                           int height, int src_stride, int dst_stride) const
{
	const EvaluatedFilter &filter = m_filter_h;

	for (int i = 0; i < height; ++i) {
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
	}
}

void ResizeImplC::process_v(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
                            int width, int src_stride, int dst_stride) const
{
	const EvaluatedFilter &filter = m_filter_v;

	for (int i = 0; i < filter.height(); ++i) {
		for (int j = 0; j < width; ++j) {
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


ResizeImpl *create_resize_impl(const Filter &f, int src_width, int src_height, int dst_width, int dst_height,
                               double shift_w, double shift_h, double subwidth, double subheight, bool x86)
{
	EvaluatedFilter filter_h = compute_filter(f, src_width, dst_width, shift_w, subwidth);
	EvaluatedFilter filter_v = compute_filter(f, src_height, dst_height, shift_h, subheight);

	return new ResizeImplC(filter_h, filter_v);
}

} // namespace resize
