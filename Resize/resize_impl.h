#ifndef RESIZE_IMPL_H_
#define RESIZE_IMPL_H_

#include "filter.h"
#include "osdep.h"

namespace resize {;

class ResizeImpl {
protected:
	EvaluatedFilter m_filter_v;
	EvaluatedFilter m_filter_h;

	ResizeImpl(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v);
public:
	virtual ~ResizeImpl() = 0;

	virtual void process_h(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	                       int src_width, int src_height, int src_stride, int dst_stride) const = 0;

	virtual void process_v(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	                       int src_width, int src_height, int src_stride, int dst_stride) const = 0;
};

class ResizeImplC final : public ResizeImpl{
public:
	ResizeImplC(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v);

	void process_h(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	               int src_width, int src_height, int src_stride, int dst_stride) const override;

	void process_v(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	               int src_width, int src_height, int src_stride, int dst_stride) const override;
};

class ResizeImplX86 final : public ResizeImpl{
public:
	ResizeImplX86(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v);

	void process_h(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	               int src_width, int src_height, int src_stride, int dst_stride) const override;

	void process_v(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
	               int src_width, int src_height, int src_stride, int dst_stride) const override;
};

ResizeImpl *create_resize_impl(const Filter &f, int src_width, int src_height, int dst_width, int dst_height,
                               double shift_w, double shift_h, double subwidth, double subheight, bool x86);

} // namespace resize

#endif // RESIZE_IMPL_H_
