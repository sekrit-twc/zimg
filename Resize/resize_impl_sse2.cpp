#ifdef ZIMG_X86

#include "except.h"
#include "resize_impl_x86.h"

namespace zimg {;
namespace resize {;

namespace {;

class ResizeImplSSE2 : public ResizeImpl {
public:
	ResizeImplSSE2(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v) : ResizeImpl(filter_h, filter_v)
	{}

	void process_u16_h(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "not implemented yet" };
	}

	void process_u16_v(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "not implemented yet" };
	}

	void process_f16_h(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "not implemented yet" };
	}

	void process_f16_v(const uint16_t * RESTRICT src, uint16_t * RESTRICT dst, uint16_t * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "not implemented yet" };
	}

	void process_f32_h(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "not implemented yet" };
	}

	void process_f32_v(const float * RESTRICT src, float * RESTRICT dst, float * RESTRICT tmp,
		int src_width, int src_height, int src_stride, int dst_stride) const override
	{
		throw ZimgUnsupportedError{ "not implemented yet" };
	}
};

} // namespace


ResizeImpl *create_resize_impl_sse2(const EvaluatedFilter &filter_h, const EvaluatedFilter &filter_v)
{
	return new ResizeImplSSE2{ filter_h, filter_v };
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86
