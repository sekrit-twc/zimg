#include <algorithm>
#include <memory>
#include "Common/align.h"
#include "Common/alloc.h"
#include "Common/linebuffer.h"
#include "Common/pair_filter.h"
#include "Common/pixel.h"
#include "resize2.h"
#include "resize_impl2.h"

namespace zimg {;
namespace resize {;

namespace {;

bool resize_h_first(double xscale, double yscale)
{
	double h_first_cost = std::max(xscale, 1.0) * 2.0 + xscale * std::max(yscale, 1.0);
	double v_first_cost = std::max(yscale, 1.0) + yscale * std::max(xscale, 1.0) * 2.0;

	return h_first_cost < v_first_cost;
}

class CopyFilter : public ZimgFilter {
	unsigned m_width;
	unsigned m_height;
	PixelType m_type;
public:
	CopyFilter(unsigned width, unsigned height, PixelType type) :
		m_width{ width },
		m_height{ height },
		m_type{ type }
	{
	}

	ZimgFilterFlags get_flags() const override
	{
		ZimgFilterFlags flags{};

		flags.same_row = 1;
		flags.in_place = 1;

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_width, m_height, m_type };
	}

	void process(void *, const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		LineBuffer<const void> src_buf{ src };
		LineBuffer<void> dst_buf{ dst };

		copy_buffer_lines(src_buf, dst_buf, i, i + 1, left * pixel_size(m_type), right * pixel_size(m_type));
	}
};

} // namespace


IZimgFilter *create_resize2(const Filter &filter, PixelType type, unsigned depth, int src_width, int src_height, int dst_width, int dst_height,
                            double shift_w, double shift_h, double subwidth, double subheight, CPUClass cpu)
{
	bool skip_h = (src_width == dst_width && shift_w == 0 && subwidth == src_width);
	bool skip_v = (src_height == dst_height && shift_h == 0 && subheight == src_height);

	if (skip_h && skip_v) {
		return new CopyFilter{ (unsigned)src_width, (unsigned)src_height, type };
	} else if (skip_h) {
		return create_resize_impl2(filter, type, false, depth, src_width, src_height, dst_width, dst_height, shift_h, subheight, cpu);
	} else if (skip_v) {
		return create_resize_impl2(filter, type, true, depth, src_width, src_height, dst_width, dst_height, shift_w, subwidth, cpu);
	} else {
		bool h_first = resize_h_first((double)dst_width / src_width, (double)dst_height / src_height);
		std::unique_ptr<IZimgFilter> stage1;
		std::unique_ptr<IZimgFilter> stage2;
		std::unique_ptr<IZimgFilter> ret;

		if (h_first) {
			stage1.reset(create_resize_impl2(filter, type, true, depth, src_width, src_height, dst_width, src_height, shift_w, subwidth, cpu));
			stage2.reset(create_resize_impl2(filter, type, false, depth, dst_width, src_height, dst_width, dst_height, shift_h, subheight, cpu));
		} else {
			stage1.reset(create_resize_impl2(filter, type, false, depth, src_width, src_height, src_width, dst_height, shift_h, subheight, cpu));
			stage2.reset(create_resize_impl2(filter, type, true, depth, src_width, dst_height, dst_width, dst_height, shift_w, subwidth, cpu));
		}

		ret.reset(new PairFilter{ stage1.get(), stage2.get() });
		stage1.release();
		stage2.release();

		return ret.release();
	}
}

} // namespace resize
} // namespace zimg
