#include <algorithm>
#include <memory>
#include "common/align.h"
#include "common/alloc.h"
#include "common/cpuinfo.h"
#include "common/linebuffer.h"
#include "common/pixel.h"
#include "graph/copy_filter.h"
#include "graph/image_filter.h"
#include "resize.h"
#include "resize_impl.h"

namespace zimg {;
namespace resize {;

namespace {;

bool resize_h_first(double xscale, double yscale)
{
	double h_first_cost = std::max(xscale, 1.0) * 2.0 + xscale * std::max(yscale, 1.0);
	double v_first_cost = std::max(yscale, 1.0) + yscale * std::max(xscale, 1.0) * 2.0;

	return h_first_cost < v_first_cost;
}

} // namespace


ResizeConversion::ResizeConversion(unsigned src_width, unsigned src_height, PixelType type) :
	src_width{ src_width },
	src_height{ src_height },
	type{ type },
	depth{ (unsigned)default_pixel_format(type).depth },
	filter{},
	dst_width{ src_width },
	dst_height{ src_height },
	shift_w{},
	shift_h{},
	subwidth{ (double)src_width },
	subheight{ (double)src_height },
	cpu{ CPUClass::CPU_NONE }
{
}

auto ResizeConversion::create() const -> filter_pair
{
	bool skip_h = (src_width == dst_width && shift_w == 0 && subwidth == src_width);
	bool skip_v = (src_height == dst_height && shift_h == 0 && subheight == src_height);

	if (skip_h && skip_v) {
		return{ std::unique_ptr<graph::ImageFilter>{ new graph::CopyFilter{ src_width, src_height, type } }, nullptr };
	} else if (skip_h) {
		return{ std::unique_ptr<graph::ImageFilter>{ create_resize_impl(*filter, type, false, depth, src_width, src_height, dst_width, dst_height, shift_h, subheight, cpu) },
			    nullptr };
	} else if (skip_v) {
		return{ std::unique_ptr<graph::ImageFilter>{ create_resize_impl(*filter, type, true, depth, src_width, src_height, dst_width, dst_height, shift_w, subwidth, cpu) },
		        nullptr };
	} else {
		bool h_first = resize_h_first((double)dst_width / src_width, (double)dst_height / src_height);
		std::unique_ptr<graph::ImageFilter> stage1;
		std::unique_ptr<graph::ImageFilter> stage2;

		if (h_first) {
			stage1 = std::unique_ptr<graph::ImageFilter>{ create_resize_impl(*filter, type, true, depth, src_width, src_height, dst_width, src_height, shift_w, subwidth, cpu) };
			stage2 = std::unique_ptr<graph::ImageFilter>{ create_resize_impl(*filter, type, false, depth, dst_width, src_height, dst_width, dst_height, shift_h, subheight, cpu) };
		} else {
			stage1 = std::unique_ptr<graph::ImageFilter>{ create_resize_impl(*filter, type, false, depth, src_width, src_height, src_width, dst_height, shift_h, subheight, cpu) };
			stage2 = std::unique_ptr<graph::ImageFilter>{ create_resize_impl(*filter, type, true, depth, src_width, dst_height, dst_width, dst_height, shift_w, subwidth, cpu) };
		}

		return{ std::move(stage1), std::move(stage2) };
	}
}

} // namespace resize
} // namespace zimg
