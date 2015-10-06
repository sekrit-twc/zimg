#include <algorithm>
#include <memory>
#include "common/cpuinfo.h"
#include "graph/copy_filter.h"
#include "unresize.h"
#include "unresize_impl.h"

namespace zimg {;
namespace unresize {;

namespace {;

bool unresize_h_first(double xscale, double yscale)
{
	double h_first_cost = std::max(xscale, 1.0) * 2.0 + xscale * std::max(yscale, 1.0);
	double v_first_cost = std::max(yscale, 1.0) + yscale * std::max(xscale, 1.0) * 2.0;

	return h_first_cost < v_first_cost;
}

} // namespace


UnresizeConversion::UnresizeConversion(unsigned up_width, unsigned up_height, PixelType type) :
	up_width{ up_width },
	up_height{ up_height },
	type{ type },
	orig_width{ up_width },
	orig_height{ up_height },
	shift_w{},
	shift_h{},
	cpu{ CPUClass::CPU_NONE }
{
}

auto UnresizeConversion::create() const -> filter_pair
{
	bool skip_h = (up_width == orig_width && shift_w == 0);
	bool skip_v = (up_height == orig_height && shift_h == 0);

	if (skip_h && skip_v) {
		return{ std::unique_ptr<graph::ImageFilter>{ new graph::CopyFilter{ up_width, up_height, type } }, nullptr };
	} else if (skip_h) {
		return{ std::unique_ptr<graph::ImageFilter>{ create_unresize_impl(type, false, up_width, up_height, orig_width, orig_height, shift_w, cpu) },
		        nullptr };
	} else if (skip_v) {
		return{ std::unique_ptr<graph::ImageFilter>{ create_unresize_impl(type, true, up_width, up_height, orig_width, orig_height, shift_h, cpu) },
		        nullptr };
	} else {
		bool h_first = unresize_h_first((double)orig_width / up_width, (double)orig_height / up_height);
		std::unique_ptr<graph::ImageFilter> stage1;
		std::unique_ptr<graph::ImageFilter> stage2;

		if (h_first) {
			stage1.reset(create_unresize_impl(type, true, up_width, up_height, orig_width, up_height, shift_w, cpu));
			stage2.reset(create_unresize_impl(type, false, orig_width, up_height, orig_width, orig_height, shift_h, cpu));
		} else {
			stage1.reset(create_unresize_impl(type, false, up_width, up_height, up_width, orig_height, shift_h, cpu));
			stage2.reset(create_unresize_impl(type, true, up_width, orig_height, orig_width, orig_height, shift_w, cpu));
		}

		return{ std::move(stage1), std::move(stage2) };
	}
}

} // namespace unresize
} // namespace zimg
