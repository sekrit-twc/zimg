#include <algorithm>
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "graph/basic_filter.h"
#include "unresize.h"
#include "unresize_impl.h"

namespace zimg {
namespace unresize {

namespace {

bool unresize_h_first(double xscale, double yscale) noexcept
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
	cpu{ CPUClass::NONE }
{}

auto UnresizeConversion::create() const -> filter_pair try
{
	if (up_width > pixel_max_width(PixelType::FLOAT) || orig_width > pixel_max_width(PixelType::FLOAT))
		throw error::OutOfMemory{};

	bool skip_h = (up_width == orig_width && shift_w == 0);
	bool skip_v = (up_height == orig_height && shift_h == 0);

	if (skip_h && skip_v)
		return{ ztd::make_unique<graph::CopyFilter>(up_width, up_height, type), nullptr };

	auto builder = UnresizeImplBuilder{ up_width, up_height, type }.set_cpu(cpu);
	filter_pair ret{};

	if (skip_h) {
		ret.first = builder.set_horizontal(false)
		                   .set_orig_dim(orig_height)
		                   .set_shift(shift_h)
		                   .create();
	} else if (skip_v) {
		ret.first = builder.set_horizontal(true)
		                   .set_orig_dim(orig_width)
		                   .set_shift(shift_w)
		                   .create();
	} else {
		bool h_first = unresize_h_first(static_cast<double>(orig_width) / up_width, static_cast<double>(orig_height) / up_height);

		if (h_first) {
			ret.first = builder.set_horizontal(true)
			                   .set_orig_dim(orig_width)
			                   .set_shift(shift_w)
			                   .create();

			builder.up_width = orig_width;
			ret.second = builder.set_horizontal(false)
			                    .set_orig_dim(orig_height)
			                    .set_shift(shift_h)
			                    .create();
		} else {
			ret.first = builder.set_horizontal(false)
			                   .set_orig_dim(orig_height)
			                   .set_shift(shift_h)
			                   .create();

			builder.up_height = orig_height;
			ret.second = builder.set_horizontal(true)
			                    .set_orig_dim(orig_width)
			                    .set_shift(shift_w)
			                    .create();
		}
	}

	return ret;
} catch (const std::bad_alloc &) {
	throw error::OutOfMemory{};
}

} // namespace unresize
} // namespace zimg
