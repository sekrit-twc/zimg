#ifdef ZIMG_ARM

#include "common/cpuinfo.h"
#include "common/arm/cpuinfo_arm.h"
#include "graphengine/filter.h"
#include "resize_impl_arm.h"

namespace zimg {
namespace resize {

std::unique_ptr<graphengine::Filter> create_resize_impl_h_arm(const FilterContext &context, unsigned height, PixelType type, unsigned depth, CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.neon && caps.vfpv4)
			ret = create_resize_impl_h_neon(context, height, type, depth);
	} else {
		if (!ret && cpu >= CPUClass::ARM_NEON)
			ret = create_resize_impl_h_neon(context, height, type, depth);
	}

	return ret;
}

std::unique_ptr<graphengine::Filter> create_resize_impl_v_arm(const FilterContext &context, unsigned width, PixelType type, unsigned depth, CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.neon && caps.vfpv4)
			ret = create_resize_impl_v_neon(context, width, type, depth);
	} else {
		if (!ret && cpu >= CPUClass::ARM_NEON)
			ret = create_resize_impl_v_neon(context, width, type, depth);
	}

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_ARM
