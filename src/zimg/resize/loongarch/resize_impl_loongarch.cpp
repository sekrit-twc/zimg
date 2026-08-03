#ifdef ZIMG_LOONGARCH

#include "common/cpuinfo.h"
#include "common/loongarch/cpuinfo_loongarch.h"
#include "graphengine/filter.h"
#include "resize_impl_loongarch.h"

namespace zimg::resize {

std::unique_ptr<graphengine::Filter> create_resize_impl_h_loongarch(const FilterContext &context, unsigned height, PixelType type, unsigned depth, CPUClass cpu)
{
	LoongArchCapabilities caps = query_loongarch_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.lsx)
			ret = create_resize_impl_h_lsx(context, height, type, depth);
	} else {
		if (!ret && cpu >= CPUClass::LOONGARCH_LSX)
			ret = create_resize_impl_h_lsx(context, height, type, depth);
	}

	return ret;
}

std::unique_ptr<graphengine::Filter> create_resize_impl_v_loongarch(const FilterContext &context, unsigned width, PixelType type, unsigned depth, CPUClass cpu)
{
	LoongArchCapabilities caps = query_loongarch_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.lsx)
			ret = create_resize_impl_v_lsx(context, width, type, depth);
	} else {
		if (!ret && cpu >= CPUClass::LOONGARCH_LSX)
			ret = create_resize_impl_v_lsx(context, width, type, depth);
	}

	return ret;
}

} // namespace zimg::resize

#endif // ZIMG_LOONGARCH
