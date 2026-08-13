#ifdef ZIMG_LOONGARCH

#include "common/cpuinfo.h"
#include "common/loongarch/cpuinfo_loongarch.h"
#include "graphengine/filter.h"
#include "unresize_impl_loongarch.h"

namespace zimg::unresize {

std::unique_ptr<graphengine::Filter> create_unresize_impl_h_loongarch(const BilinearContext &context, unsigned height, PixelType type, CPUClass cpu)
{
	LoongArchCapabilities caps = query_loongarch_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.lsx)
			ret = create_unresize_impl_h_lsx(context, height, type);
	} else {
		if (!ret && cpu >= CPUClass::LOONGARCH_LSX)
			ret = create_unresize_impl_h_lsx(context, height, type);
	}

	return ret;
}

std::unique_ptr<graphengine::Filter> create_unresize_impl_v_loongarch(const BilinearContext &context, unsigned width, PixelType type, CPUClass cpu)
{
	LoongArchCapabilities caps = query_loongarch_capabilities();
	std::unique_ptr<graphengine::Filter> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.lsx)
			ret = create_unresize_impl_v_lsx(context, width, type);
	} else {
		if (!ret && cpu >= CPUClass::LOONGARCH_LSX)
			ret = create_unresize_impl_v_lsx(context, width, type);
	}

	return ret;
}

} // namespace zimg::unresize

#endif // ZIMG_LOONGARCH
