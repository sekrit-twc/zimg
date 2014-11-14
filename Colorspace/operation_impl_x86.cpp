#ifdef ZIMG_X86

#include "Common/cpuinfo.h"
#include "operation_impl_x86.h"

namespace zimg {;
namespace colorspace {;

PixelAdapter *create_pixel_adapter_x86(CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	PixelAdapter *ret;

	if (cpu == CPUClass::CPU_X86_AUTO) {
		if (caps.avx2)
			ret = create_pixel_adapter_avx2();
		else
			ret = nullptr;
	} else if (cpu >= CPUClass::CPU_X86_AVX2) {
		ret = create_pixel_adapter_avx2();
	} else {
		ret = nullptr;
	}

	return ret;
}

Operation *create_matrix_operation_x86(const Matrix3x3 &m, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	Operation *ret;

	if (cpu == CPUClass::CPU_X86_AUTO) {
		if (caps.avx2)
			ret = create_matrix_operation_avx2(m);
		else if (caps.sse2)
			ret = create_matrix_operation_sse2(m);
		else
			ret = nullptr;
	} else if (cpu >= CPUClass::CPU_X86_AVX2) {
		ret = create_matrix_operation_avx2(m);
	} else if (cpu >= CPUClass::CPU_X86_SSE2) {
		ret = create_matrix_operation_sse2(m);
	} else {
		ret = nullptr;
	}

	return ret;
}

Operation *create_rec709_gamma_operation_x86(CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	Operation *ret;

	if (cpu == CPUClass::CPU_X86_AUTO) {
		if (caps.avx2)
			ret = create_rec709_gamma_operation_avx2();
		else
			ret = nullptr;
	} else if (cpu >= CPUClass::CPU_X86_AVX2) {
		ret = create_rec709_gamma_operation_avx2();
	} else {
		ret = nullptr;
	}

	return ret;
}

Operation *create_rec709_inverse_gamma_operation_x86(CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	Operation *ret;

	if (cpu == CPUClass::CPU_X86_AUTO) {
		if (caps.avx2)
			ret = create_rec709_inverse_gamma_operation_avx2();
		else
			ret = nullptr;
	} else if (cpu >= CPUClass::CPU_X86_AVX2) {
		ret = create_rec709_inverse_gamma_operation_avx2();
	} else {
		ret = nullptr;
	}

	return ret;
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86
