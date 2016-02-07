#ifdef ZIMG_X86

#include "common/cpuinfo.h"
#include "operation.h"
#include "operation_impl_x86.h"

namespace zimg {
namespace colorspace {

std::unique_ptr<Operation> create_matrix_operation_x86(const Matrix3x3 &m, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu == CPUClass::CPU_AUTO) {
		if (!ret && caps.avx)
			ret = create_matrix_operation_avx(m);
		if (!ret && caps.sse)
			ret = create_matrix_operation_sse(m);
	} else {
		if (!ret && cpu >= CPUClass::CPU_X86_AVX)
			ret = create_matrix_operation_avx(m);
		if (!ret && cpu >= CPUClass::CPU_X86_SSE)
			ret = create_matrix_operation_sse(m);
	}

	return ret;
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86
