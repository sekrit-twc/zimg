#ifdef ZIMG_X86

#include "common/cpuinfo.h"
#include "operation.h"
#include "operation_impl.h"
#include "operation_impl_x86.h"

namespace zimg {
namespace colorspace {

std::unique_ptr<Operation> create_matrix_operation_x86(const Matrix3x3 &m, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu == CPUClass::AUTO) {
		if (!ret && caps.avx)
			ret = create_matrix_operation_avx(m);
		if (!ret && caps.sse)
			ret = create_matrix_operation_sse(m);
	} else {
		if (!ret && cpu >= CPUClass::X86_AVX)
			ret = create_matrix_operation_avx(m);
		if (!ret && cpu >= CPUClass::X86_SSE)
			ret = create_matrix_operation_sse(m);
	}

	return ret;
}

std::unique_ptr<Operation> create_gamma_to_linear_operation_x86(TransferCharacteristics transfer, const OperationParams &params, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu == CPUClass::AUTO) {
		if (!ret && caps.avx2 && caps.f16c)
			ret = create_gamma_to_linear_operation_avx2(transfer, params);
		if (!ret && caps.sse2)
			ret = create_gamma_to_linear_operation_sse2(transfer, params);
	} else {
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_gamma_to_linear_operation_avx2(transfer, params);
		if (!ret && cpu >= CPUClass::X86_SSE2)
			ret = create_gamma_to_linear_operation_sse2(transfer, params);
	}

	return ret;
}

std::unique_ptr<Operation> create_linear_to_gamma_operation_x86(TransferCharacteristics transfer, const OperationParams &params, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu == CPUClass::AUTO) {
		if (!ret && caps.avx2 && caps.f16c)
			ret = create_linear_to_gamma_operation_avx2(transfer, params);
		if (!ret && caps.sse2)
			ret = create_linear_to_gamma_operation_sse2(transfer, params);
	} else {
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_linear_to_gamma_operation_avx2(transfer, params);
		if (!ret && cpu >= CPUClass::X86_SSE2)
			ret = create_linear_to_gamma_operation_sse2(transfer, params);
	}

	return ret;
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86
