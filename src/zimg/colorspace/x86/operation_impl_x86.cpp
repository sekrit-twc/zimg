#ifdef ZIMG_X86

#include "common/cpuinfo.h"
#include "common/x86/cpuinfo_x86.h"
#include "colorspace/operation.h"
#include "colorspace/operation_impl.h"
#include "operation_impl_x86.h"

namespace zimg::colorspace {

std::unique_ptr<Operation> create_matrix_operation_x86(const Matrix3x3 &m, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f)
			ret = create_matrix_operation_avx512(m);
		if (!ret && caps.avx2)
			ret = create_matrix_operation_avx2(m);
	} else {
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = create_matrix_operation_avx512(m);
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_matrix_operation_avx2(m);
	}

	return ret;
}

std::unique_ptr<Operation> create_gamma_operation_x86(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f && caps.avx512bw && caps.avx512dq)
			ret = create_gamma_operation_avx512(transfer, params);
		if (!ret && caps.avx2)
			ret = create_gamma_operation_avx2(transfer, params);
	} else {
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = create_gamma_operation_avx512(transfer, params);
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_gamma_operation_avx2(transfer, params);
	}

	return ret;
}

std::unique_ptr<Operation> create_inverse_gamma_operation_x86(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu)
{
	X86Capabilities caps = query_x86_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && cpu == CPUClass::AUTO_64B && caps.avx512f && caps.avx512bw && caps.avx512dq)
			ret = create_inverse_gamma_operation_avx512(transfer, params);
		if (!ret && caps.avx2)
			ret = create_inverse_gamma_operation_avx2(transfer, params);
	} else {
		if (!ret && cpu >= CPUClass::X86_AVX512)
			ret = create_inverse_gamma_operation_avx512(transfer, params);
		if (!ret && cpu >= CPUClass::X86_AVX2)
			ret = create_inverse_gamma_operation_avx2(transfer, params);
	}

	return ret;
}

} // namespace zimg::colorspace

#endif // ZIMG_X86
