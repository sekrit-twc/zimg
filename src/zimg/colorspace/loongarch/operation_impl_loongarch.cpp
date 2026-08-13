#ifdef ZIMG_LOONGARCH

#include "common/cpuinfo.h"
#include "common/loongarch/cpuinfo_loongarch.h"
#include "colorspace/operation.h"
#include "colorspace/operation_impl.h"
#include "operation_impl_loongarch.h"

namespace zimg::colorspace {

std::unique_ptr<Operation> create_matrix_operation_loongarch(const Matrix3x3 &m, CPUClass cpu)
{
	LoongArchCapabilities caps = query_loongarch_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.lsx)
			ret = create_matrix_operation_lsx(m);
	} else {
		if (!ret && cpu >= CPUClass::LOONGARCH_LSX)
			ret = create_matrix_operation_lsx(m);
	}

	return ret;
}

std::unique_ptr<Operation> create_gamma_operation_loongarch(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu)
{
	LoongArchCapabilities caps = query_loongarch_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.lsx)
			ret = create_gamma_operation_lsx(transfer, params);
	} else {
		if (!ret && cpu >= CPUClass::LOONGARCH_LSX)
			ret = create_gamma_operation_lsx(transfer, params);
	}

	return ret;
}

std::unique_ptr<Operation> create_inverse_gamma_operation_loongarch(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu)
{
	LoongArchCapabilities caps = query_loongarch_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.lsx)
			ret = create_inverse_gamma_operation_lsx(transfer, params);
	} else {
		if (!ret && cpu >= CPUClass::LOONGARCH_LSX)
			ret = create_inverse_gamma_operation_lsx(transfer, params);
	}

	return ret;
}

} // namespace zimg::colorspace

#endif // ZIMG_LOONGARCH
