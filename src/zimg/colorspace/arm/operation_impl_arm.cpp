#ifdef ZIMG_ARM

#include "common/cpuinfo.h"
#include "common/arm/cpuinfo_arm.h"
#include "colorspace/operation.h"
#include "colorspace/operation_impl.h"
#include "operation_impl_arm.h"

namespace zimg {
namespace colorspace {

std::unique_ptr<Operation> create_matrix_operation_arm(const Matrix3x3 &m, CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.neon && caps.vfpv4)
			ret = create_matrix_operation_neon(m);
	} else {
		if (!ret && cpu >= CPUClass::ARM_NEON)
			ret = create_matrix_operation_neon(m);
	}

	return ret;
}

std::unique_ptr<Operation> create_gamma_operation_arm(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.neon && caps.vfpv4)
			ret = create_gamma_operation_neon(transfer, params);
	} else {
		if (!ret && cpu >= CPUClass::ARM_NEON)
			ret = create_gamma_operation_neon(transfer, params);
	}

	return ret;
}

std::unique_ptr<Operation> create_inverse_gamma_operation_arm(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu)
{
	ARMCapabilities caps = query_arm_capabilities();
	std::unique_ptr<Operation> ret;

	if (cpu_is_autodetect(cpu)) {
		if (!ret && caps.neon && caps.vfpv4)
			ret = create_inverse_gamma_operation_neon(transfer, params);
	} else {
		if (!ret && cpu >= CPUClass::ARM_NEON)
			ret = create_inverse_gamma_operation_neon(transfer, params);
	}

	return ret;
}

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_ARM
