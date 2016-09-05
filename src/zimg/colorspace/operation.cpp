#include "common/except.h"
#include "colorspace.h"
#include "colorspace_param.h"
#include "operation.h"
#include "operation_impl.h"

namespace zimg {
namespace colorspace {

Operation::~Operation() = default;

std::unique_ptr<Operation> create_ncl_yuv_to_rgb_operation(MatrixCoefficients matrix, const OperationParams &params, CPUClass cpu)
{
	return create_matrix_operation(ncl_yuv_to_rgb_matrix(matrix), cpu);
}

std::unique_ptr<Operation> create_ncl_rgb_to_yuv_operation(MatrixCoefficients matrix, const OperationParams &params, CPUClass cpu)
{
	return create_matrix_operation(ncl_rgb_to_yuv_matrix(matrix), cpu);
}

std::unique_ptr<Operation> create_gamma_to_linear_operation(TransferCharacteristics transfer, const OperationParams &params, CPUClass cpu)
{
	switch (transfer) {
	case TransferCharacteristics::REC_709:
		return create_rec709_inverse_gamma_operation(cpu);
	default:
		throw error::InternalError{ "unsupported transfer function" };
	}
}

std::unique_ptr<Operation> create_linear_to_gamma_operation(TransferCharacteristics transfer, const OperationParams &params, CPUClass cpu)
{
	switch (transfer) {
	case TransferCharacteristics::REC_709:
		return create_rec709_gamma_operation(cpu);
	default:
		throw error::InternalError{ "unsupported transfer function" };
	}
}

std::unique_ptr<Operation> create_gamut_operation(ColorPrimaries primaries_in, ColorPrimaries primaries_out, const OperationParams &params, CPUClass cpu)
{
	return create_matrix_operation(gamut_xyz_to_rgb_matrix(primaries_out) * gamut_rgb_to_xyz_matrix(primaries_in), cpu);
}

} // namespace colorspace
} // namespace zimg
