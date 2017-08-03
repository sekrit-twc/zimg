#include "common/except.h"
#include "common/zassert.h"
#include "colorspace.h"
#include "colorspace_param.h"
#include "operation.h"
#include "operation_impl.h"

#ifdef ZIMG_X86
  #include "x86/operation_impl_x86.h"
#endif

namespace zimg {
namespace colorspace {

namespace {

bool use_display_referred_b67(ColorPrimaries primaries, const OperationParams &params)
{
	return primaries != ColorPrimaries::UNSPECIFIED && !params.approximate_gamma && !params.scene_referred;
}

} // namespace


Operation::~Operation() = default;

std::unique_ptr<Operation> create_ncl_yuv_to_rgb_operation(MatrixCoefficients matrix, const OperationParams &params, CPUClass cpu)
{
	return create_matrix_operation(ncl_yuv_to_rgb_matrix(matrix), cpu);
}

std::unique_ptr<Operation> create_ncl_rgb_to_yuv_operation(MatrixCoefficients matrix, const OperationParams &params, CPUClass cpu)
{
	return create_matrix_operation(ncl_rgb_to_yuv_matrix(matrix), cpu);
}

std::unique_ptr<Operation> create_ictcp_to_lms_operation(const OperationParams &params, CPUClass cpu)
{
	return create_matrix_operation(ictcp_to_lms_matrix(), cpu);
}

std::unique_ptr<Operation> create_lms_to_ictcp_operation(const OperationParams &params, CPUClass cpu)
{
	return create_matrix_operation(lms_to_ictcp_matrix(), cpu);
}

std::unique_ptr<Operation> create_gamma_to_linear_operation(TransferCharacteristics transfer, ColorPrimaries primaries, const OperationParams &params, CPUClass cpu)
{
	zassert_d(transfer != TransferCharacteristics::LINEAR, "linear op");

	if (transfer == TransferCharacteristics::ARIB_B67 && use_display_referred_b67(primaries, params))
		return create_inverse_arib_b67_operation(ncl_rgb_to_yuv_matrix_from_primaries(primaries), params);

#ifdef ZIMG_X86
	if (std::unique_ptr<Operation> op = create_gamma_to_linear_operation_x86(transfer, params, cpu))
		return op;
#endif
	return create_inverse_gamma_operation(transfer, params);
}

std::unique_ptr<Operation> create_linear_to_gamma_operation(TransferCharacteristics transfer, ColorPrimaries primaries, const OperationParams &params, CPUClass cpu)
{
	zassert_d(transfer != TransferCharacteristics::LINEAR, "linear op");

	if (transfer == TransferCharacteristics::ARIB_B67 && use_display_referred_b67(primaries, params))
		return create_arib_b67_operation(ncl_rgb_to_yuv_matrix_from_primaries(primaries), params);

#ifdef ZIMG_X86
	if (std::unique_ptr<Operation> op = create_linear_to_gamma_operation_x86(transfer, params, cpu))
		return op;
#endif
	return create_gamma_operation(transfer, params);
}

std::unique_ptr<Operation> create_gamut_operation(ColorPrimaries primaries_in, ColorPrimaries primaries_out, const OperationParams &params, CPUClass cpu)
{
	return create_matrix_operation(gamut_xyz_to_rgb_matrix(primaries_out) * gamut_rgb_to_xyz_matrix(primaries_in), cpu);
}

} // namespace colorspace
} // namespace zimg
