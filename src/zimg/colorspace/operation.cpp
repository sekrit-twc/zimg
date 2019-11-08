#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/zassert.h"
#include "colorspace.h"
#include "colorspace_param.h"
#include "gamma.h"
#include "operation.h"
#include "operation_impl.h"

namespace zimg {
namespace colorspace {

namespace {

bool use_display_referred_b67(ColorPrimaries primaries, const OperationParams &params)
{
	return primaries != ColorPrimaries::UNSPECIFIED && !params.approximate_gamma && !params.scene_referred;
}

} // namespace


Operation::~Operation() = default;

std::unique_ptr<Operation> create_ncl_yuv_to_rgb_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu)
{
	zassert_d(in.transfer == out.transfer, "transfer mismatch");
	zassert_d(in.primaries == out.primaries, "primaries mismatch");
	zassert_d(in.matrix != MatrixCoefficients::RGB && out.matrix == MatrixCoefficients::RGB, "wrong matrix coefficients");
	zassert_d(in.matrix != MatrixCoefficients::REC_2020_CL, "wrong matrix coefficients");

	Matrix3x3 m = in.matrix == MatrixCoefficients::CHROMATICITY_DERIVED_NCL ? ncl_yuv_to_rgb_matrix_from_primaries(in.primaries) : ncl_yuv_to_rgb_matrix(in.matrix);
	return create_matrix_operation(m, cpu);
}

std::unique_ptr<Operation> create_ncl_rgb_to_yuv_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu)
{
	zassert_d(in.transfer == out.transfer, "transfer mismatch");
	zassert_d(in.primaries == out.primaries, "primaries mismatch");
	zassert_d(in.matrix == MatrixCoefficients::RGB && out.matrix != MatrixCoefficients::RGB, "wrong matrix coefficients");
	zassert_d(out.matrix != MatrixCoefficients::REC_2020_CL, "wrong matrix coefficients");

	Matrix3x3 m = out.matrix == MatrixCoefficients::CHROMATICITY_DERIVED_NCL ? ncl_rgb_to_yuv_matrix_from_primaries(out.primaries) : ncl_rgb_to_yuv_matrix(out.matrix);
	return create_matrix_operation(m, cpu);
}

std::unique_ptr<Operation> create_ictcp_to_lms_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu)
{
	zassert_d(in.transfer == out.transfer, "transfer mismatch");
	zassert_d(in.primaries == out.primaries, "primaries mismatch");
	zassert_d(in.matrix == MatrixCoefficients::REC_2100_ICTCP && out.matrix == MatrixCoefficients::REC_2100_LMS, "wrong matrix coefficients");

	return create_matrix_operation(ictcp_to_lms_matrix(in.transfer), cpu);
}

std::unique_ptr<Operation> create_lms_to_ictcp_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu)
{
	zassert_d(in.transfer == out.transfer, "transfer mismatch");
	zassert_d(in.primaries == out.primaries, "primaries mismatch");
	zassert_d(in.matrix == MatrixCoefficients::REC_2100_LMS && out.matrix == MatrixCoefficients::REC_2100_ICTCP, "wrong matrix coefficients");

	return create_matrix_operation(lms_to_ictcp_matrix(in.transfer), cpu);
}

std::unique_ptr<Operation> create_gamma_to_linear_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu)
{
	zassert_d(in.primaries == out.primaries, "primaries mismatch");
	zassert_d(in.matrix == MatrixCoefficients::RGB && out.matrix == MatrixCoefficients::RGB, "must be RGB");
	zassert_d(in.transfer != TransferCharacteristics::LINEAR && out.transfer == TransferCharacteristics::LINEAR, "wrong transfer characteristics");

	if (in.transfer == TransferCharacteristics::ARIB_B67 && use_display_referred_b67(in.primaries, params))
		return create_inverse_arib_b67_operation(ncl_rgb_to_yuv_matrix_from_primaries(in.primaries), params);
	else
		return create_inverse_gamma_operation(select_transfer_function(in.transfer, params.peak_luminance, params.scene_referred), params, cpu);
}

std::unique_ptr<Operation> create_linear_to_gamma_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu)
{
	zassert_d(in.primaries == out.primaries, "primaries mismatch");
	zassert_d(in.matrix == MatrixCoefficients::RGB && out.matrix == MatrixCoefficients::RGB, "must be RGB");
	zassert_d(in.transfer == TransferCharacteristics::LINEAR && out.transfer != TransferCharacteristics::LINEAR, "wrong transfer characteristics");

	if (out.transfer == TransferCharacteristics::ARIB_B67 && use_display_referred_b67(out.primaries, params))
		return create_arib_b67_operation(ncl_rgb_to_yuv_matrix_from_primaries(out.primaries), params);
	else
		return create_gamma_operation(select_transfer_function(out.transfer, params.peak_luminance, params.scene_referred), params, cpu);
}

std::unique_ptr<Operation> create_gamut_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu)
{
	zassert_d(in.matrix == MatrixCoefficients::RGB && in.transfer == TransferCharacteristics::LINEAR, "must be linear RGB");
	zassert_d(out.matrix == MatrixCoefficients::RGB && out.transfer == TransferCharacteristics::LINEAR, "must be linear RGB");

	Matrix3x3 m = gamut_xyz_to_rgb_matrix(out.primaries) * white_point_adaptation_matrix(in.primaries, out.primaries) * gamut_rgb_to_xyz_matrix(in.primaries);
	return create_matrix_operation(m, cpu);
}

} // namespace colorspace
} // namespace zimg
