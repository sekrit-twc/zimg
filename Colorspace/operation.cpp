#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "colorspace_param.h"
#include "operation.h"
#include "operation_impl.h"

namespace zimg {;
namespace colorspace {;

PixelAdapter::~PixelAdapter()
{
}

Operation::~Operation()
{
}

Operation *create_ncl_yuv_to_rgb_operation(MatrixCoefficients matrix, CPUClass cpu)
{
	return create_matrix_operation(ncl_yuv_to_rgb_matrix(matrix), cpu);
}

Operation *create_ncl_rgb_to_yuv_operation(MatrixCoefficients matrix, CPUClass cpu)
{
	return create_matrix_operation(ncl_rgb_to_yuv_matrix(matrix), cpu);
}

Operation *create_gamma_to_linear_operation(TransferCharacteristics transfer, CPUClass cpu)
{
	switch (transfer) {
	case TransferCharacteristics::TRANSFER_709:
		return create_rec709_inverse_gamma_operation(cpu);
	default:
		throw ZimgUnsupportedError{ "unsupported transfer function" };
	}
}

Operation *create_linear_to_gamma_operation(TransferCharacteristics transfer, CPUClass cpu)
{
	switch (transfer) {
	case TransferCharacteristics::TRANSFER_709:
		return create_rec709_gamma_operation(cpu);
	default:
		throw ZimgUnsupportedError{ "unsupported transfer function" };
	}
}

Operation *create_gamut_operation(ColorPrimaries primaries_in, ColorPrimaries primaries_out, CPUClass cpu)
{
	return create_matrix_operation(gamut_rgb_to_xyz_matrix(primaries_in) * gamut_xyz_to_rgb_matrix(primaries_out), cpu);
}

} // namespace colorspace
} // namespace zimg
