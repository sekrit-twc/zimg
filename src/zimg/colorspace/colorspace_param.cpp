#include <cstring>
#include "common/except.h"
#include "colorspace_param.h"

namespace zimg {;
namespace colorspace {;

namespace {;

void get_yuv_constants(double *kr, double *kb, MatrixCoefficients matrix)
{
	switch (matrix) {
	case MatrixCoefficients::MATRIX_RGB:
		*kr = 0;
		*kb = 0;
		break;
	case MatrixCoefficients::MATRIX_601:
		*kr = REC_601_KR;
		*kb = REC_601_KB;
		break;
	case MatrixCoefficients::MATRIX_709:
		*kr = REC_709_KR;
		*kb = REC_709_KB;
		break;
	case MatrixCoefficients::MATRIX_2020_NCL:
	case MatrixCoefficients::MATRIX_2020_CL:
		*kr = REC_2020_KR;
		*kb = REC_2020_KB;
		break;
	default:
		throw error::IllegalArgument{ "unrecognized matrix coefficients" };
	}
}

Vector3 xy_to_xyz(double x, double y)
{
	Vector3 ret;

	ret[0] = x / y;
	ret[1] = 1.0;
	ret[2] = (1.0 - x - y) / y;

	return ret;
}

Vector3 get_d65_xyz()
{
	return xy_to_xyz(ILLUMINANT_D65[0], ILLUMINANT_D65[1]);
}

void get_primaries_xy(double out[3][2], ColorPrimaries primaries)
{
	switch (primaries) {
	case ColorPrimaries::PRIMARIES_SMPTE_C:
		memcpy(out, SMPTE_C_PRIMARIES, sizeof(SMPTE_C_PRIMARIES));
		break;
	case ColorPrimaries::PRIMARIES_709:
		memcpy(out, REC_709_PRIMARIES, sizeof(REC_709_PRIMARIES));
		break;
	case ColorPrimaries::PRIMARIES_2020:
		memcpy(out, REC_2020_PRIMARIES, sizeof(REC_2020_PRIMARIES));
		break;
	default:
		throw error::IllegalArgument{ "unrecognized primaries" };
	}
}

Matrix3x3 get_primaries_xyz(ColorPrimaries primaries)
{
	// Columns: R G B
	// Rows: X Y Z
	Matrix3x3 ret;
	double primaries_xy[3][2];

	get_primaries_xy(primaries_xy, primaries);

	ret[0] = xy_to_xyz(primaries_xy[0][0], primaries_xy[0][1]);
	ret[1] = xy_to_xyz(primaries_xy[1][0], primaries_xy[1][1]);
	ret[2] = xy_to_xyz(primaries_xy[2][0], primaries_xy[2][1]);

	return transpose(ret);
}

} // namespace


ColorspaceDefinition ColorspaceDefinition::to(MatrixCoefficients matrix_) const
{
	return{ matrix_, transfer, primaries };
}

ColorspaceDefinition ColorspaceDefinition::to(TransferCharacteristics transfer_) const
{
	return{ matrix, transfer_, primaries };
}

ColorspaceDefinition ColorspaceDefinition::to(ColorPrimaries primaries_) const
{
	return{ matrix, transfer, primaries_ };
}

ColorspaceDefinition ColorspaceDefinition::toRGB() const
{
	return to(MatrixCoefficients::MATRIX_RGB);
}

ColorspaceDefinition ColorspaceDefinition::toLinear() const
{
	return to(TransferCharacteristics::TRANSFER_LINEAR);
}

bool operator==(const ColorspaceDefinition &a, const ColorspaceDefinition &b)
{
	return a.matrix == b.matrix && a.primaries == b.primaries && a.transfer == b.transfer;
}

bool operator!=(const ColorspaceDefinition &a, const ColorspaceDefinition &b)
{
	return !operator==(a, b);
}

Matrix3x3 ncl_yuv_to_rgb_matrix(MatrixCoefficients matrix)
{
	return inverse(ncl_rgb_to_yuv_matrix(matrix));
}

Matrix3x3 ncl_rgb_to_yuv_matrix(MatrixCoefficients matrix)
{
	Matrix3x3 ret;

	if (matrix != MatrixCoefficients::MATRIX_YCGCO) {
		double kr, kg, kb;
		double uscale;
		double vscale;

		get_yuv_constants(&kr, &kb, matrix);
		kg = 1.0 - kr - kb;
		uscale = 1.0 / (2.0 - 2.0 * kb);
		vscale = 1.0 / (2.0 - 2.0 * kr);

		ret[0][0] = kr;
		ret[0][1] = kg;
		ret[0][2] = kb;

		ret[1][0] = -kr * uscale;
		ret[1][1] = -kg * uscale;
		ret[1][2] = (1.0 - kb) * uscale;

		ret[2][0] = (1.0 - kr) * vscale;
		ret[2][1] = -kg * vscale;
		ret[2][2] = -kb * vscale;
	} else {
		ret = {
			{  0.25, 0.5,  0.25 },
			{ -0.25, 0.5, -0.25 },
			{  0.5,  0,   -0.5 }
		};
	}

	return ret;
}

// http://www.brucelindbloom.com/Eqn_RGB_XYZ_Matrix.html
Matrix3x3 gamut_rgb_to_xyz_matrix(ColorPrimaries primaries)
{
	Matrix3x3 xyz_matrix = get_primaries_xyz(primaries);
	Vector3 white_xyz = get_d65_xyz();

	Vector3 s = inverse(xyz_matrix) * white_xyz;
	Matrix3x3 m = { xyz_matrix[0] * s, xyz_matrix[1] * s, xyz_matrix[2] * s };

	return m;
}

Matrix3x3 gamut_xyz_to_rgb_matrix(ColorPrimaries primaries)
{
	return inverse(gamut_rgb_to_xyz_matrix(primaries));
}

} // namespace colorspace
} // namespace zimg
