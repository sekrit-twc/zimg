#pragma once

#ifndef ZIMG_COLORSPACE_COLORSPACE_H_
#define ZIMG_COLORSPACE_COLORSPACE_H_

#include <memory>

namespace zimg {

enum class CPUClass;

namespace graph {

class ImageFilter;

} // namespace graph


namespace colorspace {

enum class MatrixCoefficients {
	UNSPECIFIED,
	RGB,
	REC_601,
	REC_709,
	FCC,
	SMPTE_240M,
	YCGCO,
	REC_2020_NCL,
	REC_2020_CL,
	CHROMATICITY_DERIVED_NCL,
	CHROMATICITY_DERIVED_CL,
	REC_2100_LMS,
	REC_2100_ICTCP,
};

enum class TransferCharacteristics {
	UNSPECIFIED,
	LINEAR,
	LOG_100,
	LOG_316,
	REC_709,
	REC_470_M,
	REC_470_BG,
	SMPTE_240M,
	XVYCC,
	SRGB,
	ST_2084,
	ARIB_B67,
};

enum class ColorPrimaries {
	UNSPECIFIED,
	REC_470_M,
	REC_470_BG,
	SMPTE_C,
	REC_709,
	FILM,
	REC_2020,
	ST_428,
	DCI_P3,
	DCI_P3_D65,
	JEDEC_P22,
};

/**
 * Definition of a working colorspace.
 */
struct ColorspaceDefinition {
	MatrixCoefficients matrix;
	TransferCharacteristics transfer;
	ColorPrimaries primaries;

	// Helper functions to create modified colorspaces.
	constexpr ColorspaceDefinition to(MatrixCoefficients matrix_) const noexcept
	{
		return{ matrix_, transfer, primaries };
	}

	constexpr ColorspaceDefinition to(TransferCharacteristics transfer_) const noexcept
	{
		return{ matrix, transfer_, primaries };
	}

	constexpr ColorspaceDefinition to(ColorPrimaries primaries_) const noexcept
	{
		return{ matrix, transfer, primaries_ };
	}

	constexpr ColorspaceDefinition to_rgb() const noexcept
	{
		return to(MatrixCoefficients::RGB);
	}

	constexpr ColorspaceDefinition to_linear() const noexcept
	{
		return to(TransferCharacteristics::LINEAR);
	}
};

// Compare colorspaces by comparing each component.
constexpr bool operator==(const ColorspaceDefinition &a, const ColorspaceDefinition &b) noexcept
{
	return a.matrix == b.matrix && a.transfer == b.transfer && a.primaries == b.primaries;
}

constexpr bool operator!=(const ColorspaceDefinition &a, const ColorspaceDefinition &b) noexcept
{
	return !(a == b);
}


constexpr bool is_valid_2020cl(const ColorspaceDefinition &csp)
{
	return csp.matrix == MatrixCoefficients::REC_2020_CL && csp.transfer == TransferCharacteristics::REC_709;
}

constexpr bool is_valid_ictcp(const ColorspaceDefinition &csp)
{
	return csp.matrix == MatrixCoefficients::REC_2100_ICTCP &&
		(csp.transfer == TransferCharacteristics::ST_2084 || csp.transfer == TransferCharacteristics::ARIB_B67) &&
		csp.primaries == ColorPrimaries::REC_2020;
}

constexpr bool is_valid_lms(const ColorspaceDefinition &csp)
{
	return csp.matrix == MatrixCoefficients::REC_2100_LMS &&
		(csp.transfer == TransferCharacteristics::LINEAR || csp.transfer == TransferCharacteristics::ST_2084 || csp.transfer == TransferCharacteristics::ARIB_B67) &&
		csp.primaries == ColorPrimaries::REC_2020;
}

constexpr bool is_valid_csp(const ColorspaceDefinition &csp)
{
	// 1. Require matrix to be set if transfer is set.
	// 2. Require transfer to be set if primaries is set.
	// 3. Check requirements for Rec.2020 CL.
	// 4. Check requirements for chromaticity-derived NCL matrix.
	// 5. Check requirements for chromaticity-derived CL matrix.
	// 6. Check requirements for Rec.2100 ICtCp.
	// 7. Check requirements for Rec.2100 LMS.
	return !(csp.matrix == MatrixCoefficients::UNSPECIFIED && csp.transfer != TransferCharacteristics::UNSPECIFIED) &&
		!(csp.transfer == TransferCharacteristics::UNSPECIFIED && csp.primaries != ColorPrimaries::UNSPECIFIED) &&
		!(csp.matrix == MatrixCoefficients::REC_2020_CL && !is_valid_2020cl(csp)) &&
		!(csp.matrix == MatrixCoefficients::CHROMATICITY_DERIVED_NCL && csp.primaries == ColorPrimaries::UNSPECIFIED) &&
		!(csp.matrix == MatrixCoefficients::CHROMATICITY_DERIVED_CL && csp.primaries == ColorPrimaries::UNSPECIFIED) &&
		!(csp.matrix == MatrixCoefficients::REC_2100_ICTCP && !is_valid_ictcp(csp)) &&
		!(csp.matrix == MatrixCoefficients::REC_2100_LMS && !is_valid_lms(csp));
}


struct ColorspaceConversion {
	unsigned width;
	unsigned height;

#include "common/builder.h"
	BUILDER_MEMBER(ColorspaceDefinition, csp_in)
	BUILDER_MEMBER(ColorspaceDefinition, csp_out)
	BUILDER_MEMBER(double, peak_luminance)
	BUILDER_MEMBER(bool, approximate_gamma)
	BUILDER_MEMBER(bool, scene_referred)
	BUILDER_MEMBER(CPUClass, cpu)
#undef BUILDER_MEMBER

	ColorspaceConversion(unsigned width, unsigned height);

	std::unique_ptr<graph::ImageFilter> create() const;
};

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_COLORSPACE2_H_
