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
	YCGCO,
	REC_2020_NCL,
	REC_2020_CL,
};

enum class TransferCharacteristics {
	UNSPECIFIED,
	LINEAR,
	REC_709,
	ST_2084,
	ARIB_B67,
};

enum class ColorPrimaries {
	UNSPECIFIED,
	SMPTE_C,
	REC_709,
	REC_2020,
	DCI_P3_D65,
};

/**
 * Definition of a working colorspace.
 */
struct ColorspaceDefinition {
	MatrixCoefficients matrix;
	TransferCharacteristics transfer;
	ColorPrimaries primaries;

	// Helper functions to create modified colorspaces.
	ColorspaceDefinition to(MatrixCoefficients matrix) const;
	ColorspaceDefinition to(TransferCharacteristics transfer) const;
	ColorspaceDefinition to(ColorPrimaries primaries) const;

	ColorspaceDefinition to_rgb() const;
	ColorspaceDefinition to_linear() const;
};

// Compare colorspaces by comparing each component.
bool operator==(const ColorspaceDefinition &a, const ColorspaceDefinition &b);
bool operator!=(const ColorspaceDefinition &a, const ColorspaceDefinition &b);


struct ColorspaceConversion {
	unsigned width;
	unsigned height;

#include "common/builder.h"
	BUILDER_MEMBER(ColorspaceDefinition, csp_in)
	BUILDER_MEMBER(ColorspaceDefinition, csp_out)
	BUILDER_MEMBER(double, peak_luminance)
	BUILDER_MEMBER(bool, approximate_gamma)
	BUILDER_MEMBER(CPUClass, cpu)
#undef BUILDER_MEMBER

	ColorspaceConversion(unsigned width, unsigned height);

	std::unique_ptr<graph::ImageFilter> create() const;
};

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_COLORSPACE2_H_
