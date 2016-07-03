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
	MATRIX_UNSPECIFIED,
	MATRIX_RGB,
	MATRIX_601,
	MATRIX_709,
	MATRIX_YCGCO,
	MATRIX_2020_NCL,
	MATRIX_2020_CL
};

enum class TransferCharacteristics {
	TRANSFER_UNSPECIFIED,
	TRANSFER_LINEAR,
	TRANSFER_709
};

enum class ColorPrimaries {
	PRIMARIES_UNSPECIFIED,
	PRIMARIES_SMPTE_C,
	PRIMARIES_709,
	PRIMARIES_2020
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
	BUILDER_MEMBER(CPUClass, cpu)
#undef BUILDER_MEMBER

	ColorspaceConversion(unsigned width, unsigned height);

	std::unique_ptr<graph::ImageFilter> create() const;
};

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_COLORSPACE2_H_
