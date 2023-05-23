#pragma once

#ifndef ZIMG_COLORSPACE_OPERATION_H_
#define ZIMG_COLORSPACE_OPERATION_H_

#include <cmath>
#include <memory>

namespace zimg {
enum class CPUClass;
}

namespace zimg::colorspace {

struct ColorspaceDefinition;

enum class MatrixCoefficients;
enum class TransferCharacteristics;
enum class ColorPrimaries;

/**
 * Parameters struct for operation factory functions.
 */
struct OperationParams {
#include "common/builder.h"
	BUILDER_MEMBER(double, peak_luminance)
	BUILDER_MEMBER(bool, approximate_gamma)
	BUILDER_MEMBER(bool, scene_referred)
#undef BUILDER_MEMBER

	/**
	 * Default construct OperationParams, initializing it with invalid values.
	 */
	OperationParams() :
		peak_luminance{ NAN },
		approximate_gamma{},
		scene_referred{}
	{}
};

/**
 * Base class for colorspace conversion operations.
 */
class Operation {
public:
	/**
	 * Destroy operation.
	 */
	virtual ~Operation() = 0;

	/**
	 * Apply operation to pixels.
	 *
	 * @param src pointer to pointer to input channels
	 * @param dst pointer to pointer to output channels
	 * @param left left column index
	 * @param right right column index
	 */
	virtual void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const = 0;
};

/**
 * Create an operation converting from YUV to RGB via a 3x3 matrix.
 *
 * @param in input colorspace
 * @param out output colorspace
 * @param params parameters
 * @param cpu create operation optimized for given cpu
 * @return concrete operation
 */
std::unique_ptr<Operation> create_ncl_yuv_to_rgb_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu);

/**
 * Create an operation converting from RGB to YUV via a 3x3 matrix.
 *
 * @see create_ncl_yuv_to_rgb_operation
 */
std::unique_ptr<Operation> create_ncl_rgb_to_yuv_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu);

/**
* Create an operation converting from ICtCp to LMS via a 3x3 matrix.
*
* @see create_ncl_yuv_to_rgb_operation
*/
std::unique_ptr<Operation> create_ictcp_to_lms_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu);

/**
* Create an operation converting from LMS to ICtCp via a 3x3 matrix.
*
* @see create_ncl_yuv_to_rgb_operation
*/
std::unique_ptr<Operation> create_lms_to_ictcp_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu);

/**
 * Create an operation inverting an optical transfer function.
 *
 * @see create_ncl_yuv_to_rgb_operation
 */
std::unique_ptr<Operation> create_gamma_to_linear_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu);

/**
 * Create an operation applying an optical transfer function.
 *
 * @see create_ncl_yuv_to_rgb_operation
 */
std::unique_ptr<Operation> create_linear_to_gamma_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu);

/**
 * Create an operation converting from YUV to RGB via Rec.2020 Constant Luminance method.
 *
 * @see create_ncl_yuv_to_rgb_operation
 */
std::unique_ptr<Operation> create_cl_yuv_to_rgb_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu);

/**
 * Create an operation converting from RGB to YUV via Rec.2020 Constant Luminance method.
 *
 * @see create_ncl_yuv_to_rgb_operation
 */
std::unique_ptr<Operation> create_cl_rgb_to_yuv_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu);

/**
 * Create an operation converting between color primaries.
 *
 * @see create_ncl_yuv_to_rgb_operation
 */
std::unique_ptr<Operation> create_gamut_operation(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu);

} // namespace zimg::colorspace

#endif // ZIMG_COLORSPACE_OPERATION_H_
