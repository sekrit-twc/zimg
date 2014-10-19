#pragma once

#ifndef ZIMG_COLORSPACE_OPERATION_H_
#define ZIMG_COLORSPACE_OPERATION_H_

#include <cstdint>

namespace zimg {;

enum class CPUClass;

namespace colorspace {;

enum class MatrixCoefficients;
enum class TransferCharacteristics;
enum class ColorPrimaries;

/**
 * Base class for implementations of pixel format conversion.
 */
class PixelAdapater {
public:
	/**
	 * Destroy implementation.
	 */
	virtual ~PixelAdapater() = 0;

	/**
	 * Convert from half precision to full precision.
	 *
	 * @param src input samples
	 * @param dst output samples
	 * @param width number of samples
	 */
	virtual void f16_to_f32(const uint16_t *src, float *dst, int width) const = 0;

	/**
	 * Convert from single precision to half precision.
	 *
	 * @see PixelAdapter::f16_to_f32
	 */
	virtual void f16_from_f32(const float *src, uint16_t *dst, int width) const = 0;
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
	 * Apply operation to pixels in-place, overwriting the input.
	 *
	 * @param ptr pointer to three image scanline pointers
	 * @param width number of samples
	 */
	virtual void process(float * const *ptr, int width) const = 0;
};

/**
 * Create a concrete pixel adapter.
 *
 * @param cpu create adapter optimized for given cpu
 */
PixelAdapater *create_pixel_adapter(CPUClass cpu);

/**
 * Create an operation converting from YUV to RGB via a 3x3 matrix.
 *
 * @param matrix matrix coefficients
 * @param cpu create operation optimized for given cpu
 * @return concrete operation
 * @throws ZimgIllegalArgument on unsupported matrix
 */
Operation *create_ncl_yuv_to_rgb_operation(MatrixCoefficients matrix, CPUClass cpu);

/**
 * Create an operation converting from RGB to YUV via a 3x3 matrix.
 *
 * @see create_ncl_yuv_to_rgb_operation
 */
Operation *create_ncl_rgb_to_yuv_operation(MatrixCoefficients matrix, CPUClass cpu);

/**
 * Create an operation inverting an optical transfer function.
 *
 * @param transfer transfer characteristics
 * @param cpu create operation optimized for given cpu
 * @return concrete operation
 * @throws ZimgIllegalArgument on unsupported transfer
 */
Operation *create_gamma_to_linear_operation(TransferCharacteristics transfer, CPUClass cpu);

/**
 * Create anm operation applying an optical transfer function.
 *
 * @see create_gamma_to_linear_operation
 */
Operation *create_linear_to_gamma_operation(TransferCharacteristics transfer, CPUClass cpu);

/**
 * Create an operation converting from YUV to RGB via Rec.2020 Constant Luminance method.
 *
 * @param cpu create operation optimized for given cpu
 */
Operation *create_2020_cl_yuv_to_rgb_operation(CPUClass cpu);

/**
 * Create an operation converting from RGB to YUV via Rec.2020 Constant Luinance method.
 *
 * @see create_2020_cl_yuv_to_rgb_operation
 */
Operation *create_2020_cl_rgb_to_yuv_operation(CPUClass cpu);

/**
 * Create an operation converting between color primaries.
 *
 * @param primaries_in input primaries
 * @param primaries_out output primaries
 * @param cpu create operation optimized for given cpu
 * @throws ZimgIlegalArgument on unsupported primaries
 */
Operation *create_gamut_operation(ColorPrimaries primaries_in, ColorPrimaries primaries_out, CPUClass cpu);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_OPERATION_H_
