#pragma once

#ifndef ZIMG_COLORSPACE_OPERATION_IMPL_H_
#define ZIMG_COLORSPACE_OPERATION_IMPL_H_

#include "common/libm_wrapper.h"
#include "operation.h"

namespace zimg {

enum class CPUClass;

namespace colorspace {

struct Matrix3x3;
class Operation;

const float REC709_ALPHA = 1.09929682680944f;
const float REC709_BETA = 0.018053968510807f;

const float SRGB_ALPHA = 1.055f;
const float SRGB_BETA = 0.0031308f;

const float ST2084_M1 = 0.1593017578125f;
const float ST2084_M2 = 78.84375f;
const float ST2084_C1 = 0.8359375f;
const float ST2084_C2 = 18.8515625f;
const float ST2084_C3 = 18.6875f;
const float ST2084_PEAK_LUMINANCE = 10000.0f; // Units of cd/m^2.

const float ARIB_B67_A = 0.17883277f;
const float ARIB_B67_B = 0.28466892f;
const float ARIB_B67_C = 0.55991073f;

float rec_709_gamma(float x) noexcept;
float rec_709_inverse_gamma(float x) noexcept;

float srgb_gamma(float x) noexcept;
float srgb_inverse_gamma(float x) noexcept;

float st_2084_gamma(float x) noexcept;
float st_2084_inverse_gamma(float x) noexcept;

float arib_b67_gamma(float x) noexcept;
float arib_b67_inverse_gamma(float x) noexcept;

/**
 * Base class for matrix operation implementations.
 */
class MatrixOperationImpl : public Operation {
protected:
	/**
	 * Transformation matrix.
	 */
	float m_matrix[3][3];

	/**
	 * Initialize the implementation with the given matrix.
	 *
	 * @param m transformation matrix
	 */
	MatrixOperationImpl(const Matrix3x3 &matrix);
};

/**
 * Create operation consisting of applying a 3x3 matrix to each pixel triplet.
 *
 * @param m matrix
 * @param cpu create operation optimized for given cpu
 * @return concrete operation
 */
std::unique_ptr<Operation> create_matrix_operation(const Matrix3x3 &m, CPUClass cpu);

/**
 * Create operation consisting of applying Rec.709 transfer function.
 *
 * @param cpu create operation optimized for given cpu
 * @return concrete operation
 */
std::unique_ptr<Operation> create_rec709_gamma_operation(CPUClass cpu);

/**
 * Create operation consisting of inverting Rec.709 transfer function.
 *
 * @see create_rec709_gamma_operation
 */
std::unique_ptr<Operation> create_rec709_inverse_gamma_operation(CPUClass cpu);

/**
 * Create operation consisting of applying sRGB transfer function.
 *
 * @param cpu create operation optimized for given cpu
 * @return concrete operation
 */
std::unique_ptr<Operation> create_srgb_gamma_operation(CPUClass cpu);

/**
 * Create operation consisting of inverting sRGB transfer function.
 *
 * @see create_rec709_gamma_operation
 */
std::unique_ptr<Operation> create_srgb_inverse_gamma_operation(CPUClass cpu);


/**
 * Create operation consisting of applying SMPTE ST 2084 (PQ) transfer function.
 *
 * @param peak_luminance physical brightness (cd/m^2) corresponding to linear value 1.0
 * @param cpu create operation optimized for given cpu
 * @return concrete operation
 */
std::unique_ptr<Operation> create_st2084_gamma_operation(double peak_luminance, CPUClass cpu);

/**
 * Create operation consisting of inverting SMPTE ST 2084 (PQ) transfer function.
 *
 * @see create_st2084_gamma_operation
 */
std::unique_ptr<Operation> create_st2084_inverse_gamma_operation(double peak_luminance, CPUClass cpu);

/**
 * Create operation consisting of applying ARIB STD-B67 (HLG) transfer function.
 *
 * @see create_rec709_gamma_operation
 */
std::unique_ptr<Operation> create_b67_gamma_operation(CPUClass cpu);

/**
 * Create operation consisting of inverting ARIB STD-B67 (HLG) transfer function.
 *
 * @see create_rec709_gamma_operation
 */
std::unique_ptr<Operation> create_b67_inverse_gamma_operation(CPUClass cpu);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_OPERATION_IMPL_H_
