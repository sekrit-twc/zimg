#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_
#define ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_

namespace zimg {;

enum class CPUClass;

namespace colorspace {;

class PixelAdapter;
class Operation;

struct Matrix3x3;

PixelAdapter *create_pixel_adapter_avx2();

Operation *create_matrix_operation_sse2(const Matrix3x3 &m);
Operation *create_matrix_operation_avx2(const Matrix3x3 &m);

Operation *create_rec709_gamma_operation_avx2();
Operation *create_rec709_inverse_gamma_operation_avx2();

/**
 * Create an appropriate x86 optimized PixelAdapter for the given CPU.
 *
 * @param cpu create operation for given cpu
 * @return concrete operation
 */
PixelAdapter *create_pixel_adapter_x86(CPUClass cpu);

/**
 * Create an appropriate x86 optimized matrix operation for the given CPU.
 *
 * @param m matrix
 * @param cpu create operation for given cpu
 */
Operation *create_matrix_operation_x86(const Matrix3x3 &m, CPUClass cpu);

/**
 * Create an appropriate x86 optimized Rec.709 operation.
 *
 * @param cpu create operation for given cpu
 */
Operation *create_rec709_gamma_operation_x86(CPUClass cpu);

/**
 * Create an appropriate x86 optimized inverse Rec.709 operation.
 *
 * @param cpu create operation for given cpu
 */
Operation *create_rec709_inverse_gamma_operation_x86(CPUClass cpu);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_

#endif // ZIMG_X86
