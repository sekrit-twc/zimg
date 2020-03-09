#pragma once

#ifdef ZIMG_ARM

#ifndef ZIMG_COLORSPACE_ARM_OPERATION_IMPL_ARM_H_
#define ZIMG_COLORSPACE_ARM_OPERATION_IMPL_ARM_H_

#include <memory>

namespace zimg {

enum class CPUClass;

namespace colorspace {

struct Matrix3x3;
struct OperationParams;
struct TransferFunction;
class Operation;

std::unique_ptr<Operation> create_matrix_operation_neon(const Matrix3x3 &m);

std::unique_ptr<Operation> create_matrix_operation_arm(const Matrix3x3 &m, CPUClass cpu);

std::unique_ptr<Operation> create_gamma_operation_neon(const TransferFunction &transfer, const OperationParams &params);

std::unique_ptr<Operation> create_gamma_operation_arm(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu);

std::unique_ptr<Operation> create_inverse_gamma_operation_neon(const TransferFunction &transfer, const OperationParams &params);

std::unique_ptr<Operation> create_inverse_gamma_operation_arm(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_ARM_OPERATION_IMPL_ARM_H_

#endif // ZIMG_ARM
