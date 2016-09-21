#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_
#define ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_

#include <memory>

namespace zimg {

enum class CPUClass;

namespace colorspace {

enum class TransferCharacteristics;
struct Matrix3x3;
struct OperationParams;
class Operation;

std::unique_ptr<Operation> create_matrix_operation_sse(const Matrix3x3 &m);

std::unique_ptr<Operation> create_matrix_operation_avx(const Matrix3x3 &m);

std::unique_ptr<Operation> create_matrix_operation_x86(const Matrix3x3 &m, CPUClass cpu);

std::unique_ptr<Operation> create_gamma_to_linear_operation_sse2(TransferCharacteristics transfer, const OperationParams &params);
std::unique_ptr<Operation> create_gamma_to_linear_operation_avx2(TransferCharacteristics transfer, const OperationParams &params);

std::unique_ptr<Operation> create_gamma_to_linear_operation_x86(TransferCharacteristics transfer, const OperationParams &params, CPUClass cpu);

std::unique_ptr<Operation> create_linear_to_gamma_operation_sse2(TransferCharacteristics transfer, const OperationParams &params);
std::unique_ptr<Operation> create_linear_to_gamma_operation_avx2(TransferCharacteristics transfer, const OperationParams &params);

std::unique_ptr<Operation> create_linear_to_gamma_operation_x86(TransferCharacteristics transfer, const OperationParams &params, CPUClass cpu);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_

#endif // ZIMG_X86
