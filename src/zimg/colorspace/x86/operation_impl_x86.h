#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_COLORSPACE_X86_OPERATION_IMPL_X86_H_
#define ZIMG_COLORSPACE_X86_OPERATION_IMPL_X86_H_

#include <memory>

namespace zimg {
enum class CPUClass;
}

namespace zimg::colorspace {

struct Matrix3x3;
struct OperationParams;
struct TransferFunction;
class Operation;

std::unique_ptr<Operation> create_matrix_operation_avx2(const Matrix3x3 &m);
std::unique_ptr<Operation> create_matrix_operation_avx512(const Matrix3x3 &m);

std::unique_ptr<Operation> create_matrix_operation_x86(const Matrix3x3 &m, CPUClass cpu);

std::unique_ptr<Operation> create_gamma_operation_sse2(const TransferFunction &transfer, const OperationParams &params);
std::unique_ptr<Operation> create_gamma_operation_avx2(const TransferFunction &transfer, const OperationParams &params);
std::unique_ptr<Operation> create_gamma_operation_avx512(const TransferFunction &transfer, const OperationParams &params);

std::unique_ptr<Operation> create_gamma_operation_x86(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu);

std::unique_ptr<Operation> create_inverse_gamma_operation_sse2(const TransferFunction &transfer, const OperationParams &params);
std::unique_ptr<Operation> create_inverse_gamma_operation_avx2(const TransferFunction &transfer, const OperationParams &params);
std::unique_ptr<Operation> create_inverse_gamma_operation_avx512(const TransferFunction &transfer, const OperationParams &params);

std::unique_ptr<Operation> create_inverse_gamma_operation_x86(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu);

} // namespace zimg::colorspace

#endif // ZIMG_COLORSPACE_X86_OPERATION_IMPL_X86_H_

#endif // ZIMG_X86
