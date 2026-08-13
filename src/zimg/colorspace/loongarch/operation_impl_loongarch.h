#pragma once

#ifdef ZIMG_LOONGARCH

#ifndef ZIMG_COLORSPACE_LOONGARCH_OPERATION_IMPL_LOONGARCH_H_
#define ZIMG_COLORSPACE_LOONGARCH_OPERATION_IMPL_LOONGARCH_H_

#include <memory>

namespace zimg {
enum class CPUClass;
}

namespace zimg::colorspace {

struct Matrix3x3;
struct OperationParams;
struct TransferFunction;
class Operation;

std::unique_ptr<Operation> create_matrix_operation_lsx(const Matrix3x3 &m);

std::unique_ptr<Operation> create_matrix_operation_loongarch(const Matrix3x3 &m, CPUClass cpu);

std::unique_ptr<Operation> create_gamma_operation_lsx(const TransferFunction &transfer, const OperationParams &params);

std::unique_ptr<Operation> create_gamma_operation_loongarch(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu);

std::unique_ptr<Operation> create_inverse_gamma_operation_lsx(const TransferFunction &transfer, const OperationParams &params);

std::unique_ptr<Operation> create_inverse_gamma_operation_loongarch(const TransferFunction &transfer, const OperationParams &params, CPUClass cpu);

} // namespace zimg::colorspace

#endif // ZIMG_COLORSPACE_LOONGARCH_OPERATION_IMPL_LOONGARCH_H_

#endif // ZIMG_LOONGARCH
