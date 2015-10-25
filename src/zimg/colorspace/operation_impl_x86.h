#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_
#define ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_

#include <memory>

namespace zimg {;

enum class CPUClass;

namespace colorspace {;

class Operation;
struct Matrix3x3;

std::unique_ptr<Operation> create_matrix_operation_sse(const Matrix3x3 &m);

std::unique_ptr<Operation> create_matrix_operation_avx(const Matrix3x3 &m);

std::unique_ptr<Operation> create_matrix_operation_x86(const Matrix3x3 &m, CPUClass cpu);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_OPERATION_IMPL_X86_H_

#endif // ZIMG_X86
