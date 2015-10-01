#pragma once

#ifndef ZIMG_COLORSPACE_MATRIX3_H_
#define ZIMG_COLORSPACE_MATRIX3_H_

#include <array>

namespace zimg {;
namespace colorspace {;

/**
 * Fixed size vector of 3 numbers.
 */
struct Vector3 : public std::array<double, 3> {
	Vector3() = default;

	Vector3(double a, double b, double c);
};

/**
 * Fixed size 3x3 matrix.
 */
struct Matrix3x3 : public std::array<Vector3, 3> {
	Matrix3x3() = default;

	Matrix3x3(const Vector3 &a, const Vector3 &b, const Vector3 &c);
};

/**
 * Element-wise multiplication between vectors.
 *
 * @param v1 lhs
 * @param v2 rhs
 * @return element-wise product
 */
Vector3 operator*(const Vector3 &v1, const Vector3 &v2);

/**
 * Matrix-vector multiplication.
 *
 * @param m matrix
 * @param v vector
 * @return product
 */
Vector3 operator*(const Matrix3x3 &m, const Vector3 &v);

/**
 * Matrix-matrix multiplication.
 *
 * @param a lhs
 * @param b rhs
 * @return product
 */
Matrix3x3 operator*(const Matrix3x3 &a, const Matrix3x3 &b);

/**
 * Determinant of matrix.
 *
 * @param m matrix
 * @return determinant
 */
double determinant(const Matrix3x3 &m);

/**
 * Inverse of matrix.
 *
 * @param m matrix
 * @return inverse
 */
Matrix3x3 inverse(const Matrix3x3 &m);

/**
 * Transpose of matrix.
 *
 * @param m matrix
 * @return transpose
 */
Matrix3x3 transpose(const Matrix3x3 &m);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_MATRIX3_H_
