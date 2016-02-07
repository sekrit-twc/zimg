#pragma once

#ifndef ZIMG_MATRIX_H_
#define ZIMG_MATRIX_H_

#include <vector>

namespace zimg {

/**
 * Row compressed sparse matrix. Stored as an array of arrays.
 *
 * The first array holds pointers to each row's data, which stores a single
 * range of non-zero entries.
 *
 * @tparam T stored numeric type, must be floating point
 */
template <class T>
class RowMatrix {
public:
	typedef size_t size_type;
private:
	struct non_copyable {
		non_copyable() = default;

		non_copyable(const non_copyable &) = delete;

		non_copyable &operator=(const non_copyable &) = delete;
	};

	class row_proxy;

	class proxy : private non_copyable {
		RowMatrix *matrix;
		size_type i;
		size_type j;

		proxy(RowMatrix *matrix, size_type i, size_type j);
	public:
		const proxy &operator=(const T &val) const;

		const proxy &operator+=(const T &val) const;

		const proxy &operator-=(const T &val) const;

		const proxy &operator*=(const T &val) const;

		const proxy &operator/=(const T &val) const;

		operator T() const;

		friend class RowMatrix::row_proxy;
	};

	class row_proxy : private non_copyable {
		RowMatrix *matrix;
		size_type i;

		row_proxy(RowMatrix *matrix, size_type i);
	public:
		proxy operator[](size_type j) const;

		friend class RowMatrix;
	};

	class row_const_proxy : private non_copyable {
		const RowMatrix *matrix;
		size_type i;

		row_const_proxy(const RowMatrix *matrix, size_type i);
	public:
		T operator[](size_type j) const;

		friend class RowMatrix;
	};

	std::vector<std::vector<T>> m_storage;
	std::vector<size_type> m_offsets;
	size_type m_rows;
	size_type m_cols;

	void check_bounds(size_type i, size_type j) const;

	T val(size_type i, size_type j) const;

	T &ref(size_type i, size_type j);
public:
	/**
	 * Default construct RowMatrix, creating a zero dimension matrix.
	 */
	RowMatrix();

	/**
	 * Construct a RowMatrix of a given size, creating an empty matrix.
	 *
	 * @param m number of rows
	 * @param n number of columns
	 */
	RowMatrix(size_type m, size_type n);

	/**
	 * Get the number of rows.
	 *
	 * @return rows
	 */
	size_type rows() const;

	/**
	 * Get the number of columns.
	 *
	 * @return columns
	 */
	size_type cols() const;

	/**
	 * Get the left-most non-sparse column in a given row.
	 *
	 * @param i row index
	 * @return column index
	 */
	size_type row_left(size_type i) const;

	/**
	 * Get the right-most non-sparse column in a given row, plus one.
	 *
	 * @param i row index
	 * @return column index
	 */
	size_type row_right(size_type i) const;

	/**
	 * Access a row of the matrix.
	 *
	 * Yields an object supporting the indexing operator, such that
	 *   m[i][j]
	 * creates an lvalue pointing at the i-th row and j-th column of m.
	 *
	 * @param i row index
	 * @return proxy to matrix row
	 */
	row_proxy operator[](size_type i);

	/**
	 * Read-only access to a matrix row. Creates an rvalue.
	 *
	 * @see operator[](size_type)
	 */
	row_const_proxy operator[](size_type i) const;

	/**
	 * Remove sparse entries from the internal storage.
	 *
	 * After compression, the results of {@link row_left} and {@link row_right}
	 * point to non-zero entries in each row.
	 */
	void compress();
};

template <class T>
RowMatrix<T> operator~(const RowMatrix<T> &r);

template <class T>
RowMatrix<T> operator*(const RowMatrix<T> &lhs, const RowMatrix<T> &rhs);

extern template class RowMatrix<float>;
extern template class RowMatrix<double>;
extern template class RowMatrix<long double>;

extern template RowMatrix<float> operator~(const RowMatrix<float> &r);
extern template RowMatrix<double> operator~(const RowMatrix<double> &r);
extern template RowMatrix<long double> operator~(const RowMatrix<long double> &r);

extern template RowMatrix<float> operator*(const RowMatrix<float> &lhs, const RowMatrix<float> &rhs);
extern template RowMatrix<double> operator*(const RowMatrix<double> &lhs, const RowMatrix<double> &rhs);
extern template RowMatrix<long double> operator*(const RowMatrix<long double> &lhs, const RowMatrix<long double> &rhs);

} // namespace zimg

#endif // ZIMG_MATRIX_H_
