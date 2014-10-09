#pragma once

#ifndef ZIMG_MATRIX_H_
#define ZIMG_MATRIX_H_

#include <vector>

namespace zimg {;
namespace unresize {;

/**
 * Row compressed matrix. Stored as an array of arrays.
 */
class RowMatrix {
	std::vector<std::vector<float>> m_storage;
	std::vector<int> m_offsets;
	int m_rows;
	int m_cols;

	class Element;

	/**
	* Proxy to a matrix row.
	*/
	class Row {
		RowMatrix &m_parent;
		int m_row;

		Row(RowMatrix &parent, int row);
	public:
		Element operator[](int j);

		friend class RowMatrix;
	};

	/**
	* Read-only proxy to a matrix row.
	*/
	class ConstRow {
		const RowMatrix &m_parent;
		int m_row;

		ConstRow(const RowMatrix &parent, int row);
	public:
		float operator[](int j);

		friend class RowMatrix;
	};

	/**
	 * Proxy to an individual element.
	 */
	class Element {
		RowMatrix &m_parent;
		int m_row;
		int m_col;

		Element(RowMatrix &parent, int row, int col);
	public:
		Element &operator=(float f);

		operator float();

		friend class Row;
	};

	void check_bounds(int i, int j) const;
public:
	/**
	 * Initialize a matrix of dimension 0.
	 */
	RowMatrix() = default;

	/**
	 * Initialize an empty matrix of dimension m-by-n.
	 *
	 * @param m number of rows
	 * @param n number of columns
	 */
	RowMatrix(int m, int n);

	/**
	 * @return the number of rows
	 */
	int rows() const;

	/**
	 * @return the number of columns
	 */
	int cols() const;

	/**
	 * @param i row index
	 * @return a proxy to the i-th row
	 */
	Row operator[](int i);

	/**
	 * @see RowMatrix::operator[](int)
	 */
	ConstRow operator[](int i) const;

	/**
	 * @param i row index
	 * @return the index of the left-most non-zero element in the i-th row.
	 */
	int row_left(int i) const;

	/**
	 * @param i row index
	 * @return the index plus one of the right-most non-zero element in the i-th row.
	 */
	int row_right(int i) const;

	/**
	 * @param i row index
	 * @param j column index
	 * @return a reference to the element at (i, j), resizing if needed
	 */
	float &get_element_ref(int i, int j);

	/**
	 * @param i row index
	 * @param j column index
	 * @return the value of the element at (i, j)
	 */
	float get_element_val(int i, int j) const;

	/**
	 * Shrink the matrix by removing leading and trailing zeros from each row.
	 */
	void compress();
};

/**
 * Storage of LU decomposition.
 *
 * See: unresize.h for LU conventions
 */
class TridiagonalLU {
	std::vector<float> m_diag_l;
	std::vector<float> m_diag_u;
	std::vector<float> m_diag_c;

	int m_dim;
public:
	/**
	 * Initialize a LU of dimension 0.
	 */
	TridiagonalLU() = default;

	/**
	 * Initialize a LU of dimension dim.
	 *
	 * @param dim dimension of LU
	 */
	TridiagonalLU(int dim);

	/**
	 * @return dimension of LU
	 */
	int dim() const;

	/**
	 * @param i diagonal index
	 * @return the i-th element from the L diagonal
	 */
	float &l(int i);

	/**
	 * @see TridiagonalLU::l(int)
	 */
	float l(int i) const;

	/**
	 * @param i diagonal index
	 * @return the i-th element from the U diagonal
	 */
	float &u(int i);

	/**
	 * @see TridiagonalLU::u(int)
	 */
	float u(int i) const;

	/**
	 * @param i diagonal index
	 * @return the i-th element from the C diagonal
	 */
	float &c(int i);

	/**
	 * @see TridiagonalLU::c(int)
	 */
	float c(int i) const;
};

/**
 * Compute the product of two matrices.
 *
 * @param lhs left hand matrix
 * @param rhs right hand matrix
 * @return the matrix product
 */
RowMatrix matrix_matrix_product(const RowMatrix &lhs, const RowMatrix &rhs);

/**
 * Compute the transpose of a matrix.
 *
 * @param r matrix
 * @return the transposed matrix
 */
RowMatrix transpose(const RowMatrix &r);

/**
 * Compute the LU decomposition of a tridiagonal matrix.
 *
 * @param r tridiagonal matrix
 * @return the LU decomposition
 */
TridiagonalLU tridiagonal_decompose(const RowMatrix &r);

} // namespace unresize
} // namespace zimg

#endif // ZIMG_MATRIX_H_
