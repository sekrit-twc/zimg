#pragma once

#ifndef ZIMG_MATRIX_H_
#define ZIMG_MATRIX_H_

#include <vector>
#include "except.h"

namespace zimg {;

/**
 * Row compressed sparse matrix. Stored as an array of arrays.
 * The first array holds pointers to each row's data, which stores a single range of non-zero entries.
 */
template <class T>
class RowMatrix {
	std::vector<std::vector<T>> m_storage;
	std::vector<size_t> m_offsets;
	size_t m_rows;
	size_t m_cols;

	class ElementProxy {
		RowMatrix *matrix;
		size_t i;
		size_t j;
	public:
		ElementProxy(RowMatrix *matrix, size_t i, size_t j) : matrix{ matrix }, i{ i }, j{ j }
		{}

		T operator=(const ElementProxy &val)
		{
			return operator=(static_cast<T>(val));
		}

		T operator=(T val)
		{
			if (val != matrix->element_val(i, j))
				matrix->element_ref(i, j) = val;

			return val;
		}

		T operator+=(T val)
		{
			return operator=(operator T() + val);
		}

		operator T() const
		{
			return matrix->element_val(i, j);
		}
	};

	class RowConstProxy {
		const RowMatrix *matrix;
		size_t i;
	public:
		RowConstProxy(const RowMatrix *matrix, size_t i) : matrix{ matrix }, i{ i }
		{}

		T operator[](size_t j) const
		{
			return matrix->element_val(i, j);
		}
	};

	class RowProxy {
		RowMatrix *matrix;
		size_t i;
	public:
		RowProxy(RowMatrix *matrix, size_t i) : matrix{ matrix }, i{ i }
		{}

		ElementProxy operator[](size_t j)
		{
			return{ matrix, i, j };
		}
	};

	void check_bounds(size_t i, size_t j) const
	{
		if (i >= m_rows)
			throw ZimgLogicError{ "row index out of bounds" };
		if (j >= m_cols)
			throw ZimgLogicError{ "column index out of bounds" };
	}

	T &element_ref(size_t i, size_t j)
	{
		check_bounds(i, j);

		auto &row_data = m_storage[i];
		size_t left = row_left(i);
		size_t right = row_right(i);

		// Resize row if needed.
		if (row_data.empty()) {
			row_data.resize(1);
			left = j;
		} else if (j < left) {
			// Zero-extend the row on the left.
			row_data.insert(row_data.begin(), left - j, static_cast<T>(0));
			left = j;
		} else if (j >= right) {
			// Zero-extend the row on the right.
			row_data.insert(row_data.end(), j - right + 1, static_cast<T>(0));
		}

		// Update offset array.
		m_offsets[i] = left;

		return row_data[j - left];
	}

	T element_val(size_t i, size_t j) const
	{
		check_bounds(i, j);

		size_t left = row_left(i);
		size_t right = row_right(i);

		if (j < left || j >= right)
			return static_cast<T>(0);
		else
			return m_storage[i][j - left];
	}
public:
	RowMatrix() = default;

	RowMatrix(size_t m, size_t n) : m_storage(m), m_offsets(m), m_rows{ m }, m_cols{ n }
	{}

	size_t rows() const
	{
		return m_rows;
	}

	size_t cols() const
	{
		return m_cols;
	}

	RowProxy operator[](size_t i)
	{
		return{ this, i };
	}

	RowConstProxy operator[](size_t i) const
	{
		return{ this, i };
	}

	size_t row_left(size_t i) const
	{
		check_bounds(i, 0);
		return m_offsets[i];
	}

	size_t row_right(size_t i) const
	{
		check_bounds(i, 0);
		return m_offsets[i] + m_storage[i].size();
	}

	void compress()
	{
		for (size_t i = 0; i < m_rows; ++i) {
			auto &row_data = m_storage[i];
			size_t left;
			size_t right;

			for (left = 0; left < row_data.size(); ++left) {
				if (row_data[left] != static_cast<T>(0))
					break;
			}
			for (right = row_data.size() - 1; right > left; --right) {
				if (row_data[right] != static_cast<T>(0))
					break;
			}

			// Shrink row if non-empty, else free row.
			if (right - left) {
				row_data.erase(row_data.begin() + right + 1, row_data.end());
				row_data.erase(row_data.begin(), row_data.begin() + left);
				m_offsets[i] += left;
			} else {
				row_data.clear();
				m_offsets[i] = 0;
			}
		}
	}
};

template <class T>
inline RowMatrix<T> operator*(const RowMatrix<T> &lhs, const RowMatrix<T> &rhs)
{
	RowMatrix<T> m{ lhs.rows(), rhs.cols() };

	for (size_t i = 0; i < lhs.rows(); ++i) {
		for (size_t j = 0; j < rhs.cols(); ++j) {
			T accum = 0;

			for (size_t k = lhs.row_left(i); k < lhs.row_right(i); ++k) {
				accum += lhs[i][k] * rhs[k][j];
			}
			m[i][j] = accum;
		}
	}

	m.compress();
	return m;
}

template <class T>
inline RowMatrix<T> transpose(const RowMatrix<T> &r)
{
	RowMatrix<T> m{ r.cols(), r.rows() };

	for (size_t i = 0; i < r.rows(); ++i) {
		for (size_t j = 0; j < r.cols(); ++j) {
			m[j][i] = r[i][j];
		}
	}

	m.compress();
	return m;
}

} // namespace zimg

#endif // ZIMG_MATRIX_H_
