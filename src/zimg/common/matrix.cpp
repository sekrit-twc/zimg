#include <algorithm>
#include "matrix.h"
#include "zassert.h"

namespace zimg {

namespace {

template <class T>
struct Equals {
	T value;

	template <class U>
	bool operator()(const U &other) const
	{
		return value == other;
	}
};

} // namespace


template <class T>
RowMatrix<T>::proxy::proxy(RowMatrix *matrix, size_type i, size_type j) :
	matrix{ matrix },
	i{ i },
	j{ j }
{
}

template <class T>
auto RowMatrix<T>::proxy::operator=(const T &val) const -> const proxy &
{
	if (matrix->val(i, j) != val)
		matrix->ref(i, j) = val;

	return *this;
}

template <class T>
auto RowMatrix<T>::proxy::operator+=(const T &val) const -> const proxy &
{
	return operator=(static_cast<T>(*this) + val);
}

template <class T>
auto RowMatrix<T>::proxy::operator-=(const T &val) const -> const proxy &
{
	return operator=(static_cast<T>(*this) - val);
}

template <class T>
auto RowMatrix<T>::proxy::operator*=(const T &val) const -> const proxy &
{
	return operator=(static_cast<T>(*this) * val);
}

template <class T>
auto RowMatrix<T>::proxy::operator/=(const T &val) const -> const proxy &
{
	return operator=(static_cast<T>(*this) / val);
}

template <class T>
RowMatrix<T>::proxy::operator T() const
{
	return matrix->val(i, j);
}

template <class T>
RowMatrix<T>::row_proxy::row_proxy(RowMatrix *matrix, size_type i) :
	matrix{ matrix },
	i{ i }
{
}

template <class T>
auto RowMatrix<T>::row_proxy::operator[](size_type j) const -> proxy
{
	return{ matrix, i, j };
}

template <class T>
RowMatrix<T>::row_const_proxy::row_const_proxy(const RowMatrix *matrix, size_type i) :
	matrix{ matrix },
	i{ i }
{
}

template <class T>
T RowMatrix<T>::row_const_proxy::operator[](size_type j) const
{
	return matrix->val(i, j);
}

template <class T>
RowMatrix<T>::RowMatrix() :
	m_rows{},
	m_cols{}
{
}

template <class T>
RowMatrix<T>::RowMatrix(size_type m, size_type n) :
	m_storage(m),
	m_offsets(m),
	m_rows{ m },
	m_cols{ n }
{
}

template <class T>
void RowMatrix<T>::check_bounds(size_type i, size_type j) const
{
	zassert_d(i < m_rows, "row index out of bounds");
	zassert_d(j < m_cols, "column index out of bounds");
}

template <class T>
T RowMatrix<T>::val(size_type i, size_type j) const
{
	check_bounds(i, j);

	size_type left = row_left(i);
	size_type right = row_right(i);

	return (j < left || j >= right) ? static_cast<T>(0) : m_storage[i][j - left];
}

template <class T>
T &RowMatrix<T>::ref(size_type i, size_type j)
{
	check_bounds(i, j);

	auto &row_data = m_storage[i];
	size_type left = row_left(i);
	size_type right = row_right(i);

	if (row_data.empty()) {
		// Initialize row if empty.
		row_data.resize(1, static_cast<T>(0));
		left = j;
	} else if (j < left) {
		// Zero-extend row on the left.
		row_data.insert(row_data.begin(), left - j, static_cast<T>(0));
		left = j;
	} else if (j >= right) {
		// Zero-extend row on the right.
		row_data.insert(row_data.end(), j - right + 1, static_cast<T>(0));
	}

	// Update offset array.
	m_offsets[i] = left;

	return row_data[j - left];
}

template <class T>
auto RowMatrix<T>::rows() const -> size_type
{
	return m_rows;
}

template <class T>
auto RowMatrix<T>::cols() const -> size_type
{
	return m_cols;
}

template <class T>
auto RowMatrix<T>::row_left(size_type i) const -> size_type
{
	check_bounds(i, 0);
	return m_offsets[i];
}

template <class T>
auto RowMatrix<T>::row_right(size_type i) const -> size_type
{
	check_bounds(i, 0);
	return m_offsets[i] + m_storage[i].size();
}

template <class T>
auto RowMatrix<T>::operator[](size_type i) const -> row_const_proxy
{
	return{ this, i };
}

template <class T>
auto RowMatrix<T>::operator[](size_type i) -> row_proxy
{
	return{ this, i };
}

template <class T>
void RowMatrix<T>::compress()
{
	Equals<T> eq_zero{ static_cast<T>(0) };

	for (size_type i = 0; i < m_rows; ++i) {
		auto &row_data = m_storage[i];

		auto left = std::find_if_not(row_data.cbegin(), row_data.cend(), eq_zero) - row_data.cbegin();
		auto right = row_data.size() - (std::find_if_not(row_data.crbegin(), row_data.crend() - left, eq_zero) - row_data.crbegin());

		// Shrink if non-empty, else free row.
		if (right - left) {
			row_data.erase(row_data.begin() + right, row_data.end());
			row_data.erase(row_data.begin(), row_data.begin() + left);
		} else {
			row_data.clear();
			m_offsets[i] = 0;
		}
	}
}

template <class T>
RowMatrix<T> operator~(const RowMatrix<T> &r)
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

template <class T>
RowMatrix<T> operator*(const RowMatrix<T> &lhs, const RowMatrix<T> &rhs)
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

template class RowMatrix<float>;
template class RowMatrix<double>;
template class RowMatrix<long double>;

template RowMatrix<float> operator~(const RowMatrix<float> &r);
template RowMatrix<double> operator~(const RowMatrix<double> &r);
template RowMatrix<long double> operator~(const RowMatrix<long double> &r);

template RowMatrix<float> operator*(const RowMatrix<float> &lhs, const RowMatrix<float> &rhs);
template RowMatrix<double> operator*(const RowMatrix<double> &lhs, const RowMatrix<double> &rhs);
template RowMatrix<long double> operator*(const RowMatrix<long double> &lhs, const RowMatrix<long double> &rhs);

} // namespace zimg
