#include "Common/except.h"
#include "matrix.h"

namespace zimg {;
namespace unresize {;

RowMatrix::Row::Row(RowMatrix &parent, int row) : m_parent(parent), m_row(row)
{
}

RowMatrix::Element RowMatrix::Row::operator[](int j)
{
	return{ m_parent, m_row, j };
}


RowMatrix::ConstRow::ConstRow(const RowMatrix &parent, int row) : m_parent(parent), m_row(row)
{
}

float RowMatrix::ConstRow::operator[](int j)
{
	return m_parent.get_element_val(m_row, j);
}


RowMatrix::Element::Element(RowMatrix &parent, int row, int col) : m_parent(parent), m_row(row), m_col(col)
{
}

RowMatrix::Element &RowMatrix::Element::operator=(float f)
{
	m_parent.get_element_ref(m_row, m_col) = f;
	return *this;
}

RowMatrix::Element::operator float()
{
	return m_parent.get_element_val(m_row, m_col);
}


void RowMatrix::check_bounds(int i, int j) const
{
	if (i < 0 || i > m_rows)
		throw ZimgLogicError{ "matrix row out of bounds" };
	if (j < 0 || j > m_cols)
		throw ZimgLogicError{ "matrix col out of bounds" };
}

RowMatrix::RowMatrix(int m, int n) : m_storage(m), m_offsets(m), m_rows(m), m_cols(n)
{
}

int RowMatrix::rows() const
{
	return m_rows;
}

int RowMatrix::cols() const
{
	return m_cols;
}

RowMatrix::Row RowMatrix::operator[](int i)
{
	return{ *this, i };
}

RowMatrix::ConstRow RowMatrix::operator[](int i) const
{
	return{ *this, i };
}

int RowMatrix::row_left(int i) const
{
	return m_offsets[i];
}

int RowMatrix::row_right(int i) const
{
	return m_offsets[i] + (int)m_storage[i].size();
}

float &RowMatrix::get_element_ref(int i, int j)
{
	check_bounds(i, j);

	std::vector<float> &data = m_storage[i];
	int left = row_left(i);
	int right = row_right(i);

	// Resize row if needed.
	if (data.empty()) {
		data.resize(1);
		left = j;
	} else if (j < left) {
		data.insert(data.begin(), left - j, 0);
		left = j;
		m_offsets[i] = j;
	} else if (j >= right) {
		data.insert(data.end(), j - right + 1, 0);
	}
	m_offsets[i] = left;

	return data[j - left];
}

float RowMatrix::get_element_val(int i, int j) const
{
	check_bounds(i, j);

	int left = row_left(i);
	int right = row_right(i);

	if (j < left || j >= right)
		return 0;
	else
		return m_storage[i][j - left];
}

void RowMatrix::compress()
{
	for (int i = 0; i < m_rows; ++i) {
		int left_off;
		int right_off;

		for (left_off = 0; left_off < (int)m_storage[i].size(); ++left_off) {
			if (m_storage[i][left_off])
				break;
		}

		for (right_off = (int)m_storage[i].size() - 1; right_off > left_off; --right_off) {
			if (m_storage[i][right_off])
				break;
		}

		if (right_off - left_off) {
			m_offsets[i] += left_off;
			m_storage[i] = std::vector<float>(m_storage[i].begin() + left_off, m_storage[i].begin() + right_off + 1);
		} else {
			m_offsets[i] = 0;
			m_storage[i].clear();
		}
	}
}


TridiagonalLU::TridiagonalLU(int dim) : m_diag_l(dim), m_diag_u(dim), m_diag_c(dim)
{
}

int TridiagonalLU::dim() const
{
	return m_dim;
}

float &TridiagonalLU::l(int i)
{
	return m_diag_l.at(i);
}

float TridiagonalLU::l(int i) const
{
	return m_diag_l.at(i);
}

float &TridiagonalLU::u(int i)
{
	return m_diag_u.at(i);
}

float TridiagonalLU::u(int i) const
{
	return m_diag_u.at(i);
}

float &TridiagonalLU::c(int i)
{
	return m_diag_c.at(i);
}

float TridiagonalLU::c(int i) const
{
	return m_diag_c.at(i);
}


RowMatrix matrix_matrix_product(const RowMatrix &lhs, const RowMatrix &rhs)
{
	RowMatrix m(lhs.rows(), rhs.cols());

	for (int i = 0; i < lhs.rows(); ++i) {
		for (int j = 0; j < rhs.cols(); ++j) {
			float accum = 0;

			for (int k = lhs.row_left(i); k < lhs.row_right(i); ++k) {
				accum += lhs[i][k] * rhs[k][j];
			}
			if (accum)
				m[i][j] = accum;
		}
	}
	m.compress();

	return m;
}

RowMatrix transpose(const RowMatrix &r)
{
	RowMatrix m(r.cols(), r.rows());

	for (int i = 0; i < r.rows(); ++i) {
		for (int j = r.row_left(i); j < r.row_right(i); ++j) {
			m[j][i] = r[i][j];
		}
	}
	m.compress();

	return m;
}

TridiagonalLU tridiagonal_decompose(const RowMatrix &r)
{
	int n = r.rows();
	TridiagonalLU lu(n);

	lu.c(0) = 0;
	lu.l(0) = r[0][0];
	lu.u(0) = r[0][1] / (r[0][0] + 0.001f);

	for (int i = 1; i < n - 1; ++i) {
		lu.c(i) = r[i][i - 1];
		lu.l(i) = r[i][i] - lu.c(i) * lu.u(i - 1);
		lu.u(i) = r[i][i + 1] / (lu.l(i) + 0.001f);
	}

	lu.c(n - 1) = r[n - 1][n - 2];
	lu.l(n - 1) = r[n - 1][n - 1] - lu.c(n - 1) * lu.u(n - 2);
	lu.u(n - 1) = 0;

	return lu;
}

} // namespace unresize
} // namespace zimg
