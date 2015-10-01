#include <algorithm>
#include <limits>
#include <random>
#include "common/linebuffer.h"
#include "common/zassert.h"
#include "common/zfilter.h"
#include "depth/quantize.h"

#include "gtest/gtest.h"
#include "audit_buffer.h"

namespace {;

template <class InputIt, class T>
bool contains_only(InputIt first, InputIt last, const T &value)
{
	return std::find_if(first, last, [=](const T &x) { return x != value; }) == last;
}

template <class T>
T float_as(float x)
{
	_zassert(false, "fail");
	return{};
}

template <>
uint16_t float_as<uint16_t>(float x)
{
	return zimg::depth::float_to_half(x);
}

template <>
float float_as<float>(float x)
{
	return x;
}

template <class T>
class Mt19937Generator {
	std::mt19937 m_gen;
	uint32_t m_depth;
	bool m_float;
	bool m_chroma;
public:
	Mt19937Generator(unsigned p, unsigned i, unsigned left, const zimg::PixelFormat &format) :
		m_gen{ ((uint_fast32_t)p << 30) | i },
		m_depth{ (unsigned)format.depth },
		m_float{ format.type == zimg::PixelType::HALF || format.type == zimg::PixelType::FLOAT },
		m_chroma{ p > 0 || format.chroma }
	{
		m_gen.discard(left);
	}

	T operator()()
	{
		if (m_float) {
			double x = static_cast<double>(m_gen() - std::mt19937::min()) / static_cast<double>(std::mt19937::max() - std::mt19937::min());
			float xf = static_cast<float>(x);
			return float_as<T>(m_chroma ? xf - 0.5f : xf);
		} else {
			return static_cast<T>(m_gen() & ((1 << m_depth) - 1));
		}
	}
};

} // namespace


template <class T>
T AuditBuffer<T>::splat_byte(unsigned char b)
{
	T val;

	for (size_t i = 0; i < sizeof(val); ++i) {
		((unsigned char *)&val)[i] = b;
	}
	return val;
}

template <class T>
void AuditBuffer<T>::add_guard_bytes()
{
	for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
		std::fill(m_vector[p].begin(), m_vector[p].begin() + m_buffer.stride[p] / (ptrdiff_t)sizeof(T), m_guard_val);

		for (unsigned i = 0; i < m_buffer_height[p]; ++i) {
			T *line_base = (T *)m_buffer.data[p] + (ptrdiff_t)i * stride_T(p);

			T *line_guard_left = line_base - zimg::AlignmentOf<T>::value;
			T *line_guard_right = line_guard_left + stride_T(p);

			std::fill(line_guard_left, line_base, m_guard_val);
			std::fill(line_base + m_width[p], line_guard_right, m_guard_val);
		}

		std::fill(m_vector[p].end() - stride_T(p), m_vector[p].end(), m_guard_val);
	}
}

template <class T>
ptrdiff_t AuditBuffer<T>::stride_T(unsigned p) const
{
	return m_buffer.stride[p] / (ptrdiff_t)sizeof(T);
}

template <class T>
AuditBuffer<T>::AuditBuffer(unsigned width, unsigned height, zimg::PixelFormat format, unsigned lines, unsigned subsample_w, unsigned subsample_h, bool color) :
	m_format(format),
	m_width{},
	m_buffer_height{},
	m_subsample_w{ subsample_w },
	m_subsample_h{ subsample_h },
	m_fill_val{ splat_byte(0xCD), splat_byte(0xCD), splat_byte(0xCD) },
	m_guard_val{ splat_byte(0xFE) },
	m_color{ color }
{
	unsigned mask = zimg::select_zimg_buffer_mask(lines);

	for (unsigned p = 0; p < (color ? 3U : 1U); ++p) {
		unsigned width_plane = p ? width >> subsample_w : width;
		unsigned height_plane = p ? height >> subsample_h : height;

		unsigned mask_plane = p ? (mask == (unsigned)-1 ? mask : mask >> subsample_h) : mask;
		unsigned buffer_height = (mask_plane == (unsigned)-1) ? height_plane : mask_plane + 1;

		size_t guarded_linesize = (zimg::align(width_plane, zimg::AlignmentOf<T>::value) + 2 * zimg::AlignmentOf<T>::value) * sizeof(T);
		size_t guarded_linecount = buffer_height + 2;

		m_vector[p].resize(guarded_linesize * guarded_linecount);

		m_buffer.data[p] = m_vector[p].data() + guarded_linesize + zimg::AlignmentOf<T>::value;
		m_buffer.stride[p] = guarded_linesize;
		m_buffer.mask[p] = mask_plane;

		m_width[p] = width_plane;
		m_buffer_height[p] = buffer_height;
	}

	add_guard_bytes();
}

template <class T>
void AuditBuffer<T>::set_format(const zimg::PixelFormat &format)
{
	m_format = format;
}

template <class T>
void AuditBuffer<T>::set_fill_val(unsigned char x)
{
	set_fill_val(x, 0);
	set_fill_val(x, 1);
	set_fill_val(x, 2);
}

template <class T>
void AuditBuffer<T>::set_fill_val(unsigned char x, unsigned plane)
{
	m_fill_val[plane] = splat_byte(x);
}

template <class T>
bool AuditBuffer<T>::detect_write(unsigned i, unsigned left, unsigned right) const
{
	bool write = true;

	for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
		zimg::LineBuffer<const T> linebuf{ m_buffer, p };

		unsigned i_plane = i >> (p ? m_subsample_h : 0);
		unsigned left_plane = left >> (p ? m_subsample_w : 0);
		unsigned right_plane = right >> (p ? m_subsample_h : 0);

		write = write && !contains_only(linebuf[i_plane] + left_plane, linebuf[i_plane] + right_plane, m_fill_val[p]);
	}
	return write;
}

template <class T>
void AuditBuffer<T>::assert_eq(const AuditBuffer &other, unsigned i, unsigned left, unsigned right) const
{
	for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
		zimg::LineBuffer<const T> this_buf{ m_buffer, p };
		zimg::LineBuffer<const T> other_buf{ other.m_buffer, p };

		unsigned left_plane = left >> (p ? m_subsample_w : 0);
		unsigned right_plane = right >> (p ? m_subsample_h : 0);

		ASSERT_TRUE(std::equal(this_buf[i] + left_plane, this_buf[i] + right_plane, other_buf[i] + left_plane)) <<
			"mismatch at line: " << i;
	}
}

template <class T>
void AuditBuffer<T>::assert_guard_bytes() const
{
	for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
		ASSERT_TRUE(contains_only(m_vector[p].begin(), m_vector[p].begin() + stride_T(p), m_guard_val)) <<
			"header guard bytes corrupted";

		for (unsigned i = 0; i < m_buffer_height[p]; ++i) {
			const T *line_base = (const T *)m_buffer.data[p] + (ptrdiff_t)i * stride_T(p);
			const T *line_guard_left = line_base - zimg::AlignmentOf<T>::value;
			const T *line_guard_right = line_guard_left + stride_T(p);

			ASSERT_TRUE(contains_only(line_guard_left, line_base, m_guard_val)) <<
				"line guard header corrupted at: " << i;
			ASSERT_TRUE(contains_only(line_base + m_width[p], line_guard_right, m_guard_val)) <<
				"line guard footer corrupted at: " << i;
		}

		ASSERT_TRUE(contains_only(m_vector[p].end() - stride_T(p), m_vector[p].end(), m_guard_val)) <<
			"footer guard bytes corrupted";
	}
}

template <class T>
void AuditBuffer<T>::random_fill(unsigned first_row, unsigned last_row, unsigned first_col, unsigned last_col)
{
	for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
		zimg::LineBuffer<T> linebuf{ m_buffer, p };

		unsigned first_row_plane = first_row << (p ? m_subsample_h : 0);
		unsigned last_row_plane = last_row << (p ? m_subsample_h : 0);
		unsigned first_col_plane = first_col << (p ? m_subsample_w : 0);
		unsigned last_col_plane = last_col << (p ? m_subsample_w : 0);

		for (unsigned i = first_row_plane; i < last_row_plane; ++i) {
			Mt19937Generator<T> engine{ p, i, first_col_plane, m_format };

			std::generate(linebuf[i] + first_col_plane, linebuf[i] + last_col_plane, engine);
		}
	}
}

template <class T>
void AuditBuffer<T>::default_fill()
{
	for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
		zimg::LineBuffer<T> linebuf{ m_buffer, p };

		for (unsigned i = 0; i < m_buffer_height[p]; ++i) {
			std::fill_n(linebuf[i], m_width[p], m_fill_val[p]);
		}
	}
}

template <class T>
const zimg::ZimgImageBuffer &AuditBuffer<T>::as_image_buffer() const
{
	return m_buffer;
}

template class AuditBuffer<uint8_t>;
template class AuditBuffer<uint16_t>;
template class AuditBuffer<float>;
