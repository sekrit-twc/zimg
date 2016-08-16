#include <algorithm>

#include "gtest/gtest.h"
#include "mock_filter.h"

MockFilter::MockFilter(unsigned width, unsigned height, zimg::PixelType type, const zimg::graph::ImageFilter::filter_flags &flags) :
	m_attr{ width, height, type },
	m_flags(flags),
	m_total_calls{},
	m_simultaneous_lines{ flags.entire_plane ? zimg::graph::BUFFER_MAX : 1 },
	m_horizontal_support{},
	m_vertical_support{}
{
}

unsigned MockFilter::get_total_calls() const
{
	return m_total_calls;
}

void MockFilter::set_simultaneous_lines(unsigned n)
{
	if (!get_flags().entire_plane)
		m_simultaneous_lines = n;
}

void MockFilter::set_horizontal_support(unsigned n)
{
	if (!get_flags().entire_row)
		m_horizontal_support = n;
}

void MockFilter::set_vertical_support(unsigned n)
{
	if (!get_flags().entire_plane)
		m_vertical_support = n;
}

zimg::graph::ImageFilter::filter_flags MockFilter::get_flags() const
{
	return m_flags;
}

zimg::graph::ImageFilter::image_attributes MockFilter::get_image_attributes() const
{
	return m_attr;
}

zimg::graph::ImageFilter::pair_unsigned MockFilter::get_required_row_range(unsigned i) const
{
	if (get_flags().entire_plane) {
		EXPECT_EQ(0U, i);
		return{ 0, m_attr.height };
	} else {
		EXPECT_LT(i, m_attr.height);
		return{ std::max(i, m_vertical_support) - m_vertical_support,
		        std::min(i + get_simultaneous_lines() + m_vertical_support, m_attr.height) };
	}
}

zimg::graph::ImageFilter::pair_unsigned MockFilter::get_required_col_range(unsigned left, unsigned right) const
{
	if (get_flags().entire_row) {
		EXPECT_EQ(0U, left);
		EXPECT_EQ(m_attr.width, right);
		return{ 0, m_attr.width };
	} else {
		EXPECT_LE(left, right);
		EXPECT_LE(right, m_attr.width);
		return{ std::max(left, m_horizontal_support) - m_horizontal_support,
		        std::min(right + m_horizontal_support, m_attr.width) };
	}
}

unsigned MockFilter::get_simultaneous_lines() const
{
	return m_simultaneous_lines;
}

unsigned MockFilter::get_max_buffering() const
{
	return get_flags().entire_plane ? zimg::graph::BUFFER_MAX : get_simultaneous_lines() + 2 * m_vertical_support;
}


size_t MockFilter::get_context_size() const
{
	return sizeof(context);
}

size_t MockFilter::get_tmp_size(unsigned left, unsigned right) const
{
	return 0;
}

void MockFilter::init_context(void *ctx) const
{
	new (ctx) context{};
}

void MockFilter::process(void *ctx, const zimg::graph::ImageBuffer<const void> *src, const zimg::graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const
{
	context *audit_ctx = static_cast<context *>(ctx);
	auto flags = get_flags();

	ASSERT_LT(i, m_attr.height);
	ASSERT_LE(left, right);
	ASSERT_LE(right, m_attr.width);

	if (flags.has_state && (left == audit_ctx->last_left && right == audit_ctx->last_right))
		ASSERT_EQ(audit_ctx->last_line + get_simultaneous_lines(), i);

	if (flags.entire_row) {
		ASSERT_EQ(0U, left);
		ASSERT_EQ(m_attr.width, right);
	}

	if (flags.entire_plane)
		ASSERT_EQ(0U, i);

	for (unsigned p = 0; p < (flags.color ? 3U : 1U); ++p) {
		if (!flags.in_place)
			ASSERT_NE(src[p].data(), dst[p].data());

		if (flags.entire_plane) {
			ASSERT_EQ(zimg::graph::BUFFER_MAX, src[p].mask());
			ASSERT_EQ(zimg::graph::BUFFER_MAX, dst[p].mask());
		}
	}

	audit_ctx->last_line = i;
	audit_ctx->last_left = left;
	audit_ctx->last_right = right;
	++m_total_calls;
}


template <class T>
T SplatFilter<T>::splat_byte(unsigned char b)
{
	T val;

	for (size_t i = 0; i < sizeof(val); ++i) {
		reinterpret_cast<unsigned char *>(&val)[i] = b;
	}
	return val;
}

template <class T>
SplatFilter<T>::SplatFilter(unsigned width, unsigned height, zimg::PixelType type, const zimg::graph::ImageFilter::filter_flags &flags) :
	MockFilter(width, height, type, flags),
	m_src_val{ splat_byte(0xCD) },
	m_dst_val{ splat_byte(0xDC) },
	m_input_checking{ true }
{
}

template <class T>
void SplatFilter<T>::set_input_val(unsigned char x)
{
	m_src_val = splat_byte(x);
}

template <class T>
void SplatFilter<T>::set_output_val(unsigned char x)
{
	m_dst_val = splat_byte(x);
}

template <class T>
void SplatFilter<T>::enable_input_checking(bool enabled)
{
	m_input_checking = enabled;
}

template <class T>
void SplatFilter<T>::process(void *ctx, const zimg::graph::ImageBuffer<const void> *src, const zimg::graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const
{
	MockFilter::process(ctx, src, dst, tmp, i, left, right);

	pair_unsigned row_range = get_required_row_range(i);
	pair_unsigned col_range = get_required_col_range(left, right);

	for (unsigned p = 0; p < (get_flags().color ? 3U : 1U); ++p) {
		if (m_input_checking) {
			for (unsigned ii = row_range.first; ii < row_range.second; ++ii) {
				const T *src_first = zimg::graph::static_buffer_cast<const T>(src[p])[ii] + col_range.first;
				const T *src_last = zimg::graph::static_buffer_cast<const T>(src[p])[ii] + col_range.second;

				const T *find_pos = std::find_if(src_first, src_last, [=](T x) { return x != m_src_val; });
				ASSERT_TRUE(find_pos == src_last) << "found invalid value at position: (" << ii << ", " << find_pos - src_first << ")";
			}
		}

		for (unsigned ii = i; ii < std::min(i + get_simultaneous_lines(), m_attr.height); ++ii) {
			std::fill(zimg::graph::static_buffer_cast<T>(dst[p])[ii] + left, zimg::graph::static_buffer_cast<T>(dst[p])[ii] + right, m_dst_val);
		}
	}
}

template class SplatFilter<uint8_t>;
template class SplatFilter<uint16_t>;
template class SplatFilter<float>;
