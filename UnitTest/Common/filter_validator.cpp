#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>

#include "Common/align.h"
#include "Common/linebuffer.h"
#include "Common/pixel.h"
#include "Common/zfilter.h"

#include "gtest/gtest.h"

extern "C" {
	#include "sha1/sha1.h"
}

#include "filter_validator.h"

namespace {;

class AuditBuffer {
	zimg::AlignedVector<uint8_t> m_vector[3];
	zimg::ZimgImageBuffer m_buffer;
	unsigned m_rowsize;
	unsigned m_buffer_height;
	unsigned m_pixel_size;
	uint8_t m_guard_val = 0xFE;
	bool m_color;

	void add_guard_bytes()
	{
		for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
			if (m_vector[p].empty())
				continue;

			uint8_t *base = m_vector[p].data();
			ptrdiff_t stride = m_buffer.stride[p];

			std::fill_n(base, stride, m_guard_val);

			for (unsigned i = 0; i < m_buffer_height; ++i) {
				uint8_t *line_base = base + (ptrdiff_t)(i + 1) * stride;

				std::fill_n(line_base, zimg::ALIGNMENT, m_guard_val);
				std::fill_n(line_base + m_rowsize, stride - m_rowsize, m_guard_val);
			}

			std::fill_n(base + (ptrdiff_t)(m_buffer_height + 1) * stride, stride, m_guard_val);
		}
	}
public:
	AuditBuffer(unsigned width, unsigned height, zimg::PixelType type, unsigned lines, bool color) :
		m_rowsize{ width * zimg::pixel_size(type) },
		m_buffer_height{},
		m_pixel_size{ (unsigned)zimg::pixel_size(type) },
		m_color{ color }
	{
		unsigned mask = zimg::select_zimg_buffer_mask(lines);
		m_buffer_height = (mask == -1) ? height : mask + 1;

		for (unsigned p = 0; p < (color ? 3U : 1U); ++p) {
			unsigned guarded_linesize = zimg::align(m_rowsize, zimg::ALIGNMENT) + 2 * zimg::ALIGNMENT;
			unsigned guarded_linecount = m_buffer_height + 2;

			m_vector[p].resize((size_t)guarded_linesize * guarded_linecount);

			m_buffer.data[p] = m_vector[p].data() + guarded_linesize + zimg::ALIGNMENT;
			m_buffer.stride[p] = guarded_linesize;
			m_buffer.mask[p] = mask;
		}

		add_guard_bytes();
	}

	bool detect_write(unsigned i, unsigned left, unsigned right)
	{
		bool write = true;
		left *= m_pixel_size;
		right *= m_pixel_size;

		for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
			zimg::LineBuffer<uint8_t> buf{ m_buffer, p };

			auto it = std::find_if(buf[i] + left, buf[i] + right, [=](uint8_t x) { return x != 0; });
			write = write && (it != buf[i] + right);
		}

		return write;
	}

	void compare(const AuditBuffer &other, unsigned i, unsigned left, unsigned right)
	{
		left *= m_pixel_size;
		right *= m_pixel_size;

		for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
			zimg::LineBuffer<uint8_t> this_buf{ m_buffer, p };
			zimg::LineBuffer<uint8_t> other_buf{ other.m_buffer, p };

			ASSERT_TRUE(std::equal(this_buf[i] + left, this_buf[i] + right, other_buf[i] + left)) <<
				"mismatch at line: " << i;
		}
	}

	void check_guard_bytes()
	{
		const auto pred = [=](uint8_t x) { return x != m_guard_val; };

		for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
			const uint8_t *base = m_vector[p].data();
			ptrdiff_t stride = m_buffer.stride[p];

			auto it = std::find_if(base, base + stride, pred);
			ASSERT_TRUE(it == base + stride) << "header guard bytes corrupted";

			for (unsigned i = 0; i < m_buffer_height; ++i) {
				const uint8_t *line_base = base + (ptrdiff_t)(i + 1) * stride;

				it = std::find_if(line_base, line_base + zimg::ALIGNMENT, pred);
				ASSERT_TRUE(it == line_base + zimg::ALIGNMENT) << "line guard header corrupted at: " << i;

				it = std::find_if(line_base  +zimg::ALIGNMENT + m_rowsize, line_base + stride, pred);
				ASSERT_TRUE(it == line_base + stride) << "line guard footer corrupted at: " << i;
			}

			it = std::find_if(base + (ptrdiff_t)(m_buffer_height + 1) * stride, base + (ptrdiff_t)(m_buffer_height + 2) * stride, pred);
			ASSERT_TRUE(it == base + (ptrdiff_t)(m_buffer_height + 2) * stride) << "footer guard bytes corrupted";
		}
	}

	void random_fill(unsigned first_row, unsigned last_row, unsigned first_col, unsigned last_col)
	{
		first_col *= m_pixel_size;
		last_col *= m_pixel_size;

		for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
			zimg::LineBuffer<uint8_t> buf{ m_buffer, p };

			for (unsigned i = first_row; i < last_row; ++i) {
				std::mt19937 engine{ i };

				for (unsigned j = 0; j < first_col; ++j) {
					engine();
				}

				std::generate_n(buf[i] + first_col, last_col - first_col, engine);
			}
		}
	}

	void zero_fill()
	{
		for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
			zimg::LineBuffer<uint8_t> buf{ m_buffer, p };

			for (unsigned i = 0; i < m_buffer_height; ++i) {
				std::fill_n(buf[i], m_rowsize, 0);
			}
		}
	}

	const zimg::ZimgImageBuffer &as_buffer() const
	{
		return m_buffer;
	}
};


void decode_sha1(const char *str, unsigned char digest[20])
{
	for (unsigned i = 0; i < 20; ++i) {
		char buf[3] = { str[i * 2], str[i * 2 + 1], '\0' };
		digest[i] = static_cast<unsigned char>(std::stoi(buf, nullptr, 16));
	}
}

void hash_buffer(const AuditBuffer &buf, unsigned p, unsigned width, unsigned height, zimg::PixelType type, unsigned char digest[20])
{
	const zimg::ZimgImageBufferConst &image_buffer = buf.as_buffer();
	SHA1_CTX sha_ctx;

	SHA1Init(&sha_ctx);

	for (unsigned i = 0; i < height; ++i) {
		const unsigned char *ptr = (const unsigned char *)image_buffer.data[p] + (ptrdiff_t)i * image_buffer.stride[p];
		SHA1Update(&sha_ctx, ptr, width * zimg::pixel_size(type));
	}

	SHA1Final(digest, &sha_ctx);
}

std::string hash_to_str(const unsigned char digest[20])
{
	std::string s;
	s.reserve(40);

	for (unsigned i = 0; i < 20; ++i) {
		char x[3];

		sprintf(x, "%02x", digest[i]);

		s.push_back(x[0]);
		s.push_back(x[1]);
	}

	return s;
}

void validate_flags(const zimg::IZimgFilter *filter)
{
	auto flags = filter->get_flags();

	ASSERT_EQ(zimg::API_VERSION, flags.version);

	if (flags.entire_plane && !flags.entire_row)
		FAIL() << "filter must set entire_row if entire_plane is set";
	if (flags.in_place && !flags.same_row)
		FAIL() << "filter must set same_row if in_place is set";

	if (flags.entire_plane && filter->get_max_buffering() != (unsigned)-1)
		FAIL() << "filter must buffer entire image if entire_plane is set";
	if (flags.entire_plane && filter->get_simultaneous_lines() != (unsigned)-1)
		FAIL() << "filter must output entire image if entire_plane is set";
}

void validate_same_row(const zimg::IZimgFilter *filter)
{
	auto flags = filter->get_flags();
	auto attr = filter->get_image_attributes();

	unsigned fstep = filter->get_simultaneous_lines();
	unsigned step = flags.has_state ? fstep : 1;

	for (unsigned i = 0; i < attr.height; i += step) {
		auto range = filter->get_required_row_range(i);
		ASSERT_EQ(i, range.first);
		ASSERT_EQ(i + fstep, range.second);
	}
}

void validate_filter_plane(const zimg::IZimgFilter *filter, AuditBuffer *src_buffer, AuditBuffer *dst_buffer)
{
	auto attr = filter->get_image_attributes();

	unsigned step = filter->get_simultaneous_lines();
	zimg::AlignedVector<char> ctx(filter->get_context_size());
	zimg::AlignedVector<char> tmp(filter->get_tmp_size(0, attr.width));

	filter->init_context(ctx.data());

	for (unsigned i = 0; i < attr.height; i += step) {
		filter->process(ctx.data(), src_buffer->as_buffer(), dst_buffer->as_buffer(), tmp.data(), i, 0, attr.width);

		for (unsigned ii = i; ii < std::min(i + step, attr.height); ++ii) {
			ASSERT_TRUE(dst_buffer->detect_write(ii, 0, attr.width)) <<
				"expected write to buffer at line: " << ii;
		}
		for (unsigned ii = i + step; ii < attr.height; ++ii) {
			ASSERT_FALSE(dst_buffer->detect_write(ii, 0, attr.width)) <<
				"unexpected write to buffer at line: " << ii;
		}
	}

	src_buffer->check_guard_bytes();
	dst_buffer->check_guard_bytes();
}

void validate_filter_buffered(const zimg::IZimgFilter *filter, unsigned src_width, unsigned src_height, zimg::PixelType src_type, const AuditBuffer &ref_buffer)
{
	auto flags = filter->get_flags();
	auto attr = filter->get_image_attributes();

	AuditBuffer src_buf{ src_width, src_height, src_type, filter->get_max_buffering(), !!flags.color };
	AuditBuffer dst_buf{ attr.width, attr.height, attr.type, filter->get_simultaneous_lines(), !!flags.color };

	unsigned init = flags.has_state ? 0 : attr.height / 4;
	unsigned vstep = filter->get_simultaneous_lines();
	unsigned step = flags.has_state ? vstep : vstep * 2;

	unsigned left = flags.entire_row ? 0 : attr.width / 4;
	unsigned right = flags.entire_row ? attr.width : attr.width / 2;

	zimg::AlignedVector<char> ctx(filter->get_context_size());
	zimg::AlignedVector<char> tmp(filter->get_tmp_size(left, right));

	auto col_range = filter->get_required_col_range(left, right);

	for (unsigned i = init; i < attr.height; i += step) {
		auto row_range = filter->get_required_row_range(i);

		src_buf.zero_fill();
		src_buf.random_fill(row_range.first, row_range.second, col_range.first, col_range.second);
		dst_buf.zero_fill();

		filter->process(ctx.data(), src_buf.as_buffer(), dst_buf.as_buffer(), tmp.data(), i, left, right);

		src_buf.check_guard_bytes();
		dst_buf.check_guard_bytes();

		for (unsigned ii = i; ii < std::min(i + vstep, attr.height); ++ii) {
			dst_buf.compare(ref_buffer, ii, left, right);
		}
	}
}

} // namespace


void validate_filter(const zimg::IZimgFilter *filter, unsigned src_width, unsigned src_height, zimg::PixelType src_type, const char * const sha1_str[3])
{
	zimg::ZimgFilterFlags flags = filter->get_flags();
	auto attr = filter->get_image_attributes();

	validate_flags(filter);

	if (flags.same_row)
		validate_same_row(filter);

	AuditBuffer src_buf{ src_width, src_height, src_type, (unsigned)-1, !!flags.color };
	AuditBuffer dst_buf{ attr.width, attr.height, attr.type, (unsigned)-1, !!flags.color };

	src_buf.random_fill(0, src_height, 0, src_width);
	dst_buf.zero_fill();

	validate_filter_plane(filter, &src_buf, &dst_buf);

	if (sha1_str) {
		for (unsigned p = 0; p < (flags.color ? 3U : 1U); ++p) {
			std::array<unsigned char, 20> expected_sha1;
			std::array<unsigned char, 20> test_sha1;

			if (sha1_str[p]) {
				decode_sha1(sha1_str[p], expected_sha1.data());
				hash_buffer(dst_buf, p, attr.width, attr.height, attr.type, test_sha1.data());

				EXPECT_TRUE(expected_sha1 == test_sha1) <<
					"sha1 mismatch: plane (" << p << ") expected (" << hash_to_str(expected_sha1.data()) << ") actual (" << hash_to_str(test_sha1.data()) << ")";
			}
		}
	}

	if (!flags.entire_plane)
		validate_filter_buffered(filter, src_width, src_height, src_type, dst_buf);
}
