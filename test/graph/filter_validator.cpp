#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <typeinfo>
#include <utility>

#include "common/alloc.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "depth/quantize.h"

#include "gtest/gtest.h"

#include <endian.h>

extern "C" {
  #include "sha1/sha1.h"
}

#include "audit_buffer.h"
#include "filter_validator.h"

namespace {

void decode_sha1(const char *str, unsigned char digest[20])
{
	for (unsigned i = 0; i < 20; ++i) {
		char buf[3] = { str[i * 2], str[i * 2 + 1], '\0' };
		digest[i] = static_cast<unsigned char>(std::stoi(buf, nullptr, 16));
	}
}

template <class T>
void hash_buffer(const AuditBuffer<T> &buf, unsigned p, unsigned width, unsigned height, unsigned char digest[20])
{
	auto image_buffer = buf.as_read_buffer();
	SHA1_CTX sha_ctx;

	SHA1Init(&sha_ctx);

	for (unsigned i = 0; i < height; ++i) {
		const unsigned char *ptr = static_cast<const unsigned char *>(image_buffer[p][i]);
#if (__BYTE_ORDER == __LITTLE_ENDIAN)
		SHA1Update(&sha_ctx, ptr, width * sizeof(T));
#else
		for (unsigned j = 0; j < width; j++)
			for (int k = sizeof(T) - 1; k >= 0; k--)
				SHA1Update(&sha_ctx, ptr + (j * sizeof(T)) + k, 1);
#endif
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


AuditBufferType select_buffer_type(bool color, bool yuv)
{
	if (color)
		return yuv ? AuditBufferType::COLOR_YUV : AuditBufferType::COLOR_RGB;
	else
		return AuditBufferType::PLANE;
}


template <class T>
std::pair<double, double> snr_line(const T *ref, const T *test, unsigned count, zimg::PixelType type)
{
	double signal = 0;
	double noise = 0;

	for (unsigned n = 0; n < count; ++n) {
		signal += (double)ref[n] * (double)ref[n];
		noise += (double)(ref[n] - test[n]) * (double)(ref[n] - test[n]);
	}

	return{ signal, noise };
}

template <>
std::pair<double, double> snr_line<uint16_t>(const uint16_t *ref, const uint16_t *test, unsigned count, zimg::PixelType type)
{
	double signal = 0;
	double noise = 0;

	for (unsigned n = 0; n < count; ++n) {
		double ref_val;
		double test_val;

		if (type == zimg::PixelType::HALF) {
			ref_val = zimg::depth::half_to_float(ref[n]);
			test_val = zimg::depth::half_to_float(test[n]);
		} else {
			ref_val = ref[n];
			test_val = test[n];
		}

		signal += ref_val * ref_val;
		noise += (ref_val - test_val) * (ref_val - test_val);
	}

	return{ signal, noise };
}

template <class T>
double snr_buffer(const AuditBuffer<T> &ref, const AuditBuffer<T> &test, unsigned width, unsigned height, zimg::PixelType type, bool color)
{
	auto ref_buf = ref.as_read_buffer();
	auto test_buf = test.as_read_buffer();
	double snr = INFINITY;

	for (unsigned p = 0; p < (color ? 3U : 1U); ++p) {
		double signal_plane = 0.0;
		double noise_plane = 0.0;

		for (unsigned i = 0; i < height; ++i) {
			const T *ref_ptr = static_cast<const T *>(ref_buf[p][i]);
			const T *test_ptr = static_cast<const T *>(test_buf[p][i]);

			auto snr_pair = snr_line(ref_ptr, test_ptr, width, type);
			signal_plane += snr_pair.first;
			noise_plane += snr_pair.second;
		}

		snr = std::min(snr, signal_plane / noise_plane);
	}

	return 10.0 * std::log10(snr);
}


void validate_flags(const zimg::graph::ImageFilter *filter)
{
	auto flags = filter->get_flags();

	if (flags.entire_plane && !flags.entire_row)
		FAIL() << "filter must set entire_row if entire_plane is set";
	if (flags.in_place && !flags.same_row)
		FAIL() << "filter must set same_row if in_place is set";

	if (flags.entire_plane && filter->get_max_buffering() != zimg::graph::BUFFER_MAX)
		FAIL() << "filter must buffer entire image if entire_plane is set";
	if (flags.entire_plane && filter->get_simultaneous_lines() != zimg::graph::BUFFER_MAX)
		FAIL() << "filter must output entire image if entire_plane is set";
}

void validate_same_row(const zimg::graph::ImageFilter *filter)
{
	auto flags = filter->get_flags();
	auto attr = filter->get_image_attributes();

	unsigned fstep = filter->get_simultaneous_lines();
	unsigned step = flags.has_state ? fstep : 1;

	for (unsigned i = 0; i < attr.height; i += step) {
		auto range = filter->get_required_row_range(i);
		ASSERT_EQ(i, range.first);
		ASSERT_EQ(std::min(i + fstep, attr.height), range.second);
	}
}

template <class T, class U>
void validate_filter_plane(const zimg::graph::ImageFilter *filter, AuditBuffer<T> *src_buffer, AuditBuffer<U> *dst_buffer)
{
	auto attr = filter->get_image_attributes();

	unsigned step = filter->get_simultaneous_lines();
	zimg::AlignedVector<char> ctx(filter->get_context_size());
	zimg::AlignedVector<char> tmp(filter->get_tmp_size(0, attr.width));

	filter->init_context(ctx.data(), 0);

	for (unsigned i = 0; i < attr.height; i += step) {
		filter->process(ctx.data(), src_buffer->as_read_buffer(), dst_buffer->as_write_buffer(), tmp.data(), i, 0, attr.width);

		for (unsigned ii = i; ii < std::min(i + step, attr.height); ++ii) {
			ASSERT_TRUE(dst_buffer->detect_write(ii, 0, attr.width)) <<
				"expected write to buffer at line: " << ii;
		}
		for (unsigned ii = i + step; ii < attr.height; ++ii) {
			ASSERT_FALSE(dst_buffer->detect_write(ii, 0, attr.width)) <<
				"unexpected write to buffer at line: " << ii;
		}
	}

	src_buffer->assert_guard_bytes();
	dst_buffer->assert_guard_bytes();
}

template <class T, class U>
void validate_filter_buffered(const zimg::graph::ImageFilter *filter, unsigned src_width, unsigned src_height, const zimg::PixelFormat &src_format, bool yuv, const AuditBuffer<U> &ref_buffer)
{
	auto flags = filter->get_flags();
	auto attr = filter->get_image_attributes();

	AuditBufferType buffer_type = select_buffer_type(flags.color, yuv);

	AuditBuffer<T> src_buf{ buffer_type, src_width, src_height, src_format, filter->get_max_buffering(), 0, 0 };
	AuditBuffer<U> dst_buf{ buffer_type, attr.width, attr.height, attr.type, filter->get_simultaneous_lines(), 0, 0 };

	unsigned init = flags.has_state ? 0 : attr.height / 4;
	unsigned vstep = filter->get_simultaneous_lines();
	unsigned step = flags.has_state ? vstep : vstep * 2;

	unsigned left = flags.entire_row ? 0 : attr.width / 4 + 1;
	unsigned right = flags.entire_row ? attr.width : attr.width / 2 - 1;

	zimg::AlignedVector<char> ctx(filter->get_context_size());
	zimg::AlignedVector<char> tmp(filter->get_tmp_size(left, right));

	auto col_range = filter->get_required_col_range(left, right);

	filter->init_context(ctx.data(), 0);

	for (unsigned i = init; i < attr.height; i += step) {
		auto row_range = filter->get_required_row_range(i);

		src_buf.default_fill();
		src_buf.random_fill(row_range.first, row_range.second, col_range.first, col_range.second);
		dst_buf.default_fill();

		filter->process(ctx.data(), src_buf.as_read_buffer(), dst_buf.as_write_buffer(), tmp.data(), i, left, right);

		src_buf.assert_guard_bytes();
		dst_buf.assert_guard_bytes();

		for (unsigned ii = i; ii < std::min(i + vstep, attr.height); ++ii) {
			dst_buf.assert_eq(ref_buffer, ii, left, right);
		}
	}
}

template <class T, class U>
struct ValidateFilter {
	void operator()(const zimg::graph::ImageFilter *filter, unsigned src_width, unsigned src_height, const zimg::PixelFormat &src_format, bool yuv,
				    const char * const sha1_str[3])
	{
		auto flags = filter->get_flags();
		auto attr = filter->get_image_attributes();

		validate_flags(filter);

		if (flags.same_row)
			validate_same_row(filter);

		AuditBufferType buffer_type = select_buffer_type(flags.color, yuv);

		AuditBuffer<T> src_buf{ buffer_type, src_width, src_height, src_format, zimg::graph::BUFFER_MAX, 0, 0 };
		AuditBuffer<U> dst_buf{ buffer_type, attr.width, attr.height, attr.type, zimg::graph::BUFFER_MAX, 0, 0 };

		src_buf.random_fill(0, src_height, 0, src_width);
		dst_buf.default_fill();

		validate_filter_plane<T>(filter, &src_buf, &dst_buf);
		validate_filter_buffered<T>(filter, src_width, src_height, src_format, yuv, dst_buf);

		if (!sha1_str)
			return;

		for (unsigned p = 0; p < (flags.color ? 3U : 1U); ++p) {
			std::array<unsigned char, 20> expected_sha1;
			std::array<unsigned char, 20> test_sha1;

			if (sha1_str[p]) {
				decode_sha1(sha1_str[p], expected_sha1.data());
				hash_buffer(dst_buf, p, attr.width, attr.height, test_sha1.data());

				EXPECT_TRUE(expected_sha1 == test_sha1) <<
					"sha1 mismatch: plane (" << p << ") expected (" << hash_to_str(expected_sha1.data()) << ") actual (" << hash_to_str(test_sha1.data()) << ")";
			}
		}
	}
};

template <class T, class U>
struct ValidateFilterReference {
	void operator()(const zimg::graph::ImageFilter *ref_filter, const zimg::graph::ImageFilter *test_filter,
	                unsigned src_width, unsigned src_height, const zimg::PixelFormat &src_format, bool yuv, double snr_thresh)
	{
		zimg::graph::ImageFilter::filter_flags flags = ref_filter->get_flags();
		auto attr = ref_filter->get_image_attributes();

		AuditBufferType buffer_type = select_buffer_type(flags.color, yuv);

		AuditBuffer<T> src_buf{ buffer_type, src_width, src_height, src_format, zimg::graph::BUFFER_MAX, 0, 0 };
		AuditBuffer<U> ref_buf{ buffer_type, attr.width, attr.height, attr.type, zimg::graph::BUFFER_MAX, 0, 0 };
		AuditBuffer<U> test_buf{ buffer_type, attr.width, attr.height, attr.type, zimg::graph::BUFFER_MAX, 0, 0 };

		src_buf.random_fill(0, src_height, 0, src_width);
		ref_buf.default_fill();
		test_buf.default_fill();

		validate_filter_plane(ref_filter, &src_buf, &ref_buf);
		validate_filter_plane(test_filter, &src_buf, &test_buf);

		EXPECT_GE(snr_buffer(ref_buf, test_buf, attr.width, attr.height, attr.type, !!flags.color), snr_thresh);
	}
};

template <template <typename, typename> class T, class... Args>
void dispatch(zimg::PixelType src_type, zimg::PixelType dst_type, Args&&... args)
{
	if (src_type == zimg::PixelType::BYTE) {
		if (dst_type == zimg::PixelType::BYTE)
			T<uint8_t, uint8_t>{}(std::forward<Args>(args)...);
		else if (dst_type == zimg::PixelType::WORD || dst_type == zimg::PixelType::HALF)
			T<uint8_t, uint16_t>{}(std::forward<Args>(args)...);
		else
			T<uint8_t, float>{}(std::forward<Args>(args)...);
	} else if (src_type == zimg::PixelType::WORD || src_type == zimg::PixelType::HALF) {
		if (dst_type == zimg::PixelType::BYTE)
			T<uint16_t, uint8_t>{}(std::forward<Args>(args)...);
		else if (dst_type == zimg::PixelType::WORD || dst_type == zimg::PixelType::HALF)
			T<uint16_t, uint16_t>{}(std::forward<Args>(args)...);
		else
			T<uint16_t, float>{}(std::forward<Args>(args)...);
	} else if (src_type == zimg::PixelType::FLOAT) {
		if (dst_type == zimg::PixelType::BYTE)
			T<float, uint8_t>{}(std::forward<Args>(args)...);
		else if (dst_type == zimg::PixelType::WORD || dst_type == zimg::PixelType::HALF)
			T<float, uint16_t>{}(std::forward<Args>(args)...);
		else
			T<float, float>{}(std::forward<Args>(args)...);
	}
}

} // namespace


FilterValidator::FilterValidator(const zimg::graph::ImageFilter *test_filter, unsigned src_width, unsigned src_height, const zimg::PixelFormat &src_format) :
	m_test_filter{ test_filter },
	m_ref_filter{},
	m_src_format(src_format),
	m_src_width{ src_width },
	m_src_height{ src_height },
	m_sha1_str{},
	m_snr_thresh{},
	m_yuv{}
{}

FilterValidator &FilterValidator::set_ref_filter(const zimg::graph::ImageFilter *ref_filter, double snr_thresh)
{
	m_ref_filter = ref_filter;
	m_snr_thresh = snr_thresh;
	return *this;
}

FilterValidator &FilterValidator::set_sha1(const char * const sha1_str[3])
{
	m_sha1_str = sha1_str;
	return *this;
}

FilterValidator &FilterValidator::set_yuv(bool yuv)
{
	m_yuv = yuv;
	return *this;
}

void FilterValidator::validate()
{
	auto test_attr = m_test_filter->get_image_attributes();
	auto test_flags = m_test_filter->get_flags();

	zimg::PixelType src_type = m_src_format.type;
	zimg::PixelType dst_type = test_attr.type;

	dispatch<ValidateFilter>(src_type, dst_type, m_test_filter, m_src_width, m_src_height, m_src_format, m_yuv, m_sha1_str);

	if (m_ref_filter) {
		auto ref_flags = m_ref_filter->get_flags();
		auto ref_attr = m_ref_filter->get_image_attributes();

		ASSERT_EQ(ref_flags.color, test_flags.color);
		ASSERT_EQ(ref_attr.width, test_attr.width);
		ASSERT_EQ(ref_attr.height, test_attr.height);
		ASSERT_EQ(ref_attr.type, test_attr.type);

		dispatch<ValidateFilterReference>(src_type, dst_type, m_ref_filter, m_test_filter, m_src_width, m_src_height, m_src_format, m_yuv, m_snr_thresh);
	}
}


bool assert_different_dynamic_type(const zimg::graph::ImageFilter *filter_a, const zimg::graph::ImageFilter *filter_b)
{
	const auto &tid_a = typeid(*filter_a);
	const auto &tid_b = typeid(*filter_b);

	if (tid_a == tid_b) {
		ADD_FAILURE() << "expected different types: " << tid_a.name() << " vs " << tid_b.name();
		return true;
	}
	return false;
}
