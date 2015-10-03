#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include "common/align.h"
#include "common/pixel.h"
#include "graph/zfilter.h"
#include "depth/quantize.h"

#include "gtest/gtest.h"

extern "C" {
  #include "sha1/sha1.h"
}

#include "audit_buffer.h"
#include "filter_validator.h"

namespace {;

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
	const zimg::graph::ZimgImageBufferConst &image_buffer = buf.as_image_buffer();
	SHA1_CTX sha_ctx;

	SHA1Init(&sha_ctx);

	for (unsigned i = 0; i < height; ++i) {
		const unsigned char *ptr = (const unsigned char *)image_buffer.data[p] + (ptrdiff_t)i * image_buffer.stride[p];
		SHA1Update(&sha_ctx, ptr, width * sizeof(T));
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
	auto ref_buf = ref.as_image_buffer();
	auto test_buf = test.as_image_buffer();
	double snr = 0.0;

	for (unsigned p = 0; p < (color ? 3U : 1U); ++p) {
		double signal_plane = 0.0;
		double noise_plane = 0.0;

		for (unsigned i = 0; i < height; ++i) {
			const T *ref_ptr = (const T *)((const char *)ref_buf.data[p] + (ptrdiff_t)i * ref_buf.stride[p]);
			const T *test_ptr = (const T *)((const char *)test_buf.data[p] + (ptrdiff_t)i * test_buf.stride[p]);

			auto snr_pair = snr_line(ref_ptr, test_ptr, width, type);
			signal_plane += snr_pair.first;
			noise_plane += snr_pair.second;
		}

		snr = std::max(snr, signal_plane / noise_plane);
	}

	return 10.0 * std::log10(snr);
}


void validate_flags(const zimg::graph::IZimgFilter *filter)
{
	auto flags = filter->get_flags();

	if (flags.entire_plane && !flags.entire_row)
		FAIL() << "filter must set entire_row if entire_plane is set";
	if (flags.in_place && !flags.same_row)
		FAIL() << "filter must set same_row if in_place is set";

	if (flags.entire_plane && filter->get_max_buffering() != (unsigned)-1)
		FAIL() << "filter must buffer entire image if entire_plane is set";
	if (flags.entire_plane && filter->get_simultaneous_lines() != (unsigned)-1)
		FAIL() << "filter must output entire image if entire_plane is set";
}

void validate_same_row(const zimg::graph::IZimgFilter *filter)
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
void validate_filter_plane(const zimg::graph::IZimgFilter *filter, AuditBuffer<T> *src_buffer, AuditBuffer<U> *dst_buffer)
{
	auto attr = filter->get_image_attributes();

	unsigned step = filter->get_simultaneous_lines();
	zimg::AlignedVector<char> ctx(filter->get_context_size());
	zimg::AlignedVector<char> tmp(filter->get_tmp_size(0, attr.width));

	filter->init_context(ctx.data());

	for (unsigned i = 0; i < attr.height; i += step) {
		filter->process(ctx.data(), src_buffer->as_image_buffer(), dst_buffer->as_image_buffer(), tmp.data(), i, 0, attr.width);

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
void validate_filter_buffered(const zimg::graph::IZimgFilter *filter, unsigned src_width, unsigned src_height, const zimg::PixelFormat &src_format, const AuditBuffer<U> &ref_buffer)
{
	auto flags = filter->get_flags();
	auto attr = filter->get_image_attributes();

	AuditBuffer<T> src_buf{ src_width, src_height, src_format, filter->get_max_buffering(), 0, 0, !!flags.color };
	AuditBuffer<U> dst_buf{ attr.width, attr.height, zimg::default_pixel_format(attr.type), filter->get_simultaneous_lines(), 0, 0, !!flags.color };

	unsigned init = flags.has_state ? 0 : attr.height / 4;
	unsigned vstep = filter->get_simultaneous_lines();
	unsigned step = flags.has_state ? vstep : vstep * 2;

	unsigned left = flags.entire_row ? 0 : attr.width / 4 + 1;
	unsigned right = flags.entire_row ? attr.width : attr.width / 2 - 1;

	zimg::AlignedVector<char> ctx(filter->get_context_size());
	zimg::AlignedVector<char> tmp(filter->get_tmp_size(left, right));

	auto col_range = filter->get_required_col_range(left, right);

	filter->init_context(ctx.data());

	for (unsigned i = init; i < attr.height; i += step) {
		auto row_range = filter->get_required_row_range(i);

		src_buf.default_fill();
		src_buf.random_fill(row_range.first, row_range.second, col_range.first, col_range.second);
		dst_buf.default_fill();

		filter->process(ctx.data(), src_buf.as_image_buffer(), dst_buf.as_image_buffer(), tmp.data(), i, left, right);

		src_buf.assert_guard_bytes();
		dst_buf.assert_guard_bytes();

		for (unsigned ii = i; ii < std::min(i + vstep, attr.height); ++ii) {
			dst_buf.assert_eq(ref_buffer, ii, left, right);
		}
	}
}

template <class T, class U>
void validate_filter_T(const zimg::graph::IZimgFilter *filter, unsigned src_width, unsigned src_height, const zimg::PixelFormat &src_format, const char * const sha1_str[3])
{
	zimg::graph::IZimgFilter::filter_flags flags = filter->get_flags();
	auto attr = filter->get_image_attributes();

	validate_flags(filter);

	if (flags.same_row)
		validate_same_row(filter);

	AuditBuffer<T> src_buf{ src_width, src_height, src_format, (unsigned)-1, 0, 0, !!flags.color };
	AuditBuffer<U> dst_buf{ attr.width, attr.height, zimg::default_pixel_format(attr.type), (unsigned)-1, 0, 0, !!flags.color };

	src_buf.random_fill(0, src_height, 0, src_width);
	dst_buf.default_fill();

	validate_filter_plane(filter, &src_buf, &dst_buf);

	if (sha1_str) {
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

	if (!flags.entire_plane)
		validate_filter_buffered<T, U>(filter, src_width, src_height, src_format, dst_buf);
}

template <class T, class U>
void validate_filter_reference_T(const zimg::graph::IZimgFilter *ref_filter, const zimg::graph::IZimgFilter *test_filter,
                                 unsigned src_width, unsigned src_height, const zimg::PixelFormat &src_format, double snr_thresh)
{
	zimg::graph::IZimgFilter::filter_flags flags = ref_filter->get_flags();
	auto attr = ref_filter->get_image_attributes();

	AuditBuffer<T> src_buf{ src_width, src_height, src_format, (unsigned)-1, 0, 0, !!flags.color };
	AuditBuffer<U> ref_buf{ attr.width, attr.height, zimg::default_pixel_format(attr.type), (unsigned)-1, 0, 0, !!flags.color };
	AuditBuffer<U> test_buf{ attr.width, attr.height, zimg::default_pixel_format(attr.type), (unsigned)-1, 0, 0, !!flags.color };

	src_buf.random_fill(0, src_height, 0, src_width);
	ref_buf.default_fill();
	test_buf.default_fill();

	validate_filter_plane(ref_filter, &src_buf, &ref_buf);
	validate_filter_plane(test_filter, &src_buf, &test_buf);

	EXPECT_GE(snr_buffer(ref_buf, test_buf, attr.width, attr.height, attr.type, !!flags.color), snr_thresh);
}

} // namespace


void validate_filter(const zimg::graph::IZimgFilter *filter, unsigned src_width, unsigned src_height, zimg::PixelType src_type, const char * const sha1_str[3])
{
	validate_filter(filter, src_width, src_height, zimg::default_pixel_format(src_type), sha1_str);
}

void validate_filter(const zimg::graph::IZimgFilter *filter, unsigned src_width, unsigned src_height, const zimg::PixelFormat &src_format, const char * const sha1_str[3])
{
	zimg::PixelType src_type = src_format.type;
	auto attr = filter->get_image_attributes();

	if (src_type == zimg::PixelType::BYTE) {
		if (attr.type == zimg::PixelType::BYTE)
			validate_filter_T<uint8_t, uint8_t>(filter, src_width, src_height, src_format, sha1_str);
		else if (attr.type == zimg::PixelType::WORD || attr.type == zimg::PixelType::HALF)
			validate_filter_T<uint8_t, uint16_t>(filter, src_width, src_height, src_format, sha1_str);
		else
			validate_filter_T<uint8_t, float>(filter, src_width, src_height, src_format, sha1_str);
	} else if (src_type == zimg::PixelType::WORD || src_type == zimg::PixelType::HALF) {
		if (attr.type == zimg::PixelType::BYTE)
			validate_filter_T<uint16_t, uint8_t>(filter, src_width, src_height, src_format, sha1_str);
		else if (attr.type == zimg::PixelType::WORD || attr.type == zimg::PixelType::HALF)
			validate_filter_T<uint16_t, uint16_t>(filter, src_width, src_height, src_format, sha1_str);
		else
			validate_filter_T<uint16_t, float>(filter, src_width, src_height, src_format, sha1_str);
	} else {
		if (attr.type == zimg::PixelType::BYTE)
			validate_filter_T<float, uint8_t>(filter, src_width, src_height, src_format, sha1_str);
		else if (attr.type == zimg::PixelType::WORD || attr.type == zimg::PixelType::HALF)
			validate_filter_T<float, uint16_t>(filter, src_width, src_height, src_format, sha1_str);
		else
			validate_filter_T<float, float>(filter, src_width, src_height, src_format, sha1_str);
	}
}

void validate_filter_reference(const zimg::graph::IZimgFilter *ref_filter, const zimg::graph::IZimgFilter *test_filter,
                               unsigned src_width, unsigned src_height, zimg::PixelType src_type, double snr_thresh)
{
	validate_filter_reference(ref_filter, test_filter, src_width, src_height, zimg::default_pixel_format(src_type), snr_thresh);
}

void validate_filter_reference(const zimg::graph::IZimgFilter *ref_filter, const zimg::graph::IZimgFilter *test_filter,
                               unsigned src_width, unsigned src_height, const zimg::PixelFormat &src_format, double snr_thresh)
{
	auto ref_flags = ref_filter->get_flags();
	auto test_flags = test_filter->get_flags();

	auto ref_attr = ref_filter->get_image_attributes();
	auto test_attr = test_filter->get_image_attributes();

	zimg::PixelType src_type = src_format.type;
	zimg::PixelType dst_type = ref_attr.type;

	ASSERT_EQ(ref_flags.color, test_flags.color);
	ASSERT_EQ(ref_attr.width, test_attr.width);
	ASSERT_EQ(ref_attr.height, test_attr.height);
	ASSERT_EQ(ref_attr.type, test_attr.type);

	if (src_type == zimg::PixelType::BYTE) {
		if (dst_type == zimg::PixelType::BYTE)
			validate_filter_reference_T<uint8_t, uint8_t>(ref_filter, test_filter, src_width, src_height, src_format, snr_thresh);
		else if (dst_type == zimg::PixelType::WORD || dst_type == zimg::PixelType::HALF)
			validate_filter_reference_T<uint8_t, uint16_t>(ref_filter, test_filter, src_width, src_height, src_format, snr_thresh);
		else
			validate_filter_reference_T<uint8_t, float>(ref_filter, test_filter, src_width, src_height, src_format, snr_thresh);
	} else if (src_type == zimg::PixelType::WORD || src_type == zimg::PixelType::HALF) {
		if (dst_type == zimg::PixelType::BYTE)
			validate_filter_reference_T<uint16_t, uint8_t>(ref_filter, test_filter, src_width, src_height, src_format, snr_thresh);
		else if (dst_type == zimg::PixelType::WORD || dst_type == zimg::PixelType::HALF)
			validate_filter_reference_T<uint16_t, uint16_t>(ref_filter, test_filter, src_width, src_height, src_format, snr_thresh);
		else
			validate_filter_reference_T<uint16_t, float>(ref_filter, test_filter, src_width, src_height, src_format, snr_thresh);
	} else {
		if (dst_type == zimg::PixelType::BYTE)
			validate_filter_reference_T<float, uint8_t>(ref_filter, test_filter, src_width, src_height, src_format, snr_thresh);
		else if (dst_type == zimg::PixelType::WORD || dst_type == zimg::PixelType::HALF)
			validate_filter_reference_T<float, uint16_t>(ref_filter, test_filter, src_width, src_height, src_format, snr_thresh);
		else
			validate_filter_reference_T<float, float>(ref_filter, test_filter, src_width, src_height, src_format, snr_thresh);
	}
}
