#include <algorithm>
#include <cstdint>
#include <cstring>

#include "common/align.h"
#include "common/except.h"
#include "common/linebuffer.h"
#include "common/pixel.h"
#include "graph/filtergraph.h"
#include "graph/image_filter.h"

#include "gtest/gtest.h"
#include "audit_buffer.h"
#include "mock_filter.h"

namespace {;

template <class T>
class AuditImage : public AuditBuffer<T> {
	unsigned m_width;
	unsigned m_height;
public:
	AuditImage(unsigned width, unsigned height, zimg::PixelType type, unsigned subsample_w, unsigned subsample_h, bool color) :
		AuditBuffer<T>(width, height, zimg::default_pixel_format(type), -1, subsample_w, subsample_h, color),
		m_width{ width },
		m_height{ height }
	{
	}

	void validate()
	{
		for (unsigned i = 0; i < m_height; ++i) {
			ASSERT_FALSE(AuditBuffer<T>::detect_write(i, 0, m_width)) << "unexpected write at line: " << i;
		}
	}
};

}


TEST(FilterGraphTest, test_noop)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::BYTE;

	for (unsigned x = 0; x < 2; ++x) {
		SCOPED_TRACE(!!x);

		zimg::graph::FilterGraph graph{ w, h, type, 0, 0, !!x };
		graph.complete();

		AuditImage<uint8_t> src_image{ w, h, type, 0, 0, !!x };
		AuditImage<uint8_t> dst_image{ w, h, type, 0, 0, !!x };
		zimg::AlignedVector<char> tmp(graph.get_tmp_size());

		src_image.default_fill();
		graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);

		SCOPED_TRACE("validating src");
		src_image.validate();
		SCOPED_TRACE("validating dst");
		dst_image.validate();
	}
}

TEST(FilterGraphTest, test_noop_subsampling)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::BYTE;

	for (unsigned sw = 0; sw < 3; ++sw) {
		for (unsigned sh = 0; sh < 3; ++sh) {
			SCOPED_TRACE(sw);
			SCOPED_TRACE(sh);

			zimg::graph::FilterGraph graph{ w, h, type, sw, sh, true };
			graph.complete();

			AuditImage<uint8_t> src_image{ w, h, type, sw, sh, true };
			AuditImage<uint8_t> dst_image{ w, h, type, sw, sh, true };
			zimg::AlignedVector<char> tmp(graph.get_tmp_size());

			src_image.default_fill();
			graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);

			SCOPED_TRACE("validating src");
			src_image.validate();
			SCOPED_TRACE("validating dst");
			dst_image.validate();
		}
	}
}

TEST(FilterGraphTest, test_basic)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::WORD;

	const uint8_t test_byte1 = 0xCD;
	const uint8_t test_byte2 = 0xDD;
	const uint8_t test_byte3 = 0xDC;

	for (unsigned x = 0; x < 2; ++x) {
		SCOPED_TRACE(!!x);

		zimg::graph::ImageFilter::filter_flags flags1{};
		flags1.has_state = true;
		flags1.entire_row = true;
		flags1.entire_plane = true;
		flags1.color = !!x;

		zimg::graph::ImageFilter::filter_flags flags2{};
		flags2.has_state = true;
		flags2.entire_row = false;
		flags2.entire_plane = false;
		flags2.color = !!x;

		std::unique_ptr<SplatFilter<uint16_t>> filter1_uptr{ new SplatFilter<uint16_t>{ w, h, type, flags1 } };
		std::unique_ptr<SplatFilter<uint16_t>> filter2_uptr{ new SplatFilter<uint16_t>{ w, h, type, flags2 } };
		SplatFilter<uint16_t> *filter1 = filter1_uptr.get();
		SplatFilter<uint16_t> *filter2 = filter2_uptr.get();

		filter1->set_input_val(test_byte1);
		filter1->set_output_val(test_byte2);

		filter2->set_input_val(test_byte2);
		filter2->set_output_val(test_byte3);

		zimg::graph::FilterGraph graph{ w, h, type, 0, 0, !!x };

		graph.attach_filter(filter1);
		filter1_uptr.release();
		graph.attach_filter(filter2);
		filter2_uptr.release();
		graph.complete();

		AuditImage<uint16_t> src_image{ w, h, type, 0, 0, !!x };
		AuditImage<uint16_t> dst_image{ w, h, type, 0, 0, !!x };
		zimg::AlignedVector<char> tmp(graph.get_tmp_size());

		src_image.set_fill_val(test_byte1);
		src_image.default_fill();
		graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);
		dst_image.set_fill_val(test_byte3);

		ASSERT_EQ(1U, filter1->get_total_calls());
		ASSERT_EQ(h, filter2->get_total_calls());

		SCOPED_TRACE("validating src");
		src_image.validate();
		SCOPED_TRACE("validating dst");
		dst_image.validate();
	}
}

TEST(FilterGraphTest, test_skip_plane)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::FLOAT;

	const uint8_t test_byte1 = 0xCD;
	const uint8_t test_byte2 = 0xDD;
	const uint8_t test_byte3 = 0xDC;

	for (unsigned x = 0; x < 2; ++x) {
		SCOPED_TRACE(!!x);

		zimg::graph::ImageFilter::filter_flags flags1{};
		flags1.has_state = true;
		flags1.entire_row = true;
		flags1.color = true;

		zimg::graph::ImageFilter::filter_flags flags2{};
		flags2.has_state = false;
		flags2.entire_row = true;
		flags2.color = false;

		std::unique_ptr<SplatFilter<float>> filter1_uptr{ new SplatFilter<float>{ w, h, type, flags1} };
		std::unique_ptr<SplatFilter<float>> filter2_uptr{ new SplatFilter<float>{ w, h, type, flags2} };
		SplatFilter<float> *filter1 = filter1_uptr.get();
		SplatFilter<float> *filter2 = filter2_uptr.get();

		filter1->set_input_val(test_byte1);
		filter1->set_output_val(test_byte2);

		filter2->set_input_val(test_byte2);
		filter2->set_output_val(test_byte3);

		zimg::graph::FilterGraph graph{ w, h, type, 0, 0, true };

		graph.attach_filter(filter1);
		filter1_uptr.release();

		if (x)
			graph.attach_filter(filter2);
		else
			graph.attach_filter_uv(filter2);
		filter2_uptr.release();

		graph.complete();

		AuditImage<float> src_image{ w, h, type, 0, 0, true };
		AuditImage<float> dst_image{ w, h, type, 0, 0, true };
		zimg::AlignedVector<char> tmp(graph.get_tmp_size());

		src_image.set_fill_val(test_byte1);
		src_image.default_fill();

		graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);

		if (x) {
			dst_image.set_fill_val(test_byte3, 0);
			dst_image.set_fill_val(test_byte2, 1);
			dst_image.set_fill_val(test_byte2, 2);
		} else {
			dst_image.set_fill_val(test_byte2, 0);
			dst_image.set_fill_val(test_byte3, 1);
			dst_image.set_fill_val(test_byte3, 2);
		}

		ASSERT_EQ(h, filter1->get_total_calls());
		ASSERT_EQ(h * (x ? 1 : 2), filter2->get_total_calls());

		SCOPED_TRACE("validating src");
		src_image.validate();
		SCOPED_TRACE("validating dst");
		dst_image.validate();
	}
}

TEST(FilterGraphTest, test_color_to_grey)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::BYTE;

	const uint8_t test_byte1 = 0xCD;
	const uint8_t test_byte2 = 0xDC;

	zimg::graph::ImageFilter::filter_flags flags{};
	flags.has_state = true;
	flags.entire_row = true;
	flags.color = true;

	std::unique_ptr<SplatFilter<uint8_t>> filter_uptr{ new SplatFilter<uint8_t>{ w, h, type, flags} };
	SplatFilter<uint8_t> *filter = filter_uptr.get();

	filter->set_input_val(test_byte1);
	filter->set_output_val(test_byte2);

	zimg::graph::FilterGraph graph{ w, h, type, 0, 0, true };

	graph.attach_filter(filter);
	filter_uptr.release();

	graph.color_to_grey();
	graph.complete();

	AuditImage<uint8_t> src_image{ w, h, type, 0, 0, true };
	AuditImage<uint8_t> dst_image{ w, h, type, 0, 0, false };
	zimg::AlignedVector<char> tmp(graph.get_tmp_size());

	src_image.set_fill_val(test_byte1);
	src_image.default_fill();

	graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);

	dst_image.set_fill_val(test_byte2);

	ASSERT_EQ(h, filter->get_total_calls());

	SCOPED_TRACE("validating src");
	src_image.validate();
	SCOPED_TRACE("validating dst");
	dst_image.validate();
}

TEST(FilterGraphTest, test_grey_to_color_rgb)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::BYTE;

	const uint8_t test_byte1 = 0xCD;
	const uint8_t test_byte2 = 0xDD;
	const uint8_t test_byte3 = 0xDC;

	zimg::graph::ImageFilter::filter_flags flags1{};
	flags1.has_state = true;
	flags1.entire_row = true;
	flags1.color = false;

	zimg::graph::ImageFilter::filter_flags flags2{};
	flags2.has_state = true;
	flags2.entire_row = true;
	flags2.color = true;

	std::unique_ptr<SplatFilter<uint8_t>> filter1_uptr{ new SplatFilter<uint8_t>{ w, h, type, flags1 } };
	std::unique_ptr<SplatFilter<uint8_t>> filter2_uptr{ new SplatFilter<uint8_t>{ w, h, type, flags2 } };
	SplatFilter<uint8_t> *filter1 = filter1_uptr.get();
	SplatFilter<uint8_t> *filter2 = filter2_uptr.get();

	filter1->set_input_val(test_byte1);
	filter1->set_output_val(test_byte2);

	filter2->set_input_val(test_byte2);
	filter2->set_output_val(test_byte3);

	zimg::graph::FilterGraph graph{ w, h, type, 0, 0, false };

	graph.attach_filter(filter1);
	filter1_uptr.release();

	graph.grey_to_color(false, 0, 0, 8);

	graph.attach_filter(filter2);
	filter2_uptr.release();

	graph.complete();

	AuditImage<uint8_t> src_image{ w, h, type, 0, 0, false };
	AuditImage<uint8_t> dst_image{ w, h, type, 0, 0, true };
	zimg::AlignedVector<char> tmp(graph.get_tmp_size());

	src_image.set_fill_val(test_byte1);
	src_image.default_fill();

	graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);

	dst_image.set_fill_val(test_byte3);

	ASSERT_EQ(h, filter1->get_total_calls());
	ASSERT_EQ(h, filter2->get_total_calls());

	SCOPED_TRACE("validating src");
	src_image.validate();
	SCOPED_TRACE("validating dst");
	dst_image.validate();
}

TEST(FilterGraphTest, test_grey_to_color_yuv)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::BYTE;

	const uint8_t test_byte1 = 0xCD;
	const uint8_t test_byte2 = 0xDD;
	const uint8_t test_byte2_uv = 128;

	zimg::graph::ImageFilter::filter_flags flags{};
	flags.has_state = true;
	flags.entire_row = true;
	flags.color = false;

	std::unique_ptr<SplatFilter<uint8_t>> filter_uptr{ new SplatFilter<uint8_t>{ w, h, type, flags } };
	SplatFilter<uint8_t> *filter = filter_uptr.get();

	filter->set_input_val(test_byte1);
	filter->set_output_val(test_byte2);

	zimg::graph::FilterGraph graph{ w, h, type, 0, 0, false };

	graph.attach_filter(filter);
	filter_uptr.release();

	graph.grey_to_color(true, 1, 1, 8);
	graph.complete();

	AuditImage<uint8_t> src_image{ w, h, type, 0, 0, false };
	AuditImage<uint8_t> dst_image{ w, h, type, 1, 1, true };
	zimg::AlignedVector<char> tmp(graph.get_tmp_size());

	src_image.set_fill_val(test_byte1);
	src_image.default_fill();

	graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);

	dst_image.set_fill_val(test_byte2, 0);
	dst_image.set_fill_val(test_byte2_uv, 1);
	dst_image.set_fill_val(test_byte2_uv, 2);

	ASSERT_EQ(h, filter->get_total_calls());

	SCOPED_TRACE("validating src");
	src_image.validate();
	SCOPED_TRACE("validating dst");
	dst_image.validate();
}

TEST(FilterGraphTest, test_support)
{
	const unsigned w = 1024;
	const unsigned h = 576;
	const zimg::PixelType type = zimg::PixelType::HALF;

	const uint8_t test_byte1 = 0xCD;
	const uint8_t test_byte2 = 0xDD;
	const uint8_t test_byte3 = 0xDC;

	for (unsigned x = 0; x < 2; ++x) {
		SCOPED_TRACE(!!x);

		std::unique_ptr<SplatFilter<uint16_t>> filter1_uptr{ new SplatFilter<uint16_t>{ w, h, type} };
		std::unique_ptr<SplatFilter<uint16_t>> filter2_uptr{ new SplatFilter<uint16_t>{ w, h, type} };
		SplatFilter<uint16_t> *filter1 = filter1_uptr.get();
		SplatFilter<uint16_t> *filter2 = filter2_uptr.get();

		filter1->set_input_val(test_byte1);
		filter1->set_output_val(test_byte2);

		filter2->set_input_val(test_byte2);
		filter2->set_output_val(test_byte3);

		if (x) {
			filter1->set_horizontal_support(5);
			filter1->set_simultaneous_lines(5);

			filter2->set_horizontal_support(3);
			filter2->set_simultaneous_lines(3);
		} else {
			filter1->set_horizontal_support(3);
			filter1->set_simultaneous_lines(3);

			filter2->set_horizontal_support(5);
			filter2->set_simultaneous_lines(5);
		}

		zimg::graph::FilterGraph graph{ w, h, type, 0, 0, false };

		graph.attach_filter(filter1);
		filter1_uptr.release();
		graph.attach_filter(filter2);
		filter2_uptr.release();
		graph.complete();

		AuditImage<uint16_t> src_image{ w, h, type, 0, 0, false };
		AuditImage<uint16_t> dst_image{ w, h, type, 0, 0, false };
		zimg::AlignedVector<char> tmp(graph.get_tmp_size());

		src_image.set_fill_val(test_byte1);
		src_image.default_fill();

		if (x) {
			EXPECT_EQ(8U, graph.get_input_buffering());
			EXPECT_EQ(4U, graph.get_output_buffering());
		} else {
			EXPECT_EQ(4U, graph.get_input_buffering());
			EXPECT_EQ(8U, graph.get_output_buffering());
		}

		graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), nullptr, nullptr);
		dst_image.set_fill_val(test_byte3);

		SCOPED_TRACE("validating src");
		src_image.validate();
		SCOPED_TRACE("validating dst");
		dst_image.validate();

		if (x) {
			EXPECT_EQ(2 * 116U, filter1->get_total_calls());
			EXPECT_EQ(2 * 192U, filter2->get_total_calls());
		} else {
			EXPECT_EQ(2 * 192U, filter1->get_total_calls());
			EXPECT_EQ(2 * 116U, filter2->get_total_calls());
		}
	}
}

TEST(FilterGraphTest, test_callback)
{
	static const unsigned w = 1024;
	static const unsigned h = 576;
	const zimg::PixelType type = zimg::PixelType::BYTE;

	const uint8_t test_byte1 = 0xCD;
	const uint8_t test_byte2 = 0xFF;
	const uint8_t test_byte3 = 0xDC;

	struct callback_data {
		zimg::graph::ImageBuffer buffer;
		unsigned subsample_w;
		unsigned subsample_h;
		unsigned call_count;
		uint8_t byte_val;
	};

	auto cb = [](void *ptr, unsigned i, unsigned left, unsigned right) -> int
	{
		callback_data *xptr = reinterpret_cast<callback_data *>(ptr);

		EXPECT_LT(i, h);
		EXPECT_EQ(0U, i % (1 << xptr->subsample_h));
		EXPECT_LT(left, right);
		EXPECT_LE(right, w);

		for (unsigned ii = i; ii < i + (1 << xptr->subsample_h); ++ii) {
			zimg::LineBuffer<uint8_t> buf{ xptr->buffer, 0 };

			std::fill(buf[ii] + left, buf[ii] + right, xptr->byte_val);
		}
		for (unsigned p = 1; p < 3; ++p) {
			zimg::LineBuffer<uint8_t> chroma_buf{ xptr->buffer, p };
			unsigned i_chroma = i >> xptr->subsample_h;
			unsigned left_chroma = (left % 2 ? left - 1 : left) >> xptr->subsample_w;
			unsigned right_chroma = (right % 2 ? right + 1 : right) >> xptr->subsample_w;

			std::fill(chroma_buf[i_chroma] + left_chroma, chroma_buf[i_chroma] + right_chroma, xptr->byte_val);
		}

		++xptr->call_count;
		return HasFatalFailure();
	};

	for (unsigned sw = 0; sw < 3; ++sw) {
		for (unsigned sh = 0; sh < 3; ++sh) {
			for (unsigned x = 0; x < 2; ++x) {
				SCOPED_TRACE(sw);
				SCOPED_TRACE(sh);
				SCOPED_TRACE(!!x);

				zimg::graph::ImageFilter::filter_flags flags{};
				flags.entire_row = !!x;
				flags.color = false;

				std::unique_ptr<SplatFilter<uint8_t>> filter1_uptr{ new SplatFilter<uint8_t>{ w, h, type, flags } };
				std::unique_ptr<SplatFilter<uint8_t>> filter2_uptr{ new SplatFilter<uint8_t>{ w >> sw, h >> sh, type, flags } };
				SplatFilter<uint8_t> *filter1 = filter1_uptr.get();
				SplatFilter<uint8_t> *filter2 = filter2_uptr.get();

				filter1->set_input_val(test_byte1);
				filter1->set_output_val(test_byte2);

				filter2->set_input_val(test_byte1);
				filter2->set_output_val(test_byte2);

				zimg::graph::FilterGraph graph{ w, h, type, sw, sh, true };
				graph.attach_filter(filter1);
				filter1_uptr.release();
				graph.attach_filter_uv(filter2);
				filter2_uptr.release();
				graph.complete();

				AuditImage<uint8_t> src_image{ w, h, type, sw, sh, true };
				AuditImage<uint8_t> tmp_image{ w, h, type, sw, sh, true };
				AuditImage<uint8_t> dst_image{ w, h, type, sw, sh, true };
				zimg::AlignedVector<char> tmp(graph.get_tmp_size());

				callback_data cb1_data = { src_image.as_image_buffer(), sw, sh, 0, test_byte1 };
				callback_data cb2_data = { dst_image.as_image_buffer(), sw, sh, 0, test_byte3 };

				src_image.set_fill_val(test_byte1);
				tmp_image.set_fill_val(test_byte2);
				dst_image.set_fill_val(test_byte3);

				graph.process(src_image.as_image_buffer(), tmp_image.as_image_buffer(), tmp.data(), { cb, &cb1_data }, { cb, &cb2_data });

				SCOPED_TRACE("validating src");
				src_image.validate();
				SCOPED_TRACE("validating tmp");
				tmp_image.validate();
				SCOPED_TRACE("validating dst");
				dst_image.validate();

				EXPECT_EQ((h >> sh) * (x ? 1 : 2), cb1_data.call_count);
				EXPECT_EQ((h >> sh) * (x ? 1 : 2), cb2_data.call_count);
			}
		}
	}
}

TEST(FilterGraphTest, test_callback_failed)
{
	const unsigned w = 640;
	const unsigned h = 480;
	zimg::PixelType type = zimg::PixelType::BYTE;

	auto cb = [](void *, unsigned i, unsigned left, unsigned right) -> int
	{
		return 1;
	};

	zimg::graph::FilterGraph graph{ w, h, type, 0, 0, false };
	graph.complete();

	AuditImage<uint8_t> src_image{ w, h, type, 0, 0, false };
	AuditImage<uint8_t> dst_image{ w, h, type, 0, 0, false };
	zimg::AlignedVector<char> tmp(graph.get_tmp_size());

	src_image.set_fill_val(255);
	dst_image.set_fill_val(0);

	src_image.default_fill();
	dst_image.default_fill();

	ASSERT_THROW(graph.process(src_image.as_image_buffer(), dst_image.as_image_buffer(), tmp.data(), { cb, nullptr }, nullptr), zimg::error::UserCallbackFailed);

	SCOPED_TRACE("validating dst");
	dst_image.validate();
}
