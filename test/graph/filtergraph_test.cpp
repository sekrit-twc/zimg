#include <algorithm>
#include <cstdint>
#include "common/alloc.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "graph/basic_filter.h"
#include "graph/filtergraph.h"
#include "graph/image_filter.h"

#include "gtest/gtest.h"
#include "audit_buffer.h"
#include "mock_filter.h"

namespace {

using zimg::graph::node_id;
using zimg::graph::plane_mask;
using zimg::graph::id_map;
using zimg::graph::invalid_id;

plane_mask enabled_planes(bool color)
{
	plane_mask mask{};
	mask[zimg::graph::PLANE_Y] = true;
	mask[zimg::graph::PLANE_U] = color;
	mask[zimg::graph::PLANE_V] = color;
	return mask;
}

id_map id_to_map(node_id id, bool color)
{
	id_map map = zimg::graph::null_ids;
	map[zimg::graph::PLANE_Y] = id;
	map[zimg::graph::PLANE_U] = color ? id : invalid_id;
	map[zimg::graph::PLANE_V] = color ? id : invalid_id;
	return map;
}


template <class T>
class AuditImage : public AuditBuffer<T> {
	unsigned m_width;
	unsigned m_height;
public:
	AuditImage(AuditBufferType buffer_type, unsigned width, unsigned height, zimg::PixelType type, unsigned subsample_w, unsigned subsample_h) :
		AuditBuffer<T>(buffer_type, width, height, type, zimg::graph::BUFFER_MAX, subsample_w, subsample_h),
		m_width{ width },
		m_height{ height }
	{}

	void validate()
	{
		for (unsigned i = 0; i < m_height; ++i) {
			ASSERT_FALSE(AuditBuffer<T>::detect_write(i, 0, m_width)) << "unexpected write at line: " << i;
		}
	}
};

} // namespace


TEST(FilterGraphTest, test_noop)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::BYTE;

	for (unsigned x = 0; x < 2; ++x) {
		SCOPED_TRACE(!!x);

		bool color = !!x;
		AuditBufferType buffer_type = color ? AuditBufferType::COLOR_RGB : AuditBufferType::PLANE;

		zimg::graph::FilterGraph graph;
		node_id id = graph.add_source({ w, h, type }, 0, 0, enabled_planes(color));
		graph.set_output(id_to_map(id, color));

		AuditImage<uint8_t> src_image{ buffer_type, w, h, type, 0, 0 };
		AuditImage<uint8_t> dst_image{ buffer_type, w, h, type, 0, 0 };
		zimg::AlignedVector<char> tmp(graph.get_tmp_size());

		src_image.default_fill();
		graph.process(src_image.as_read_buffer(), dst_image.as_write_buffer(), tmp.data(), nullptr, nullptr);

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

			zimg::graph::FilterGraph graph;
			node_id id = graph.add_source({ w, h, type }, sw, sh, enabled_planes(true));
			graph.set_output(id_to_map(id, true));

			AuditImage<uint8_t> src_image{ AuditBufferType::COLOR_YUV, w, h, type, sw, sh };
			AuditImage<uint8_t> dst_image{ AuditBufferType::COLOR_YUV, w, h, type, sw, sh };
			zimg::AlignedVector<char> tmp(graph.get_tmp_size());

			src_image.default_fill();
			graph.process(src_image.as_read_buffer(), dst_image.as_write_buffer(), tmp.data(), nullptr, nullptr);

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

		bool color = !!x;
		AuditBufferType buffer_type = color ? AuditBufferType::COLOR_RGB : AuditBufferType::PLANE;

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

		auto filter1 = std::make_shared<SplatFilter<uint16_t>>(w, h, type, flags1);
		auto filter2 = std::make_shared<SplatFilter<uint16_t>>(w, h, type, flags2);

		filter1->set_input_val(test_byte1);
		filter1->set_output_val(test_byte2);

		filter2->set_input_val(test_byte2);
		filter2->set_output_val(test_byte3);

		zimg::graph::FilterGraph graph;
		node_id id = graph.add_source({ w, h, type }, 0, 0, enabled_planes(color));

		id = graph.attach_filter(filter1, id_to_map(id, color), enabled_planes(color));
		id = graph.attach_filter(filter2, id_to_map(id, color), enabled_planes(color));
		graph.set_output(id_to_map(id, color));

		AuditImage<uint16_t> src_image{ buffer_type, w, h, type, 0, 0 };
		AuditImage<uint16_t> dst_image{ buffer_type, w, h, type, 0, 0 };
		zimg::AlignedVector<char> tmp(graph.get_tmp_size());

		src_image.set_fill_val(test_byte1);
		src_image.default_fill();
		graph.process(src_image.as_read_buffer(), dst_image.as_write_buffer(), tmp.data(), nullptr, nullptr);
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

		auto filter1 = std::make_shared<SplatFilter<float>>(w, h, type, flags1);
		auto filter2 = std::make_shared<SplatFilter<float>>(w, h, type, flags2);

		filter1->set_input_val(test_byte1);
		filter1->set_output_val(test_byte2);

		filter2->set_input_val(test_byte2);
		filter2->set_output_val(test_byte3);

		zimg::graph::FilterGraph graph;
		node_id id_y = graph.add_source({ w, h, type }, 0, 0, enabled_planes(true));
		id_y = graph.attach_filter(filter1, id_to_map(id_y, true), enabled_planes(true));

		node_id id_u = id_y;
		node_id id_v = id_y;

		if (x) {
			id_y = graph.attach_filter(filter2, id_to_map(id_y, false), enabled_planes(false));
		} else {
			id_u = graph.attach_filter(filter2, { invalid_id, id_u, invalid_id, invalid_id }, { false, true, false, false });
			id_v = graph.attach_filter(filter2, { invalid_id, invalid_id, id_v, invalid_id }, { false, false, true, false });
		}

		graph.set_output({ id_y, id_u, id_v, invalid_id });

		AuditImage<float> src_image{ AuditBufferType::COLOR_YUV, w, h, type, 0, 0 };
		AuditImage<float> dst_image{ AuditBufferType::COLOR_YUV, w, h, type, 0, 0 };
		zimg::AlignedVector<char> tmp(graph.get_tmp_size());

		src_image.set_fill_val(test_byte1);
		src_image.default_fill();

		graph.process(src_image.as_read_buffer(), dst_image.as_write_buffer(), tmp.data(), nullptr, nullptr);

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

	auto filter = std::make_shared<SplatFilter<uint8_t>>(w, h, type, flags);
	filter->set_input_val(test_byte1);
	filter->set_output_val(test_byte2);

	zimg::graph::FilterGraph graph;
	node_id id = graph.add_source({ w, h, type }, 0, 0, enabled_planes(true));

	id = graph.attach_filter(filter, id_to_map(id, true), enabled_planes(true));
	graph.set_output(id_to_map(id, false));

	AuditImage<uint8_t> src_image{ AuditBufferType::COLOR_YUV, w, h, type, 0, 0 };
	AuditImage<uint8_t> dst_image{ AuditBufferType::PLANE, w, h, type, 0, 0 };
	zimg::AlignedVector<char> tmp(graph.get_tmp_size());

	src_image.set_fill_val(test_byte1);
	src_image.default_fill();

	graph.process(src_image.as_read_buffer(), dst_image.as_write_buffer(), tmp.data(), nullptr, nullptr);

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

	auto filter1 = std::make_shared<SplatFilter<uint8_t>>(w, h, type, flags1);
	auto filter2 = std::make_shared<SplatFilter<uint8_t>>(w, h, type, flags2);

	filter1->set_input_val(test_byte1);
	filter1->set_output_val(test_byte2);

	filter2->set_input_val(test_byte2);
	filter2->set_output_val(test_byte3);

	zimg::graph::FilterGraph graph;
	node_id id = graph.add_source({ w, h, type }, 0, 0, enabled_planes(false));

	id = graph.attach_filter(filter1, id_to_map(id, false), enabled_planes(false));

	auto rgbextend = ztd::make_unique<zimg::graph::RGBExtendFilter>(w, h, type);
	id = graph.attach_filter(std::move(rgbextend), id_to_map(id, false), enabled_planes(true));

	id = graph.attach_filter(filter2, id_to_map(id, true), enabled_planes(true));
	graph.set_output(id_to_map(id, true));

	AuditImage<uint8_t> src_image{ AuditBufferType::PLANE, w, h, type, 0, 0 };
	AuditImage<uint8_t> dst_image{ AuditBufferType::COLOR_RGB, w, h, type, 0, 0 };
	zimg::AlignedVector<char> tmp(graph.get_tmp_size());

	src_image.set_fill_val(test_byte1);
	src_image.default_fill();

	graph.process(src_image.as_read_buffer(), dst_image.as_write_buffer(), tmp.data(), nullptr, nullptr);

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

	auto filter = std::make_shared<SplatFilter<uint8_t>>(w, h, type, flags);
	filter->set_input_val(test_byte1);
	filter->set_output_val(test_byte2);

	zimg::graph::FilterGraph graph;
	node_id id_y = graph.add_source({ w, h, type }, 0, 0, enabled_planes(false));
	node_id id_u = id_y;
	node_id id_v = id_y;

	id_y = graph.attach_filter(filter, id_to_map(id_y, false), enabled_planes(false));

	zimg::graph::ValueInitializeFilter::value_type init_val{};
	init_val.b = test_byte2_uv;
	auto chroma_init = std::make_shared<zimg::graph::ValueInitializeFilter>(w >> 1, h >> 1, type, init_val);
	id_u = graph.attach_filter(chroma_init, zimg::graph::null_ids, { false, true, false, false });
	id_v = graph.attach_filter(chroma_init, zimg::graph::null_ids, { false, false, true, false });

	graph.set_output({ id_y, id_u, id_v, invalid_id });

	AuditImage<uint8_t> src_image{ AuditBufferType::PLANE, w, h, type, 0, 0 };
	AuditImage<uint8_t> dst_image{ AuditBufferType::COLOR_YUV, w, h, type, 1, 1 };
	zimg::AlignedVector<char> tmp(graph.get_tmp_size());

	src_image.set_fill_val(test_byte1);
	src_image.default_fill();

	graph.process(src_image.as_read_buffer(), dst_image.as_write_buffer(), tmp.data(), nullptr, nullptr);

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

		auto filter1 = std::make_shared<SplatFilter<uint16_t>>(w, h, type);
		auto filter2 = std::make_shared<SplatFilter<uint16_t>>(w, h, type);

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

		zimg::graph::FilterGraph graph;
		node_id id = graph.add_source({ w, h, type }, 0, 0, enabled_planes(false));

		id = graph.attach_filter(filter1, id_to_map(id, false), enabled_planes(false));
		id = graph.attach_filter(filter2, id_to_map(id, false), enabled_planes(false));
		graph.set_output(id_to_map(id, false));

		graph.set_tile_width(512);

		AuditImage<uint16_t> src_image{ AuditBufferType::PLANE, w, h, type, 0, 0 };
		AuditImage<uint16_t> dst_image{ AuditBufferType::PLANE, w, h, type, 0, 0 };
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

		graph.process(src_image.as_read_buffer(), dst_image.as_write_buffer(), tmp.data(), nullptr, nullptr);
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
		zimg::graph::ColorImageBuffer<void> buffer;
		unsigned subsample_w;
		unsigned subsample_h;
		unsigned call_count;
		uint8_t byte_val;
	};

	auto cb = [](void *ptr, unsigned i, unsigned left, unsigned right) -> int
	{
		callback_data *xptr = static_cast<callback_data *>(ptr);

		EXPECT_LT(i, h);
		EXPECT_EQ(0U, i % (1 << xptr->subsample_h));
		EXPECT_LT(left, right);
		EXPECT_LE(right, w);

		if (HasFailure())
			return 1;

		for (unsigned ii = i; ii < i + (1 << xptr->subsample_h); ++ii) {
			const auto &buf = zimg::graph::static_buffer_cast<uint8_t>(xptr->buffer[0]);

			std::fill(buf[ii] + left, buf[ii] + right, xptr->byte_val);
		}
		for (unsigned p = 1; p < 3; ++p) {
			const auto &chroma_buf = zimg::graph::static_buffer_cast<uint8_t>(xptr->buffer[p]);
			unsigned i_chroma = i >> xptr->subsample_h;
			unsigned left_chroma = (left % 2 ? left - 1 : left) >> xptr->subsample_w;
			unsigned right_chroma = (right % 2 ? right + 1 : right) >> xptr->subsample_w;

			std::fill(chroma_buf[i_chroma] + left_chroma, chroma_buf[i_chroma] + right_chroma, xptr->byte_val);
		}

		++xptr->call_count;
		return 0;
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

				auto filter1 = std::make_shared<SplatFilter<uint8_t>>(w, h, type, flags);
				auto filter2 = std::make_shared<SplatFilter<uint8_t>>(w >> sw, h >> sh, type, flags);

				filter1->set_input_val(test_byte1);
				filter1->set_output_val(test_byte2);

				filter2->set_input_val(test_byte1);
				filter2->set_output_val(test_byte2);

				zimg::graph::FilterGraph graph;
				node_id id_y = graph.add_source({ w, h, type }, sw, sh, enabled_planes(true));
				node_id id_u = id_y;
				node_id id_v = id_y;

				id_y = graph.attach_filter(filter1, id_to_map(id_y, false), enabled_planes(false));
				id_u = graph.attach_filter(filter2, { invalid_id, id_u, invalid_id, invalid_id }, { false, true, false, false });
				id_v = graph.attach_filter(filter2, { invalid_id, invalid_id, id_v, invalid_id }, { false, false, true, false });
				graph.set_output({ id_y, id_u, id_v, invalid_id });

				graph.set_tile_width(512);

				AuditImage<uint8_t> src_image{ AuditBufferType::COLOR_RGB, w, h, type, sw, sh };
				AuditImage<uint8_t> tmp_image{ AuditBufferType::COLOR_RGB, w, h, type, sw, sh };
				AuditImage<uint8_t> dst_image{ AuditBufferType::COLOR_RGB, w, h, type, sw, sh };
				zimg::AlignedVector<char> tmp(graph.get_tmp_size());

				callback_data cb1_data = { src_image.as_write_buffer(), sw, sh, 0, test_byte1 };
				callback_data cb2_data = { dst_image.as_write_buffer(), sw, sh, 0, test_byte3 };

				src_image.set_fill_val(test_byte1);
				tmp_image.set_fill_val(test_byte2);
				dst_image.set_fill_val(test_byte3);

				graph.process(src_image.as_read_buffer(), tmp_image.as_write_buffer(), tmp.data(), { cb, &cb1_data }, { cb, &cb2_data });

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

	zimg::graph::FilterGraph graph;
	node_id id = graph.add_source({ w, h, type }, 0, 0, enabled_planes(false));
	graph.set_output(id_to_map(id, false));

	AuditImage<uint8_t> src_image{ AuditBufferType::PLANE, w, h, type, 0, 0 };
	AuditImage<uint8_t> dst_image{ AuditBufferType::PLANE, w, h, type, 0, 0 };
	zimg::AlignedVector<char> tmp(graph.get_tmp_size());

	src_image.set_fill_val(255);
	dst_image.set_fill_val(0);

	src_image.default_fill();
	dst_image.default_fill();

	ASSERT_THROW(graph.process(src_image.as_read_buffer(), dst_image.as_write_buffer(), tmp.data(), { cb, nullptr }, nullptr), zimg::error::UserCallbackFailed);

	SCOPED_TRACE("validating dst");
	dst_image.validate();
}
