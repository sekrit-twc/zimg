#include "common/make_unique.h"
#include "common/pixel.h"
#include "graph/basic_filter.h"

#include "gtest/gtest.h"
#include "filter_validator.h"
#include "mock_filter.h"

TEST(BasicFilterTest, test_copy_filter)
{
	const zimg::PixelType types[] = { zimg::PixelType::BYTE, zimg::PixelType::WORD, zimg::PixelType::HALF, zimg::PixelType::FLOAT };
	const unsigned w = 591;
	const unsigned h = 333;

	const char *expected_sha1[][3] = {
		{ "b7399d798c5f96b4c9ac4c6cccd4c979468bdc7a" },
		{ "43362943f1de4b51f45679a0c460f55c8bd8d2f2" },
		{ "1a25ec59d5708d3bfc36d87b05f6d7625d4a3d24" },
		{ "078016e8752bcfb63b16c86b4ae212a51579f028" }
	};

	for (unsigned x = 0; x < 4; ++x) {
		SCOPED_TRACE(static_cast<int>(types[x]));

		zimg::graph::CopyFilter copy{ w, h, types[x] };

		FilterValidator validator{ &copy, w, h, types[x] };
		validator.set_sha1(expected_sha1[x]);
		validator.validate();
	}
}

TEST(BasicFilterTest, test_mux_filter)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::BYTE;
	const unsigned hsupport = 3;
	const unsigned vsupport = 2;

	const uint8_t test_byte = 0xDC;

	const char *expected_sha1[3] = { "380abfe5ff750e2542332295b0338cfc91d805ef" };

	for (unsigned state = 0; state < 2; ++state) {
		for (unsigned entire_row = 0; entire_row < 2; ++entire_row) {
			SCOPED_TRACE(!!state);
			SCOPED_TRACE(!!entire_row);

			zimg::graph::ImageFilter::filter_flags flags{};
			flags.has_state = !!state;
			flags.entire_row = !!entire_row;

			auto filter = ztd::make_unique<SplatFilter<uint8_t>>(w, h, type, flags);
			filter->set_horizontal_support(hsupport);
			filter->set_vertical_support(vsupport);
			filter->set_output_val(test_byte);
			filter->enable_input_checking(false);

			zimg::graph::MuxFilter mux{ std::move(filter) };

			auto mux_flags = mux.get_flags();
			auto mux_attr = mux.get_image_attributes();

			EXPECT_EQ(!!state, !!mux_flags.has_state);
			EXPECT_EQ(!!entire_row, !!mux_flags.entire_row);

			EXPECT_EQ(w, mux_attr.width);
			EXPECT_EQ(h, mux_attr.height);
			EXPECT_EQ(type, mux_attr.type);
			EXPECT_EQ(vsupport * 2 + 1, mux.get_max_buffering());

			if (!flags.entire_row) {
				auto range = mux.get_required_col_range(5, 300);
				EXPECT_EQ(2U, range.first);
				EXPECT_EQ(303U, range.second);
			}

			FilterValidator validator{ &mux, w, h, type };
			validator.set_sha1(expected_sha1);
			validator.validate();
		}
	}
}
