#include <cstdint>
#include <memory>
#include "common/mux_filter.h"
#include "common/pixel.h"
#include "common/zfilter.h"

#include "gtest/gtest.h"
#include "filter_validator.h"
#include "mock_filter.h"

TEST(MuxFilterTest, test_one_filter)
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

			zimg::ZimgFilterFlags flags{};
			flags.has_state = !!state;
			flags.entire_row = !!entire_row;

			std::unique_ptr<SplatFilter<uint8_t>> filter{ new SplatFilter<uint8_t>{ w, h, type, flags } };
			filter->set_horizontal_support(hsupport);
			filter->set_vertical_support(vsupport);
			filter->set_output_val(test_byte);
			filter->enable_input_checking(false);

			zimg::MuxFilter mux{ filter.get(), nullptr };
			filter.release();

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

			validate_filter(&mux, w, h, type, expected_sha1);
		}
	}
}

TEST(MuxFilterTest, test_two_filter)
{
	const unsigned w = 640;
	const unsigned h = 480;
	const zimg::PixelType type = zimg::PixelType::BYTE;
	const unsigned hsupport = 3;
	const unsigned vsupport = 2;

	const uint8_t test_byte1 = 0xDC;
	const uint8_t test_byte2 = 0xAA;

	const char *expected_sha1[3] = {
		"380abfe5ff750e2542332295b0338cfc91d805ef",
		"6bfc86a323d47a0814dc57a6143920ede3ffa0e4",
		"6bfc86a323d47a0814dc57a6143920ede3ffa0e4"
	};

	for (unsigned state = 0; state < 2; ++state) {
		for (unsigned entire_row = 0; entire_row < 2; ++entire_row) {
			SCOPED_TRACE(!!state);
			SCOPED_TRACE(!!entire_row);

			zimg::ZimgFilterFlags flags1{};
			flags1.has_state = false;
			flags1.entire_row = false;

			zimg::ZimgFilterFlags flags2{};
			flags2.has_state = !!state;
			flags2.entire_row = !!entire_row;

			std::unique_ptr<SplatFilter<uint8_t>> filter1{ new SplatFilter<uint8_t>{ w, h, type, flags1 } };
			std::unique_ptr<SplatFilter<uint8_t>> filter2{ new SplatFilter<uint8_t>{ w, h, type, flags2 } };

			filter1->set_horizontal_support(hsupport);
			filter1->set_vertical_support(vsupport);
			filter1->set_output_val(test_byte1);
			filter1->enable_input_checking(false);

			filter2->set_horizontal_support(hsupport);
			filter2->set_vertical_support(vsupport);
			filter2->set_output_val(test_byte2);
			filter2->enable_input_checking(false);

			zimg::MuxFilter mux{ filter1.get(), filter2.get() };
			filter1.release();
			filter2.release();

			auto mux_flags = mux.get_flags();
			auto mux_attr = mux.get_image_attributes();

			EXPECT_EQ(!!state, !!mux_flags.has_state);
			EXPECT_EQ(!!entire_row, !!mux_flags.entire_row);

			EXPECT_EQ(w, mux_attr.width);
			EXPECT_EQ(h, mux_attr.height);
			EXPECT_EQ(type, mux_attr.type);
			EXPECT_EQ(vsupport * 2 + 1, mux.get_max_buffering());

			if (!flags2.entire_row) {
				auto range = mux.get_required_col_range(5, 300);
				EXPECT_EQ(2U, range.first);
				EXPECT_EQ(303U, range.second);
			}

			validate_filter(&mux, w, h, type, expected_sha1);
		}
	}
}
