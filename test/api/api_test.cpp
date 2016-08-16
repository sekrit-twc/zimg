#include <cmath>
#include <cstddef>
#include <cstring>
#include "api/zimg.h"

#include "gtest/gtest.h"

TEST(APITest, test_api_2_0_compat)
{
	const unsigned API_2_0 = ZIMG_MAKE_API_VERSION(2, 0);
	const size_t extra_off = offsetof(zimg_image_format, active_region);
	const size_t extra_len = sizeof(zimg_image_format) - extra_off;

	zimg_image_format format;
	std::memset(reinterpret_cast<unsigned char *>(&format) + extra_off, 0xCC, extra_len);

	zimg_image_format_default(&format, API_2_0);
	EXPECT_EQ(API_2_0, format.version);
	for (size_t i = extra_off; i < extra_len; ++i) {
		EXPECT_EQ(0xCC, *(reinterpret_cast<unsigned char *>(&format) + i));
	}

	format.width = 640;
	format.height = 480;
	format.pixel_type = ZIMG_PIXEL_BYTE;

	// Should trigger error in API 2.1+.
	format.active_region.left = 0;
	format.active_region.top = 0;
	format.active_region.width = -INFINITY;
	format.active_region.height = -INFINITY;

	zimg_graph_builder_params params;
	zimg_graph_builder_params_default(&params, API_2_0);
	EXPECT_EQ(API_2_0, params.version);

	zimg_filter_graph *graph = zimg_filter_graph_build(&format, &format, &params);
	EXPECT_TRUE(graph);
	zimg_filter_graph_free(graph);
}
