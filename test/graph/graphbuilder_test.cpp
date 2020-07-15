#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include "colorspace/colorspace.h"
#include "common/pixel.h"
#include "depth/depth.h"
#include "graph/filtergraph.h"
#include "graph/graphbuilder.h"
#include "resize/resize.h"
#include "unresize/unresize.h"

#include "gtest/gtest.h"

namespace {

using zimg::colorspace::MatrixCoefficients;
using zimg::colorspace::TransferCharacteristics;
using zimg::colorspace::ColorPrimaries;
using zimg::graph::GraphBuilder;

typedef std::vector<std::string> TraceList;

class TracingObserver : public zimg::graph::FilterObserver {
	TraceList m_trace;
public:
	const TraceList &trace() const { return m_trace; }

	void yuv_to_grey() override { m_trace.push_back("yuv_to_grey"); }

	void grey_to_yuv() override { m_trace.push_back("grey_to_yuv"); }
	void grey_to_rgb() override { m_trace.push_back("grey_to_rgb"); }

	void premultiply() override { m_trace.push_back("premultiply"); }
	void unpremultiply() override { m_trace.push_back("unpremultiply"); }
	void add_opaque() override { m_trace.push_back("add_opaque"); }
	void discard_alpha() override { m_trace.push_back("discard_alpha"); }

	void colorspace(const zimg::colorspace::ColorspaceConversion &conv) override
	{
		char buffer[128];
		sprintf(buffer, "colorspace: [%d, %d, %d] => [%d, %d, %d] (%f)\n",
			static_cast<int>(conv.csp_in.matrix),
			static_cast<int>(conv.csp_in.transfer),
			static_cast<int>(conv.csp_in.primaries),
			static_cast<int>(conv.csp_out.matrix),
			static_cast<int>(conv.csp_out.transfer),
			static_cast<int>(conv.csp_out.primaries),
			conv.peak_luminance);
		m_trace.push_back(buffer);
	}

	void depth(const zimg::depth::DepthConversion &conv, int plane) override
	{
		char buffer[128];
		sprintf(buffer, "depth[%d]: [%d/%u %c:%c%s] => [%d/%u %c:%c%s]\n",
			plane,
			static_cast<int>(conv.pixel_in.type),
			conv.pixel_in.depth,
			conv.pixel_in.fullrange ? 'f' : 'l',
			conv.pixel_in.chroma ? 'c' : 'l',
			conv.pixel_in.ycgco ? " ycgco" : "",
			static_cast<int>(conv.pixel_out.type),
			conv.pixel_out.depth,
			conv.pixel_out.fullrange ? 'f' : 'l',
			conv.pixel_out.chroma ? 'c' : 'l',
			conv.pixel_out.ycgco ? " ycgco" : "");
		m_trace.push_back(buffer);
	}

	void resize(const zimg::resize::ResizeConversion &conv, int plane) override
	{
		char buffer[128];
		sprintf(buffer, "resize[%d]: [%u, %u] => [%u, %u] (%f, %f, %f, %f)\n",
			plane,
			conv.src_width,
			conv.src_height,
			conv.dst_width,
			conv.dst_height,
			conv.shift_w,
			conv.shift_h,
			conv.subwidth,
			conv.subheight);
		m_trace.push_back(buffer);
	}

	void unresize(const zimg::unresize::UnresizeConversion &conv, int plane) override
	{
		char buffer[128];
		sprintf(buffer, "unresize: [%u, %u] => [%u, %u] (%f, %f)\n",
			conv.up_width,
			conv.up_height,
			conv.orig_width,
			conv.orig_height,
			conv.shift_w,
			conv.shift_h);
		m_trace.push_back(buffer);
	}
};


GraphBuilder::state make_basic_rgb_state()
{
	GraphBuilder::state state{};
	state.width = 64;
	state.height = 48;
	state.type = zimg::PixelType::FLOAT;
	state.subsample_w = 0;
	state.subsample_h = 0;
	state.color = GraphBuilder::ColorFamily::RGB;
	state.colorspace = { MatrixCoefficients::RGB, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 };
	state.depth = zimg::pixel_depth(zimg::PixelType::FLOAT);
	state.fullrange = false;
	state.parity = GraphBuilder::FieldParity::PROGRESSIVE;
	state.chroma_location_w = GraphBuilder::ChromaLocationW::CENTER;
	state.chroma_location_h = GraphBuilder::ChromaLocationH::CENTER;
	state.active_left = 0.0;
	state.active_top = 0.0;
	state.active_width = 64.0;
	state.active_height = 48.0;
	state.alpha = GraphBuilder::AlphaType::NONE;
	return state;
}

GraphBuilder::state make_basic_yuv_state()
{
	GraphBuilder::state state = make_basic_rgb_state();
	state.color = GraphBuilder::ColorFamily::YUV;
	state.colorspace = { MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 };
	return state;
}

void set_resolution(GraphBuilder::state &state, unsigned width, unsigned height)
{
	state.width = width;
	state.height = height;
	state.active_left = 0;
	state.active_top = 0;
	state.active_width = width;
	state.active_height = height;
}

void test_case(const GraphBuilder::state &source, const GraphBuilder::state &target, const TraceList &trace)
{
	GraphBuilder builder;
	TracingObserver observer;
	builder.set_source(source).connect(target, nullptr, &observer).complete();

	EXPECT_EQ(trace.size(), observer.trace().size());
	for (size_t i = 0; i < std::min(trace.size(), observer.trace().size()); ++i) {
		EXPECT_TRUE(observer.trace()[i].rfind(trace[i]) == 0)
			<< "Expected: " << trace[i] << "\nActual: " << observer.trace()[i];
	}
	for (size_t i = std::min(trace.size(), observer.trace().size()); i < observer.trace().size(); ++i) {
		ADD_FAILURE() << "Unexpected[" << i << "]: " << observer.trace()[i];
	}
}

} // namespace


TEST(GraphBuilderTest, test_noop)
{
	auto source = make_basic_rgb_state();
	auto target = source;
	test_case(source, target, {});
}

TEST(GraphBuilderTest, test_resize_only_rgb)
{
	auto source = make_basic_rgb_state();
	set_resolution(source, 64, 48);

	auto target = source;
	set_resolution(target, 128, 96);

	test_case(source, target, { "resize[0]: [64, 48] => [128, 96]" });
}

TEST(GraphBuilderTest, test_resize_only_444)
{
	auto source = make_basic_yuv_state();
	set_resolution(source, 64, 48);

	auto target = source;
	set_resolution(target, 128, 96);

	test_case(source, target, {
		"resize[0]: [64, 48] => [128, 96]",
		"resize[1]: [64, 48] => [128, 96]",
	});
}

TEST(GraphBuilderTest, test_resize_only_420)
{
	auto source = make_basic_yuv_state();
	set_resolution(source, 64, 48);
	source.subsample_w = 1;
	source.subsample_h = 1;

	auto target = source;
	set_resolution(target, 128, 96);

	test_case(source, target, {
		"resize[0]: [64, 48] => [128, 96]",
		"resize[1]: [32, 24] => [64, 48]"
	});
}

TEST(GraphBuilderTest, test_resize_420_to_444)
{
	auto source = make_basic_yuv_state();
	source.subsample_w = 1;
	source.subsample_h = 1;

	auto target = make_basic_yuv_state();

	test_case(source, target, { "resize[1]" });
}

TEST(GraphBuilderTest, test_resize_444_to_420)
{
	auto source = make_basic_yuv_state();

	auto target = make_basic_yuv_state();
	target.subsample_w = 1;
	target.subsample_h = 1;

	test_case(source, target, { "resize[1]" });
}

TEST(GraphBuilderTest, test_resize_chromaloc_upsample)
{
	auto source = make_basic_yuv_state();
	set_resolution(source, 64, 48);
	source.subsample_w = 1;
	source.subsample_h = 1;
	source.chroma_location_w = GraphBuilder::ChromaLocationW::LEFT;
	source.chroma_location_h = GraphBuilder::ChromaLocationH::BOTTOM;

	auto target = make_basic_yuv_state();
	set_resolution(target, 64, 48);

	test_case(source, target, { "resize[1]: [32, 24] => [64, 48] (0.250000, -0.250000, 32.000000, 24.000000)" });
}

TEST(GraphBuilderTest, test_resize_chromaloc_downsample)
{
	auto source = make_basic_yuv_state();
	set_resolution(source, 64, 48);

	auto target = make_basic_yuv_state();
	set_resolution(target, 64, 48);
	target.subsample_w = 1;
	target.subsample_h = 1;
	target.chroma_location_w = GraphBuilder::ChromaLocationW::LEFT;
	target.chroma_location_h = GraphBuilder::ChromaLocationH::BOTTOM;

	test_case(source, target, { "resize[1]: [64, 48] => [32, 24] (-0.500000, 0.500000, 64.000000, 48.000000)" });
}

TEST(GraphBuilderTest, test_resize_deinterlace)
{
	auto source = make_basic_yuv_state();
	set_resolution(source, 64, 48);
	source.subsample_h = 1;

	auto target = make_basic_yuv_state();
	set_resolution(target, 64, 48);
	target.subsample_h = 1;
	target.parity = GraphBuilder::FieldParity::PROGRESSIVE;

	source.parity = GraphBuilder::FieldParity::TOP;
	test_case(source, target, {
		"resize[0]: [64, 48] => [64, 48] (0.000000, 0.250000, 64.000000, 48.000000)",
		"resize[1]: [64, 24] => [64, 24] (0.000000, 0.250000, 64.000000, 24.000000)",
	});

	source.parity = GraphBuilder::FieldParity::BOTTOM;
	test_case(source, target, {
		"resize[0]: [64, 48] => [64, 48] (0.000000, -0.250000, 64.000000, 48.000000)",
		"resize[1]: [64, 24] => [64, 24] (0.000000, -0.250000, 64.000000, 24.000000)",
	});
}

TEST(GraphBuilderTest, test_resize_interlace_to_interlace)
{
	auto source = make_basic_yuv_state();
	set_resolution(source, 64, 48);
	source.subsample_w = 1;
	source.subsample_h = 1;
	source.parity = GraphBuilder::FieldParity::TOP;
	source.chroma_location_w = GraphBuilder::ChromaLocationW::LEFT;
	source.chroma_location_h = GraphBuilder::ChromaLocationH::BOTTOM;

	auto target = make_basic_yuv_state();
	set_resolution(target, 64, 96);
	target.subsample_w = 1;
	target.subsample_h = 1;
	target.parity = GraphBuilder::FieldParity::TOP;
	target.chroma_location_w = GraphBuilder::ChromaLocationW::LEFT;
	target.chroma_location_h = GraphBuilder::ChromaLocationH::BOTTOM;

	test_case(source, target, {
		"resize[0]: [64, 48] => [64, 96] (0.000000, 0.125000, 64.000000, 48.000000)",
		"resize[1]: [32, 24] => [32, 48] (0.000000, 0.062500, 32.000000, 24.000000)",
	});
}

TEST(GraphBuilderTest, test_resize_byte_fast_path)
{
	auto source = make_basic_rgb_state();
	source.type = zimg::PixelType::BYTE;
	source.depth = 8;
	source.fullrange = true;
	set_resolution(source, 64, 48);

	auto target = source;
	set_resolution(target, 128, 96);

	test_case(source, target, {
		"depth[0]: [0/8 l:l] => [1/16 l:l]",
		"resize",
		"depth[0]: [1/16 l:l] => [0/8 l:l]",
	});
}

TEST(GraphBuilderTest, test_resize_byte_slow_path)
{
	auto source = make_basic_rgb_state();
	source.type = zimg::PixelType::BYTE;
	source.depth = 8;
	source.fullrange = true;
	set_resolution(source, 64, 48);

	auto target = source;
	target.fullrange = false;
	set_resolution(target, 128, 96);

	test_case(source, target, {
		"depth[0]: [0/8 f:l] => [3/32 l:l]",
		"resize",
		"depth[0]: [3/32 l:l] => [0/8 l:l]",
		});
}

TEST(GraphBuilderTest, test_resize_byte_word)
{
	auto source = make_basic_yuv_state();
	source.type = zimg::PixelType::BYTE;
	source.depth = 8;
	source.fullrange = true;
	source.colorspace.matrix = MatrixCoefficients::YCGCO;
	set_resolution(source, 64, 48);

	auto target = source;
	target.type = zimg::PixelType::WORD;
	target.depth = 10;
	target.fullrange = false;
	set_resolution(target, 128, 96);

	test_case(source, target, {
		"depth[0]: [0/8 f:l ycgco] => [1/10 l:l ycgco]",
		"resize[0]",
		"depth[1]: [0/8 f:c ycgco] => [1/10 l:c ycgco]",
		"resize[1]",
	});
}

TEST(GraphBuilderTest, test_resize_word_byte)
{
	auto source = make_basic_yuv_state();
	source.type = zimg::PixelType::WORD;
	source.depth = 10;
	source.fullrange = false;
	source.colorspace.matrix = MatrixCoefficients::YCGCO;
	set_resolution(source, 64, 48);

	auto target = source;
	target.type = zimg::PixelType::BYTE;
	target.depth = 8;
	target.fullrange = true;
	set_resolution(target, 128, 96);

	test_case(source, target, {
		"resize[0]",
		"depth[0]: [1/10 l:l ycgco] => [0/8 f:l ycgco]",
		"resize[1]",
		"depth[1]: [1/10 l:c ycgco] => [0/8 f:c ycgco]",
	});
}

TEST(GraphBuilderTest, test_colorspace_only)
{
	auto source = make_basic_rgb_state();
	auto target = make_basic_yuv_state();
	test_case(source, target, { "colorspace" });
}

TEST(GraphBuilderTest, test_upscale_colorspace)
{
	auto source = make_basic_yuv_state();
	set_resolution(source, 64, 48);
	source.subsample_w = 1;
	source.subsample_h = 1;

	auto target = source;
	set_resolution(target, 96, 72);
	target.colorspace = { MatrixCoefficients::REC_2020_NCL, TransferCharacteristics::REC_709, ColorPrimaries::REC_2020 };

	test_case(source, target, {
		"resize[1]: [32, 24] => [64, 48]",
		"colorspace",
		"resize[0]: [64, 48] => [96, 72]",
		"resize[1]: [64, 48] => [48, 36]",
	});
}

TEST(GraphBuilderTest, test_downscale_colorspace)
{
	auto source = make_basic_yuv_state();
	set_resolution(source, 96, 72);
	source.subsample_w = 1;
	source.subsample_h = 1;

	auto target = source;
	set_resolution(target, 64, 48);
	target.colorspace = { MatrixCoefficients::REC_2020_NCL, TransferCharacteristics::REC_709, ColorPrimaries::REC_2020 };

	test_case(source, target, {
		"resize[0]: [96, 72] => [64, 48]",
		"resize[1]: [48, 36] => [64, 48]",
		"colorspace",
		"resize[1]: [64, 48] => [32, 24]",
	});
}

TEST(GraphBuilderTest, test_upscale_colorspace_tile)
{
	auto source = make_basic_yuv_state();
	set_resolution(source, 64, 48);
	source.subsample_w = 1;
	source.subsample_h = 1;
	source.active_left = 32;
	source.active_top = 24;
	source.active_width = 32;
	source.active_height = 24;

	auto target = make_basic_rgb_state();
	set_resolution(target, 48, 36);

	test_case(source, target, {
		"resize[0]: [64, 48] => [48, 36] (32.000000, 24.000000, 32.000000, 24.000000)",
		"resize[1]: [32, 24] => [48, 36] (16.000000, 12.000000, 16.000000, 12.000000)",
		"colorspace",
	});
}

TEST(GraphBuilderTest, test_downscale_colorspace_tile)
{
	auto source = make_basic_rgb_state();
	set_resolution(source, 96, 72);
	source.active_left = 48;
	source.active_top = 36;
	source.active_width = 48;
	source.active_height = 36;

	auto target = make_basic_yuv_state();
	set_resolution(target, 32, 24);
	target.subsample_w = 1;
	target.subsample_h = 1;

	test_case(source, target, {
		"resize[0]: [96, 72] => [32, 24] (48.000000, 36.000000, 48.000000, 36.000000)",
		"colorspace",
		"resize[1]: [32, 24] => [16, 12] (0.000000, 0.000000, 32.000000, 24.000000)",
	});
}

TEST(GraphBuilderTest, test_grey_to_grey_noop)
{
	auto source = make_basic_yuv_state();
	source.color = GraphBuilder::ColorFamily::GREY;
	source.colorspace = { MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 };

	auto target = source;
	target.colorspace = { MatrixCoefficients::REC_601, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 };

	test_case(source, target, {});
}

TEST(GraphBuilderTest, test_grey_to_rgb_noop)
{
	auto source = make_basic_yuv_state();
	source.color = GraphBuilder::ColorFamily::GREY;
	source.colorspace.matrix = MatrixCoefficients::REC_2020_CL;

	auto target = make_basic_rgb_state();
	target.colorspace.matrix = MatrixCoefficients::UNSPECIFIED;

	test_case(source, target, { "grey_to_rgb" });
}

TEST(GraphBuilderTest, test_grey_to_yuv_noop)
{
	auto source = make_basic_yuv_state();
	source.color = GraphBuilder::ColorFamily::GREY;
	source.colorspace.matrix = MatrixCoefficients::UNSPECIFIED;

	auto target = make_basic_yuv_state();
	target.colorspace.matrix = MatrixCoefficients::REC_2020_CL;

	test_case(source, target, { "grey_to_yuv" });
}

TEST(GraphBuilderTest, test_grey_to_grey_colorspace)
{
	auto source = make_basic_yuv_state();
	source.color = GraphBuilder::ColorFamily::GREY;
	source.colorspace = { MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::DCI_P3 };

	auto target = source;
	target.colorspace = { MatrixCoefficients::REC_601, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C };

	test_case(source, target, {
		"grey_to_rgb",
		"colorspace",
		"yuv_to_grey",
		});
}

TEST(GraphBuilderTest, test_yuv_to_grey_colorspace)
{
	auto source = make_basic_yuv_state();
	source.colorspace = { MatrixCoefficients::REC_709, TransferCharacteristics::REC_709, ColorPrimaries::DCI_P3 };

	auto target = source;
	target.color = GraphBuilder::ColorFamily::GREY;
	target.colorspace = { MatrixCoefficients::REC_601, TransferCharacteristics::REC_709, ColorPrimaries::SMPTE_C };

	test_case(source, target, {
		"colorspace",
		"yuv_to_grey",
	});
}

TEST(GraphBuilderTest, test_straight_to_premul)
{
	auto source = make_basic_yuv_state();
	source.subsample_w = 1;
	source.subsample_h = 1;
	source.alpha = GraphBuilder::AlphaType::STRAIGHT;

	auto target = source;
	target.alpha = GraphBuilder::AlphaType::PREMULTIPLIED;

	test_case(source, target, {
		"resize[1]",
		"premultiply",
		"resize[1]",
	});
}

TEST(GraphBuilderTest, test_premul_to_straight)
{
	auto source = make_basic_yuv_state();
	source.subsample_w = 1;
	source.subsample_h = 1;
	source.alpha = GraphBuilder::AlphaType::PREMULTIPLIED;

	auto target = source;
	target.alpha = GraphBuilder::AlphaType::STRAIGHT;

	test_case(source, target, {
		"resize[1]",
		"unpremultiply",
		"resize[1]",
	});
}

TEST(GraphBuilderTest, test_straight_to_opaque)
{
	auto source = make_basic_rgb_state();
	source.alpha = GraphBuilder::AlphaType::STRAIGHT;

	auto target = source;
	target.alpha = GraphBuilder::AlphaType::NONE;

	test_case(source, target, { "premultiply", "discard_alpha" });
}

TEST(GraphBuilderTest, test_opaque_to_straight)
{
	auto source = make_basic_rgb_state();
	source.alpha = GraphBuilder::AlphaType::NONE;

	auto target = make_basic_yuv_state();
	target.alpha = GraphBuilder::AlphaType::STRAIGHT;

	test_case(source, target, { "colorspace", "add_opaque" });
}

TEST(GraphBuilderTest, test_colorspace_straight_alpha)
{
	auto source = make_basic_yuv_state();
	source.subsample_w = 1;
	source.subsample_h = 1;
	source.alpha = GraphBuilder::AlphaType::STRAIGHT;

	auto target = make_basic_rgb_state();
	target.alpha = GraphBuilder::AlphaType::STRAIGHT;

	test_case(source, target, {
		"resize[1]",
		"premultiply",
		"colorspace",
		"unpremultiply",
	});
}

TEST(GraphBuilderTest, test_resize_straight_alpha)
{
	auto source = make_basic_rgb_state();
	set_resolution(source, 64, 48);
	source.alpha = GraphBuilder::AlphaType::STRAIGHT;

	auto target = source;
	set_resolution(target, 128, 96);

	test_case(source, target, {
		"premultiply",
		"resize[0]: [64, 48] => [128, 96]",
		"resize[3]: [64, 48] => [128, 96]",
		"unpremultiply",
	});
}

TEST(GraphBuilderTest, test_resize_premul_alpha)
{
	auto source = make_basic_rgb_state();
	set_resolution(source, 64, 48);
	source.alpha = GraphBuilder::AlphaType::PREMULTIPLIED;

	auto target = source;
	set_resolution(target, 128, 96);

	test_case(source, target, {
		"resize[0]: [64, 48] => [128, 96]",
		"resize[3]: [64, 48] => [128, 96]",
	});
}

TEST(GraphBuilderTest, test_straight_depth)
{
	auto source = make_basic_yuv_state();
	set_resolution(source, 64, 48);
	source.alpha = GraphBuilder::AlphaType::STRAIGHT;

	auto target = source;
	target.type = zimg::PixelType::WORD;
	target.depth = 16;

	test_case(source, target, {
		"depth[0]: [3/32 l:l] => [1/16 l:l]",
		"depth[1]: [3/32 l:c] => [1/16 l:c]",
		"depth[3]: [3/32 l:l] => [1/16 f:l]",
	});
}

TEST(GraphBuilderTest, test_straight_depth_tile)
{
	auto source = make_basic_yuv_state();
	set_resolution(source, 64, 48);
	source.alpha = GraphBuilder::AlphaType::STRAIGHT;
	source.active_width = 32;
	source.active_height = 24;

	auto target = source;
	set_resolution(target, 32, 24);
	target.type = zimg::PixelType::WORD;
	target.depth = 16;

	test_case(source, target, {
		"resize[0]: [64, 48] => [32, 24] (0.000000, 0.000000, 32.000000, 24.000000)",
		"depth[0]: [3/32 l:l] => [1/16 l:l]",
		"resize[1]: [64, 48] => [32, 24] (0.000000, 0.000000, 32.000000, 24.000000)",
		"depth[1]: [3/32 l:c] => [1/16 l:c]",
		"resize[3]: [64, 48] => [32, 24] (0.000000, 0.000000, 32.000000, 24.000000)",
		"depth[3]: [3/32 l:l] => [1/16 f:l]",
	});
}
