// z.lib example code for integration with graphengine.
//
// Example code demonstrates the use of z.lib to scale a BMP image in linear
// light by combining multiple z.lib filter graphs through graphengine.

#define ZIMG_GRAPHENGINE_API

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include <zimg++.hpp>
#include <graphengine/graph.h>

#include "aligned_malloc.h"
#include "argparse.h"
#include "win32_bitmap.h"

#if ZIMG_API_VERSION < ZIMG_MAKE_API_VERSION(2, 255)
  #error graphengine requires z.lib API 3.0
#endif

namespace {

struct Arguments {
	const char *inpath;
	const char *outpath;
	unsigned out_w;
	unsigned out_h;
	char half;
	char gamma;
};

const ArgparseOption program_switches[] = {
	{ OPTION_FLAG, "f", "half",  offsetof(Arguments, half), nullptr,  "half precision intermediate" },
	{ OPTION_FLAG, "g", "gamma", offsetof(Arguments, gamma), nullptr, "scale in gamma domain instead of linnear" },
	{ OPTION_NULL },
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING, nullptr, "inpath",  offsetof(Arguments, inpath),  nullptr, "input path" },
	{ OPTION_STRING, nullptr, "outpath", offsetof(Arguments, outpath), nullptr, "output path" },
	{ OPTION_UINT,   "w",     "width",   offsetof(Arguments, out_w),   nullptr, "width" },
	{ OPTION_UINT,   "h",     "height",  offsetof(Arguments, out_h),   nullptr, "height" },
	{ OPTION_NULL },
};

const ArgparseCommandLine program_def = { program_switches, program_positional, "graphengine_example", "resize BMP images in linear light" };

struct Callback {
	const graphengine::BufferDescriptor *buffer;
	void *packed;
	ptrdiff_t packed_stride;
	unsigned packed_bit_depth;
};

std::shared_ptr<void> allocate_buffer(graphengine::BufferDescriptor buf[], unsigned num_planes, size_t rowsize, unsigned height, unsigned mask)
{
	rowsize = (rowsize + 63) & ~63;
	height = mask == graphengine::BUFFER_MAX ? height : mask + 1;

	std::shared_ptr<void> data{ aligned_malloc(rowsize * height * 3, 64), aligned_free };
	buf[0] = { data.get(), static_cast<ptrdiff_t>(rowsize), mask };
	buf[1] = { static_cast<unsigned char *>(data.get()) + 1 * rowsize * height, static_cast<ptrdiff_t>(rowsize), mask };
	buf[2] = { static_cast<unsigned char *>(data.get()) + 2 * rowsize * height, static_cast<ptrdiff_t>(rowsize), mask };

	return data;
}

// Input callback to convert BGR24/32 to planar RGB.
void unpack_bgr(const void *bgr, void * const planar[4], unsigned bit_depth, unsigned left, unsigned right)
{
	const uint8_t *packed_bgr = static_cast<const uint8_t *>(bgr);
	uint8_t *planar_r = static_cast<uint8_t *>(planar[0]);
	uint8_t *planar_g = static_cast<uint8_t *>(planar[1]);
	uint8_t *planar_b = static_cast<uint8_t *>(planar[2]);
	uint8_t *planar_a = static_cast<uint8_t *>(planar[3]);
	unsigned step = bit_depth / 8;

	for (unsigned j = left; j < right; ++j) {
		uint8_t r, g, b;

		b = packed_bgr[j * step + 0];
		g = packed_bgr[j * step + 1];
		r = packed_bgr[j * step + 2];

		planar_r[j] = r;
		planar_g[j] = g;
		planar_b[j] = b;

		if (planar_a && step == 4)
			planar_a[j] = packed_bgr[j * step + 3];
	}
}

// Output callback to convert planar RGB to BGR24/32.
void pack_bgr(const void * const planar[4], void *bgr, unsigned bit_depth, unsigned left, unsigned right)
{
	const uint8_t *planar_r = static_cast<const uint8_t *>(planar[0]);
	const uint8_t *planar_g = static_cast<const uint8_t *>(planar[1]);
	const uint8_t *planar_b = static_cast<const uint8_t *>(planar[2]);
	const uint8_t *planar_a = static_cast<const uint8_t *>(planar[3]);
	uint8_t *packed_bgr = static_cast<uint8_t *>(bgr);
	unsigned step = bit_depth / 8;

	for (unsigned j = left; j < right; ++j) {
		uint8_t r, g, b;

		r = planar_r[j];
		g = planar_g[j];
		b = planar_b[j];

		packed_bgr[j * step + 0] = b;
		packed_bgr[j * step + 1] = g;
		packed_bgr[j * step + 2] = r;

		if (planar_a && step == 4)
			packed_bgr[j * step + 3] = planar_a[j];
	}
}

int unpack_image(void *user, unsigned i, unsigned left, unsigned right)
{
	const Callback *cb = static_cast<Callback *>(user);
	const void *img = static_cast<uint8_t *>(cb->packed) + static_cast<ptrdiff_t>(i) * cb->packed_stride;
	void *buf_data[4] = { 0 };

	for (unsigned p = 0; p < 3; ++p) {
		buf_data[p] = static_cast<char *>(cb->buffer[p].get_line(i));
	}

	unpack_bgr(img, buf_data, cb->packed_bit_depth, left, right);
	return 0;
}

int pack_image(void *user, unsigned i, unsigned left, unsigned right)
{
	const Callback *cb = static_cast<Callback *>(user);
	void *img = static_cast<uint8_t *>(cb->packed) + static_cast<ptrdiff_t>(i) * cb->packed_stride;
	void *buf_data[4] = { 0 };

	for (unsigned p = 0; p < 3; ++p) {
		buf_data[p] = static_cast<char *>(cb->buffer[p].get_line(i));
	}

	pack_bgr(buf_data, img, cb->packed_bit_depth, left, right);
	return 0;
}

void execute_graphengine(const graphengine::Graph *graph, graphengine::node_id source_id, graphengine::node_id sink_id, const WindowsBitmap &in_bmp, WindowsBitmap &out_bmp)
{
	graphengine::BufferDescriptor in_buf[3];
	graphengine::BufferDescriptor out_buf[3];

	Callback unpack_cb{ in_buf, const_cast<uint8_t *>(in_bmp.read_ptr()), in_bmp.stride(), static_cast<unsigned>(in_bmp.bit_count()) };
	Callback pack_cb{ out_buf, out_bmp.write_ptr(), out_bmp.stride(), static_cast<unsigned>(out_bmp.bit_count()) };

	graphengine::Graph::Endpoint endpoints[2];

	endpoints[0].id = source_id;
	endpoints[0].buffer = in_buf;
	endpoints[0].callback = { unpack_image, &unpack_cb };

	endpoints[1].id = sink_id;
	endpoints[1].buffer = out_buf;
	endpoints[1].callback = { pack_image, &pack_cb };

	graphengine::Graph::BufferingRequirement buffering = graph->get_buffering_requirement();

	auto &in_buffering = *std::find_if(buffering.begin(), buffering.end(),
		[=](const graphengine::Graph::Buffering &entry) { return entry.id == source_id; });
	std::shared_ptr<void> in_buf_mem = allocate_buffer(in_buf, 3, static_cast<size_t>(in_bmp.width()) * 3, in_bmp.height(), in_buffering.mask);

	auto &out_buffering = *std::find_if(buffering.begin(), buffering.end(),
		[=](const graphengine::Graph::Buffering &entry) { return entry.id == sink_id; });
	std::shared_ptr<void> out_buf_mem = allocate_buffer(out_buf, 3, static_cast<size_t>(out_bmp.width()) * 3, out_bmp.height(), out_buffering.mask);

	std::shared_ptr<void> tmp{ aligned_malloc(graph->get_tmp_size(), 64), aligned_free };
	graph->run(endpoints, tmp.get());
}

void execute(const Arguments &args)
{
	WindowsBitmap in_bmp{ args.inpath, WindowsBitmap::READ_TAG };
	WindowsBitmap out_bmp{ args.outpath, static_cast<int>(args.out_w), static_cast<int>(args.out_h), 24 };

	// Common parameters for all steps.
	zimgxx::zfilter_graph_builder_params params;
	params.allow_approximate_gamma = 1;
	params.cpu_type = ZIMG_CPU_AUTO_64B;

	// (1) Define the source format.
	zimgxx::zimage_format in_format;
	in_format.width = in_bmp.width();
	in_format.height = in_bmp.height();
	in_format.pixel_type = ZIMG_PIXEL_BYTE;
	in_format.color_family = ZIMG_COLOR_RGB;
	in_format.pixel_range = ZIMG_RANGE_FULL;
	in_format.matrix_coefficients = ZIMG_MATRIX_RGB;
	in_format.transfer_characteristics = ZIMG_TRANSFER_IEC_61966_2_1;

	// (2) Define each stage of filtering.
	std::vector<zimgxx::SubGraph> steps;
	zimgxx::zimage_format prev_format = in_format;

	// Step 1: conversion to linear light
	{
		zimgxx::zimage_format format;
		format.width = prev_format.width;
		format.height = prev_format.height;
		format.pixel_type = args.half ? ZIMG_PIXEL_HALF : ZIMG_PIXEL_FLOAT;
		format.color_family = ZIMG_COLOR_RGB;
		format.matrix_coefficients = ZIMG_MATRIX_RGB;
		format.transfer_characteristics = args.gamma ? in_format.transfer_characteristics : ZIMG_TRANSFER_LINEAR;

		steps.push_back(zimgxx::SubGraph::build(prev_format, format, &params));
		prev_format = format;
	}

	// Step 2: scaling to output resolution
	{
		zimgxx::zimage_format format;
		format.width = out_bmp.width();
		format.height = out_bmp.height();
		format.pixel_type = prev_format.pixel_type;
		format.color_family = ZIMG_COLOR_RGB;
		format.matrix_coefficients = ZIMG_MATRIX_RGB;
		format.transfer_characteristics = prev_format.transfer_characteristics;

		steps.push_back(zimgxx::SubGraph::build(prev_format, format, &params));
		prev_format = format;
	}

	// Step 3: conversion to gamma
	{
		zimgxx::zimage_format format;
		format.width = prev_format.width;
		format.height = prev_format.height;
		format.pixel_type = ZIMG_PIXEL_BYTE;
		format.color_family = ZIMG_COLOR_RGB;
		format.pixel_range = ZIMG_RANGE_FULL;
		format.matrix_coefficients = ZIMG_MATRIX_RGB;
		format.transfer_characteristics = ZIMG_TRANSFER_IEC_61966_2_1;

		steps.push_back(zimgxx::SubGraph::build(prev_format, format, &params));
		prev_format = format;
	}

	assert(prev_format.width == out_bmp.width());
	assert(prev_format.height == out_bmp.height());

	// (3) Create a graphengine::GraphImpl and set the source node according to the input format.
	std::unique_ptr<graphengine::Graph> graph = std::make_unique<graphengine::GraphImpl>();
	graphengine::PlaneDescriptor in_format_desc[3];
	std::fill_n(in_format_desc, 3, graphengine::PlaneDescriptor{ in_format.width, in_format.height, 1 });
	graphengine::node_id source_id = graph->add_source(3, in_format_desc);

	// (4) Attach subgraphs.
	graphengine::node_dep_desc last[3] = { {source_id, 0}, {source_id, 1}, {source_id, 2} };

	for (const auto &step : steps) {
		int in_ids[4], out_ids[4];
		auto num_endpoints = step.get_endpoint_ids(in_ids, out_ids);
		assert(num_endpoints.first == 3);
		assert(num_endpoints.second == 3);

		graphengine::SubGraph::Mapping in_mapping[3] = { { in_ids[0], last[0] }, { in_ids[1], last[1] }, { in_ids[2], last[2] } };
		graphengine::SubGraph::Mapping out_mapping[3];

		step.get_subgraph()->connect(graph.get(), 3, in_mapping, out_mapping);

		for (unsigned p = 0; p < 3; ++p) {
			for (const auto &mapping : out_mapping) {
				if (mapping.internal_id == out_ids[p]) {
					last[p] = mapping.external_dep;
					break;
				}
			}
		}
	}

	// (5) Finalize the graph by setting the sink node.
	graphengine::node_id sink_id = graph->add_sink(3, last);

	// (6) Invoke graphengine API to run the filter graph.
	execute_graphengine(graph.get(), source_id, sink_id, in_bmp, out_bmp);
}

} // namespace


int main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)) < 0)
		return ret == ARGPARSE_HELP_MESSAGE ? 0 : ret;

	try {
		execute(args);
	} catch (const std::system_error &e) {
		std::cerr << "system_error " << e.code() << ": " << e.what() << '\n';
		return 2;
	} catch (const zimgxx::zerror &e) {
		std::cerr << "zimg error " << e.code << ": " << e.msg << '\n';
		return 2;
	} catch (const graphengine::Exception &e) {
		std::cerr << "graphengine error " << e.code << ": " << e.msg << '\n';
	} catch (const std::runtime_error &e) {
		std::cerr << "runtime_error: " << e.what() << '\n';
		return 2;
	} catch (const std::logic_error &e) {
		std::cerr << "logic_error: " << e.what() << '\n';
		return 2;
	}

	return 0;
}
