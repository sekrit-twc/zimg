#include <atomic>
#include <cstdio>
#include <fstream>
#include <exception>
#include <iostream>
#include <mutex>
#include <streambuf>
#include <string>
#include <thread>
#include "common/alloc.h"
#include "common/except.h"
#include "common/static_map.h"
#include "graph/filtergraph2.h"
#include "graph/graphbuilder2.h"
#include "graph/image_filter.h"
#include "resize/filter.h"

#include "apps.h"
#include "argparse.h"
#include "frame.h"
#include "json.h"
#include "table.h"
#include "timer.h"

namespace {

class TracingFilterFactory2 : public zimg::graph::DefaultFilterFactory2 {
public:
	filter_list create_colorspace(const zimg::colorspace::ColorspaceConversion &conv) override
	{
		printf("colorspace: [%d, %d, %d] => [%d, %d, %d] (%f)\n",
		       static_cast<int>(conv.csp_in.matrix),
		       static_cast<int>(conv.csp_in.transfer),
		       static_cast<int>(conv.csp_in.primaries),
		       static_cast<int>(conv.csp_out.matrix),
		       static_cast<int>(conv.csp_out.transfer),
		       static_cast<int>(conv.csp_out.primaries),
		       conv.peak_luminance);

		return zimg::graph::DefaultFilterFactory2::create_colorspace(conv);
	}

	filter_list create_depth(const zimg::depth::DepthConversion &conv) override
	{
		printf("depth: [%d/%u %c:%c] => [%d/%u %c:%c]\n",
		       static_cast<int>(conv.pixel_in.type),
		       conv.pixel_in.depth,
		       conv.pixel_in.fullrange ? 'f' : 'l',
			   conv.pixel_in.chroma ? 'c' : 'l',
		       static_cast<int>(conv.pixel_out.type),
		       conv.pixel_out.depth,
		       conv.pixel_out.fullrange ? 'f' : 'l',
		       conv.pixel_out.chroma ? 'c' : 'l');

		return zimg::graph::DefaultFilterFactory2::create_depth(conv);
	}

	filter_list create_resize(const zimg::resize::ResizeConversion &conv) override
	{
		printf("resize: [%d, %d] => [%d, %d] (%f, %f, %f, %f)\n",
		       conv.src_width,
		       conv.src_height,
		       conv.dst_width,
		       conv.dst_height,
		       conv.shift_w,
		       conv.shift_h,
		       conv.subwidth,
		       conv.subheight);

		return zimg::graph::DefaultFilterFactory2::create_resize(conv);
	}

	filter_list create_unresize(const zimg::unresize::UnresizeConversion &conv) override
	{
		printf("unresize: [%d, %d] => [%d, %d] (%f, %f)\n",
		       conv.up_width,
		       conv.up_height,
		       conv.orig_width,
		       conv.orig_height,
		       conv.shift_w,
		       conv.shift_h);

		return zimg::graph::DefaultFilterFactory2::create_unresize(conv);
	}
};


json::Object read_graph_spec(const char *path)
{
	std::ifstream f;

	f.exceptions(std::ios_base::badbit | std::ios_base::failbit);
	f.open(path);

	std::string spec_json{ std::istreambuf_iterator<char>{ f }, std::istreambuf_iterator<char>{} };
	return std::move(json::parse_document(spec_json).object());
}

void read_graph_state(zimg::graph::GraphBuilder2::state *state, const json::Object &obj)
{
	static const zimg::static_string_map<zimg::graph::GraphBuilder2::ColorFamily, 3> color_map{
		{ "grey", zimg::graph::GraphBuilder2::ColorFamily::GREY },
		{ "rgb",  zimg::graph::GraphBuilder2::ColorFamily::RGB },
		{ "yuv",  zimg::graph::GraphBuilder2::ColorFamily::YUV },
	};
	static const zimg::static_string_map<zimg::graph::GraphBuilder2::FieldParity, 3> parity_map{
		{ "progressive", zimg::graph::GraphBuilder2::FieldParity::PROGRESSIVE },
		{ "top",         zimg::graph::GraphBuilder2::FieldParity::TOP },
		{ "bottom",      zimg::graph::GraphBuilder2::FieldParity::BOTTOM },
	};
	static const zimg::static_string_map<zimg::graph::GraphBuilder2::ChromaLocationW, 2> chromaloc_w_map{
		{ "left",   zimg::graph::GraphBuilder2::ChromaLocationW::LEFT },
		{ "center", zimg::graph::GraphBuilder2::ChromaLocationW::CENTER },
	};
	static const zimg::static_string_map<zimg::graph::GraphBuilder2::ChromaLocationH, 3> chromaloc_h_map{
		{ "center", zimg::graph::GraphBuilder2::ChromaLocationH::CENTER },
		{ "top",    zimg::graph::GraphBuilder2::ChromaLocationH::TOP },
		{ "bottom", zimg::graph::GraphBuilder2::ChromaLocationH::BOTTOM },
	};
	static const zimg::static_string_map<zimg::graph::GraphBuilder2::AlphaType, 3> alpha_map{
		{ "none", zimg::graph::GraphBuilder2::AlphaType::NONE },
		{ "straight",    zimg::graph::GraphBuilder2::AlphaType::STRAIGHT },
		{ "premul", zimg::graph::GraphBuilder2::AlphaType::PREMULTIPLED },
	};

	if (const auto &val = obj["width"])
		state->width = static_cast<unsigned>(val.integer());
	if (const auto &val = obj["height"])
		state->height = static_cast<unsigned>(val.integer());
	if (const auto &val = obj["type"])
		state->type = g_pixel_table[val.string().c_str()];

	if (const auto &val = obj["subsample_w"])
		state->subsample_w = static_cast<unsigned>(val.integer());
	if (const auto &val = obj["subsample_h"])
		state->subsample_h = static_cast<unsigned>(val.integer());

	if (const auto &val = obj["color"])
		state->color = color_map[val.string().c_str()];
	if (const auto &val = obj["colorspace"]) {
		const json::Object &colorspace_obj = val.object();

		state->colorspace.matrix = g_matrix_table[colorspace_obj["matrix"].string().c_str()];
		state->colorspace.transfer = g_transfer_table[colorspace_obj["transfer"].string().c_str()];
		state->colorspace.primaries = g_primaries_table[colorspace_obj["primaries"].string().c_str()];
	}

	if (const auto &val = obj["depth"])
		state->depth = static_cast<unsigned>(val.integer());
	if (const auto &val = obj["fullrange"])
		state->fullrange = val.boolean();

	if (const auto &val = obj["parity"])
		state->parity = parity_map[val.string().c_str()];
	if (const auto &val = obj["chroma_location_w"])
		state->chroma_location_w = chromaloc_w_map[val.string().c_str()];
	if (const auto &val = obj["chroma_location_h"])
		state->chroma_location_h = chromaloc_h_map[val.string().c_str()];

	if (const auto &val = obj["active_region"]) {
		state->active_left = val.object()["left"].number();
		state->active_top = val.object()["top"].number();
		state->active_width = val.object()["width"].number();
		state->active_height = val.object()["height"].number();
	} else {
		state->active_left = 0.0;
		state->active_top = 0.0;
		state->active_width = state->width;
		state->active_height = state->height;
	}

	if (const auto &val = obj["alpha"])
		state->alpha = alpha_map[val.string().c_str()];
}

void read_graph_params(zimg::graph::GraphBuilder2::params *params, const json::Object &obj)
{
	if (const auto &val = obj["filter"]) {
		const json::Object &filter_obj = val.object();
		auto factory_func = g_resize_table[filter_obj["name"].string().c_str()];
		params->filter = factory_func(filter_obj["param_a"].number(), filter_obj["param_b"].number());
		params->unresize = filter_obj["name"].string() == "unresize";
	} else {
		params->filter.reset(new zimg::resize::BicubicFilter{});
	}

	if (const auto &val = obj["filter_uv"]) {
		const json::Object &filter_obj = val.object();
		auto factory_func = g_resize_table[filter_obj["name"].string().c_str()];
		params->filter_uv = factory_func(filter_obj["param_a"].number(), filter_obj["param_b"].number());
	} else {
		params->filter_uv.reset(new zimg::resize::BilinearFilter{});
	}

	if (const auto &val = obj["dither_type"])
		params->dither_type = g_dither_table[val.string().c_str()];
	if (const auto &val = obj["peak_luminance"])
		params->peak_luminance = val.number();
	if (const auto &val = obj["approximate_gamma"])
		params->approximate_gamma = val.boolean();
	if (const auto &val = obj["scene_referred"])
		params->scene_referred = val.boolean();
	if (const auto &val = obj["cpu"])
		params->cpu = g_cpu_table[val.string().c_str()];
}

std::unique_ptr<zimg::graph::FilterGraph2> create_graph(const json::Object &spec,
                                                       zimg::graph::GraphBuilder2::state *src_state_out,
                                                       zimg::graph::GraphBuilder2::state *dst_state_out,
                                                       zimg::CPUClass cpu)
{
	zimg::graph::GraphBuilder2::state src_state{};
	zimg::graph::GraphBuilder2::state dst_state{};
	zimg::graph::GraphBuilder2::params params{};
	TracingFilterFactory2 factory;
	bool has_params = false;

	try {
		read_graph_state(&src_state, spec["source"].object());

		dst_state = src_state;
		read_graph_state(&dst_state, spec["target"].object());

		if (const auto &val = spec["params"]) {
			read_graph_params(&params, val.object());
			has_params = true;
		}

		if (cpu >= static_cast<zimg::CPUClass>(0))
			params.cpu = cpu;
	} catch (const std::invalid_argument &e) {
		throw std::runtime_error{ e.what() };
	} catch (const std::out_of_range &e) {
		throw std::runtime_error{ e.what() };
	}

	*src_state_out = src_state;
	*dst_state_out = dst_state;

	return zimg::graph::GraphBuilder2{}.set_source(src_state)
	                                  .connect_graph(dst_state, has_params ? &params : nullptr, &factory)
	                                  .complete_graph();
}

ImageFrame allocate_frame(const zimg::graph::GraphBuilder2::state &state)
{
	return{
		state.width,
		state.height,
		state.type,
		(state.color != zimg::graph::GraphBuilder2::ColorFamily::GREY ? 3U : 1U) + (state.alpha != zimg::graph::GraphBuilder2::AlphaType::NONE ? 1U : 0U),
		state.color != zimg::graph::GraphBuilder2::ColorFamily::RGB,
		state.subsample_w,
		state.subsample_h
	};
}

void thread_target(const zimg::graph::FilterGraph2 *graph,
                   const zimg::graph::GraphBuilder2::state *src_state,
                   const zimg::graph::GraphBuilder2::state *dst_state,
                   std::atomic_int *counter,
                   std::exception_ptr *eptr,
                   std::mutex *mutex)
{
	try {
		ImageFrame src_frame = allocate_frame(*src_state);
		ImageFrame dst_frame = allocate_frame(*dst_state);
		zimg::AlignedVector<char> tmp(graph->get_tmp_size());

		while (true) {
			if ((*counter)-- <= 0)
				break;

			graph->process(src_frame.as_read_buffer(), dst_frame.as_write_buffer(), tmp.data(), nullptr, nullptr);
		}
	} catch (...) {
		std::lock_guard<std::mutex> lock{ *mutex };
		*eptr = std::current_exception();
	}
}

void execute(const json::Object &spec, unsigned times, unsigned threads, unsigned tile_width, zimg::CPUClass cpu)
{
	zimg::graph::GraphBuilder2::state src_state;
	zimg::graph::GraphBuilder2::state dst_state;
	std::unique_ptr<zimg::graph::FilterGraph2> graph = create_graph(spec, &src_state, &dst_state, cpu);

	//if (tile_width)
		//graph->set_tile_width(tile_width);

	std::cout << '\n';
	std::cout << "input buffering:  " << graph->get_input_buffering() << '\n';
	std::cout << "output buffering: " << graph->get_output_buffering() << '\n';
	std::cout << "heap size:        " << graph->get_tmp_size() << '\n';
	//std::cout << "tile width:       " << graph->tile_width() << '\n';

	if (!threads && !std::thread::hardware_concurrency())
		throw std::runtime_error{ "could not auto-detect CPU count" };

	unsigned thread_min = threads ? threads : 1;
	unsigned thread_max = threads ? threads : std::thread::hardware_concurrency();

	for (unsigned n = thread_min; n <= thread_max; ++n) {
		std::vector<std::thread> thread_pool;
		std::atomic_int counter{ static_cast<int>(times * n) };
		std::exception_ptr eptr{};
		std::mutex mutex;
		Timer timer;

		thread_pool.reserve(n);

		timer.start();
		for (unsigned nn = 0; nn < n; ++nn) {
			thread_pool.emplace_back(thread_target, graph.get(), &src_state, &dst_state, &counter, &eptr, &mutex);
		}

		for (auto &th : thread_pool) {
			th.join();
		}
		timer.stop();

		if (eptr)
			std::rethrow_exception(eptr);

		std::cout << '\n';
		std::cout << "threads:    " << n << '\n';
		std::cout << "iterations: " << times * n << '\n';
		std::cout << "fps:        " << (times * n) / timer.elapsed() << '\n';
	}
}


struct Arguments {
	const char *specpath;
	unsigned times;
	unsigned threads;
	unsigned tile_width;
	zimg::CPUClass cpu;
};

const ArgparseOption program_switches[] = {
	{ OPTION_UINT,  nullptr, "times",      offsetof(Arguments, times),      nullptr, "number of benchmark cycles per thread" },
	{ OPTION_UINT,  nullptr, "threads",    offsetof(Arguments, threads),    nullptr, "number of threads" },
	{ OPTION_UINT,  nullptr, "tile-width", offsetof(Arguments, tile_width), nullptr, "graph tile width" },
	{ OPTION_USER1, nullptr, "cpu",        offsetof(Arguments, cpu),        arg_decode_cpu, "select CPU type" },
	{ OPTION_NULL }
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING, nullptr, "specpath", offsetof(Arguments, specpath), nullptr, "graph specification file" },
	{ OPTION_NULL }
};

const ArgparseCommandLine program_def = { program_switches, program_positional, "graph", "benchmark filter graph", };

} // namespace


int graph2_main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	args.times = 100;
	args.cpu = static_cast<zimg::CPUClass>(-1);

	if ((ret = argparse_parse(&program_def, &args, argc, argv)) < 0)
		return ret == ARGPARSE_HELP_MESSAGE ? 0 : ret;

	try {
		json::Object spec = read_graph_spec(args.specpath);
		execute(spec, args.times, args.threads, args.tile_width, args.cpu);
	} catch (const zimg::error::Exception &e) {
		std::cerr << e.what() << '\n';
		return 2;
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 2;
	}

	return 0;
}
