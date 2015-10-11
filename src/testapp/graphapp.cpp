#include <cstdio>
#include <fstream>
#include <exception>
#include <iostream>
#include <mutex>
#include <streambuf>
#include <string>
#include <thread>
#include <type_traits>
#include "common/alloc.h"
#include "common/static_map.h"
#include "graph/filtergraph.h"
#include "graph/graphbuilder.h"
#include "graph/image_filter.h"
#include "resize/filter.h"

#include "apps.h"
#include "argparse.h"
#include "frame.h"
#include "json.h"
#include "table.h"
#include "timer.h"

namespace {;

class TracingFilterFactory : public zimg::graph::DefaultFilterFactory {
public:
	filter_list create_colorspace(const zimg::colorspace::ColorspaceConversion &conv) override
	{
		printf("colorspace: [%d, %d, %d] => [%d, %d, %d]\n",
		       static_cast<int>(conv.csp_in.matrix),
		       static_cast<int>(conv.csp_in.transfer),
		       static_cast<int>(conv.csp_in.primaries),
		       static_cast<int>(conv.csp_out.matrix),
		       static_cast<int>(conv.csp_out.transfer),
		       static_cast<int>(conv.csp_out.primaries));

		return zimg::graph::DefaultFilterFactory::create_colorspace(conv);
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

		return zimg::graph::DefaultFilterFactory::create_depth(conv);
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

		return zimg::graph::DefaultFilterFactory::create_resize(conv);
	}
};


JsonObject read_graph_spec(const char *path)
{
	std::ifstream f;

	f.exceptions(std::ios_base::badbit | std::ios_base::failbit);
	f.open(path);

	std::string spec_json{ std::istreambuf_iterator<char>{ f }, std::istreambuf_iterator<char>{} };
	return json::parse_document(spec_json);
}

void read_graph_state(zimg::graph::GraphBuilder::state *state, const JsonObject &obj)
{
	static const zimg::static_string_map<zimg::graph::GraphBuilder::ColorFamily, 3> color_map{
		{ "grey", zimg::graph::GraphBuilder::ColorFamily::COLOR_GREY },
		{ "rgb",  zimg::graph::GraphBuilder::ColorFamily::COLOR_RGB },
		{ "yuv",  zimg::graph::GraphBuilder::ColorFamily::COLOR_YUV },
	};
	static const zimg::static_string_map<zimg::graph::GraphBuilder::FieldParity, 3> parity_map{
		{ "progressive", zimg::graph::GraphBuilder::FieldParity::FIELD_PROGRESSIVE },
		{ "top",         zimg::graph::GraphBuilder::FieldParity::FIELD_TOP },
		{ "bottom",      zimg::graph::GraphBuilder::FieldParity::FIELD_BOTTOM },
	};
	static const zimg::static_string_map<zimg::graph::GraphBuilder::ChromaLocationW, 2> chromaloc_w_map{
		{ "left",   zimg::graph::GraphBuilder::ChromaLocationW::CHROMA_W_LEFT },
		{ "center", zimg::graph::GraphBuilder::ChromaLocationW::CHROMA_W_CENTER },
	};
	static const zimg::static_string_map<zimg::graph::GraphBuilder::ChromaLocationH, 3> chromaloc_h_map{
		{ "center", zimg::graph::GraphBuilder::ChromaLocationH::CHROMA_H_CENTER },
		{ "top",    zimg::graph::GraphBuilder::ChromaLocationH::CHROMA_H_TOP },
		{ "bottom", zimg::graph::GraphBuilder::ChromaLocationH::CHROMA_H_BOTTOM },
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
		const JsonObject &colorspace_obj = val.object();

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
}

void read_graph_params(zimg::graph::GraphBuilder::params *params, const JsonObject &obj)
{
	if (const auto &val = obj["filter"]) {
		const JsonObject &filter_obj = val.object();
		auto factory_func = g_resize_table[filter_obj["name"].string().c_str()];
		params->filter = factory_func(filter_obj["param_a"].number(), filter_obj["param_b"].number());
	} else {
		params->filter.reset(new zimg::resize::BicubicFilter{ 1.0 / 3.0, 1.0 / 3.0 });
	}

	if (const auto &val = obj["filter_uv"]) {
		const JsonObject &filter_obj = val.object();
		auto factory_func = g_resize_table[filter_obj["name"].string().c_str()];
		params->filter_uv = factory_func(filter_obj["param_a"].number(), filter_obj["param_b"].number());
	} else {
		params->filter_uv.reset(new zimg::resize::BilinearFilter{});
	}

	if (const auto &val = obj["dither_type"])
		params->dither_type = g_dither_table[val.string().c_str()];
	if (const auto &val = obj["cpu"])
		params->cpu = g_cpu_table[val.string().c_str()];
}

std::unique_ptr<zimg::graph::FilterGraph> create_graph(const JsonObject &spec,
                                                       zimg::graph::GraphBuilder::state *src_state_out,
                                                       zimg::graph::GraphBuilder::state *dst_state_out)
{
	zimg::graph::GraphBuilder::state src_state{};
	zimg::graph::GraphBuilder::state dst_state{};
	zimg::graph::GraphBuilder::params params{};
	TracingFilterFactory factory;
	bool has_params = false;

	try {
		read_graph_state(&src_state, spec["source"].object());

		dst_state = src_state;
		read_graph_state(&dst_state, spec["target"].object());

		if (const auto &val = spec["params"]) {
			read_graph_params(&params, val.object());
			has_params = true;
		}
	} catch (const std::invalid_argument &e) {
		throw std::runtime_error{ e.what() };
	} catch (const std::out_of_range &e) {
		throw std::runtime_error{ e.what() };
	}

	*src_state_out = src_state;
	*dst_state_out = dst_state;

	return zimg::graph::GraphBuilder{}.set_factory(&factory).
	                                   set_source(src_state).
	                                   connect_graph(dst_state, has_params ? &params : nullptr).
	                                   complete_graph();
}

ImageFrame allocate_frame(const zimg::graph::GraphBuilder::state &state)
{
	return{
		state.width,
		state.height,
		state.type,
		state.color != zimg::graph::GraphBuilder::ColorFamily::COLOR_GREY ? 3U : 1U,
		state.color != zimg::graph::GraphBuilder::ColorFamily::COLOR_RGB,
		state.subsample_w,
		state.subsample_h
	};
}

void thread_target(const zimg::graph::FilterGraph *graph,
                   const zimg::graph::GraphBuilder::state *src_state,
                   const zimg::graph::GraphBuilder::state *dst_state,
                   unsigned times,
                   std::exception_ptr *eptr,
                   std::mutex *mutex)
{
	try {
		ImageFrame src_frame = allocate_frame(*src_state);
		ImageFrame dst_frame = allocate_frame(*dst_state);
		zimg::AlignedVector<char> tmp(graph->get_tmp_size());

		for (unsigned n = 0; n < times; ++n) {
			graph->process(src_frame.as_read_buffer(), dst_frame.as_write_buffer(), tmp.data(), nullptr, nullptr);
		}
	} catch (...) {
		std::lock_guard<std::mutex> lock{ *mutex };
		*eptr = std::current_exception();
	}
}

void execute(const JsonObject &spec, unsigned times, unsigned threads)
{
	zimg::graph::GraphBuilder::state src_state;
	zimg::graph::GraphBuilder::state dst_state;
	std::unique_ptr<zimg::graph::FilterGraph> graph = create_graph(spec, &src_state, &dst_state);

	if (!threads && !std::thread::hardware_concurrency())
		throw std::runtime_error{ "could not auto-detect CPU count" };

	unsigned thread_min = threads ? threads : 1;
	unsigned thread_max = threads ? threads : std::thread::hardware_concurrency();

	for (unsigned n = thread_min; n <= thread_max; ++n) {
		std::vector<std::thread> thread_pool;
		std::exception_ptr eptr{};
		std::mutex mutex;
		Timer timer;

		thread_pool.reserve(n);

		timer.start();
		for (unsigned nn = 0; nn < n; ++nn) {
			thread_pool.emplace_back(thread_target, graph.get(), &src_state, &dst_state, times, &eptr, &mutex);
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
};

const ArgparseOption program_switches[] = {
	{ OPTION_UINTEGER, nullptr, "times",   offsetof(Arguments, times),   nullptr, "number of benchmark cycles per thread" },
	{ OPTION_UINTEGER, nullptr, "threads", offsetof(Arguments, threads), nullptr, "number of threads" },
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING, nullptr, "specpath", offsetof(Arguments, specpath), nullptr, "graph specification file" }
};

const ArgparseCommandLine program_def = {
	program_switches,
	sizeof(program_switches) / sizeof(program_switches[0]),
	program_positional,
	sizeof(program_positional) / sizeof(program_positional[0]),
	"graph",
	"benchmark filter graph",
	nullptr
};

} // namespace


int graph_main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	args.times = 100;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)))
		return ret == ARGPARSE_HELP ? 0 : ret;

	JsonObject spec = read_graph_spec(args.specpath);
	execute(spec, args.times, args.threads);

	return 0;
}
