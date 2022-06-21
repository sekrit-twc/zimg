#include <cstring>
#include <iostream>
#include <memory>
#include <regex>
#include <utility>
#include "common/except.h"
#include "common/pixel.h"
#include "depth/depth.h"
#include "graphengine/filter.h"

#include "apps.h"
#include "argparse.h"
#include "frame.h"
#include "table.h"
#include "timer.h"
#include "utils.h"

namespace {

int decode_dither(const struct ArgparseOption *, void *out, const char *param, int)
{
	try {
		zimg::depth::DitherType *dither = static_cast<zimg::depth::DitherType *>(out);
		*dither = g_dither_table[param];
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return -1;
	}

	return 0;
}

int decode_force_rgb_yuv(const struct ArgparseOption *opt, void *out, const char *, int negated)
{
	int *family = static_cast<int *>(out);

	if (negated) {
		std::cerr << "argument '" << opt->long_name << "' can not be negated\n";
		return 1;
	}

	if (!strcmp(opt->long_name, "yuv"))
		*family = 1;
	else
		*family = 2;

	return 0;
}


double ns_per_sample(const ImageFrame &frame, double seconds)
{
	double samples = static_cast<double>(static_cast<size_t>(frame.width()) * frame.height() * frame.planes());
	return seconds * 1e9 / samples;
}

void execute(const std::vector<std::pair<int, const graphengine::Filter *>> &filters, const ImageFrame *src_frame, ImageFrame *dst_frame, unsigned times)
{
	auto results = measure_benchmark(times, FilterExecutor{ filters, src_frame, dst_frame }, [](unsigned n, double d)
	{
		std::cout << '#' << n << ": " << d << '\n';
	});

	std::cout << "avg: " << results.first << " (" << ns_per_sample(*src_frame, results.first) << " ns/sample)\n";
	std::cout << "min: " << results.second << " (" << ns_per_sample(*src_frame, results.second) << " ns/sample)\n";
}


struct Arguments {
	const char *inpath;
	const char *outpath;
	unsigned width;
	unsigned height;
	zimg::depth::DitherType dither;
	zimg::PixelFormat pixel_in;
	zimg::PixelFormat pixel_out;
	int force_color_family;
	const char *visualise_path;
	unsigned times;
	zimg::CPUClass cpu;
};

const ArgparseOption program_switches[] = {
	{ OPTION_UINT,   "w",     "width",      offsetof(Arguments, width),              nullptr, "image width" },
	{ OPTION_UINT,   "h",     "height",     offsetof(Arguments, height),             nullptr, "image height"},
	{ OPTION_USER1,  nullptr, "dither",     offsetof(Arguments, dither),             decode_dither, "select dithering method" },
	{ OPTION_USER0,  nullptr, "yuv",        offsetof(Arguments, force_color_family), decode_force_rgb_yuv, "interpret RGB image as YUV" },
	{ OPTION_USER0,  nullptr, "rgb",        offsetof(Arguments, force_color_family), decode_force_rgb_yuv, "interpret YUV image as RGB"},
	{ OPTION_STRING, nullptr, "visualise",  offsetof(Arguments, visualise_path),     nullptr, "path to BMP file for visualisation"},
	{ OPTION_UINT,   nullptr, "times",      offsetof(Arguments, times),              nullptr, "number of benchmark cycles"},
	{ OPTION_USER1,  nullptr, "cpu",        offsetof(Arguments, cpu),                arg_decode_cpu, "select CPU type"},
	{ OPTION_NULL }
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING, nullptr, "inpath",     offsetof(Arguments, inpath),     nullptr, "input path specifier" },
	{ OPTION_STRING, nullptr, "outpath",    offsetof(Arguments, outpath),    nullptr, "output path specifier" },
	{ OPTION_USER1,  nullptr, "format-in",  offsetof(Arguments, pixel_in),  arg_decode_pixfmt, "input pixel format"},
	{ OPTION_USER1,  nullptr, "format-out", offsetof(Arguments, pixel_out), arg_decode_pixfmt, "output pixel format"},
	{ OPTION_NULL }
};

const char help_str[] =
"Dithering methods: none, ordered, random, error_diffusion\n"
"\n"
PIXFMT_SPECIFIER_HELP_STR
"\n"
PATH_SPECIFIER_HELP_STR;

const ArgparseCommandLine program_def = { program_switches, program_positional, "depth", "convert images between pixel formats", help_str };

} // namespace


int depth_main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	args.times = 1;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)) < 0)
		return ret == ARGPARSE_HELP_MESSAGE ? 0 : ret;

	try {
		ImageFrame src_frame = imageframe::read(args.inpath, "i444", args.width, args.height);

		bool is_yuv;
		if (args.force_color_family == 1)
			is_yuv = true;
		else if (args.force_color_family == 2)
			is_yuv = false;
		else
			is_yuv = src_frame.is_yuv();

		if (src_frame.is_yuv() != is_yuv)
			std::cerr << "warning: input file is of different color family than declared format\n";
		if (src_frame.pixel_type() != args.pixel_in.type)
			std::cerr << "warning: input file is of a different pixel type than declared format\n";

		if (zimg::pixel_size(src_frame.pixel_type()) != zimg::pixel_size(args.pixel_in.type))
			throw std::runtime_error{ "pixel sizes not compatible" };

		ImageFrame dst_frame{
			src_frame.width(), src_frame.height(), args.pixel_out.type, src_frame.planes(),
			is_yuv, src_frame.subsample_w(), src_frame.subsample_h()
		};

		auto conv = zimg::depth::DepthConversion{ src_frame.width(), src_frame.height() }
			.set_pixel_in(args.pixel_in)
			.set_pixel_out(args.pixel_out)
			.set_dither_type(args.dither)
			.set_cpu(args.cpu)
			.set_planes({ true, false, false, false });

		zimg::depth::DepthConversion::result luma_result = conv.create();
		zimg::depth::DepthConversion::result chroma_result;
		std::vector<std::pair<int, const graphengine::Filter *>> filters;
		
		if (luma_result.filter_refs[0])
			filters.push_back({ 0, luma_result.filter_refs[0] });

		if (src_frame.planes() >= 3) {
			zimg::PixelFormat format_in_uv = args.pixel_in;
			zimg::PixelFormat format_out_uv = args.pixel_out;

			format_in_uv.chroma = is_yuv;
			format_out_uv.chroma = is_yuv;

			chroma_result = conv.set_pixel_in(format_in_uv)
				.set_pixel_out(format_out_uv)
				.set_planes({ false, true, true, false })
				.create();

			if (chroma_result.filter_refs[1])
				filters.push_back({ 1, chroma_result.filter_refs[1] });
			if (chroma_result.filter_refs[2])
				filters.push_back({ 2, chroma_result.filter_refs[2] });
		}

		execute(filters, &src_frame, &dst_frame, args.times);

		if (args.visualise_path)
			imageframe::write(dst_frame, args.visualise_path, "bmp", args.pixel_out.depth, true);

		imageframe::write(dst_frame, args.outpath, "i444", args.pixel_out.fullrange);
	} catch (const zimg::error::Exception &e) {
		std::cerr << e.what() << '\n';
		return 2;
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 2;
	}

	return 0;
}
