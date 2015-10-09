#include <cstring>
#include <iostream>
#include <memory>
#include <regex>
#include "common/cpuinfo.h"
#include "common/pixel.h"
#include "common/static_map.h"
#include "graph/image_filter.h"
#include "depth/depth.h"
#include "apps.h"
#include "argparse.h"
#include "frame.h"
#include "timer.h"
#include "utils.h"

namespace {;

zimg::depth::DitherType lookup_dither(const char *dither)
{
	using zimg::depth::DitherType;

	static const zimg::static_string_map<DitherType, 4> map{
		{ "none",            DitherType::DITHER_NONE },
		{ "ordered",         DitherType::DITHER_ORDERED },
		{ "random",          DitherType::DITHER_RANDOM },
		{ "error_diffusion", DitherType::DITHER_ERROR_DIFFUSION },
	};
	auto it = map.find(dither);
	return it == map.end() ? throw std::invalid_argument{ "bad dither type" } : it->second;
}

int decode_dither(const ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	zimg::depth::DitherType *dither = static_cast<zimg::depth::DitherType *>(out);

	try {
		*dither = lookup_dither(*argv);
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return -1;
	}

	return 1;
}

int decode_force_rgb_yuv(const ArgparseOption *opt, void *out, int, char **)
{
	int *family = static_cast<int *>(out);

	if (!strcmp(opt->long_name, "yuv"))
		*family = 1;
	else
		*family = 2;

	return 0;
}


double ns_per_sample(const ImageFrame &frame, double seconds)
{
	double samples = static_cast<double>(static_cast<size_t>(frame.width()) * frame.height() * 3);
	return seconds * 1e9 / samples;
}

void execute(const zimg::graph::ImageFilter *filter, const zimg::graph::ImageFilter *filter_uv, const ImageFrame *src_frame, ImageFrame *dst_frame, unsigned times)
{
	auto results = measure_benchmark(times, FilterExecutor{ filter, filter_uv, src_frame, dst_frame }, [](unsigned n, double d)
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
	zimg::PixelFormat format_in;
	zimg::PixelFormat format_out;
	int force_color_family;
	const char *visualise_path;
	unsigned times;
	zimg::CPUClass cpu;
};

const ArgparseOption program_switches[] = {
	{ OPTION_UINTEGER, "w",     "width",      offsetof(Arguments, width),              nullptr, "image width" },
	{ OPTION_UINTEGER, "h",     "height",     offsetof(Arguments, height),             nullptr, "image height"},
	{ OPTION_USER,     nullptr, "dither",     offsetof(Arguments, dither),             decode_dither, "select dithering method" },
	{ OPTION_USER,     nullptr, "yuv",        offsetof(Arguments, force_color_family), decode_force_rgb_yuv, "interpret RGB image as YUV" },
	{ OPTION_USER,     nullptr, "rgb",        offsetof(Arguments, force_color_family), decode_force_rgb_yuv, "interpret YUV image as RGB"},
	{ OPTION_STRING,   nullptr, "visualise",  offsetof(Arguments, visualise_path),     nullptr, "path to BMP file for visualisation"},
	{ OPTION_UINTEGER, nullptr, "times",      offsetof(Arguments, times),              nullptr, "number of benchmark cycles"},
	{ OPTION_USER,     nullptr, "cpu",        offsetof(Arguments, cpu),                arg_decode_cpu, "select CPU type"},
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING, nullptr, "inpath",     offsetof(Arguments, inpath),     nullptr, "input path specifier" },
	{ OPTION_STRING, nullptr, "outpath",    offsetof(Arguments, outpath),    nullptr, "output path specifier" },
	{ OPTION_USER,   nullptr, "format-in",  offsetof(Arguments, format_in),  arg_decode_pixfmt, "input pixel format"},
	{ OPTION_USER,   nullptr, "format-out", offsetof(Arguments, format_out), arg_decode_pixfmt, "output pixel format"},
};

const char help_str[] =
"Dithering methods: none, ordered, random, error_diffusion\n"
"\n"
PIXFMT_SPECIFIER_HELP_STR
"\n"
PATH_SPECIFIER_HELP_STR;

const ArgparseCommandLine program_def = {
	program_switches,
	sizeof(program_switches) / sizeof(program_switches[0]),
	program_positional,
	sizeof(program_positional) / sizeof(program_positional[0]),
	"depth",
	"convert images between pixel formats",
	help_str
};

} // namespace


int depth_main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	args.times = 1;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)))
		return ret == ARGPARSE_HELP ? 0 : ret;

	ImageFrame src_frame = imageframe::read_from_pathspec(args.inpath, "i444", args.width, args.height);

	bool is_yuv;
	if (args.force_color_family == 1)
		is_yuv = true;
	else if (args.force_color_family == 2)
		is_yuv = false;
	else
		is_yuv = src_frame.is_yuv();

	if (src_frame.is_yuv() != is_yuv)
		std::cerr << "warning: input file is of different color family than declared format\n";
	if (src_frame.pixel_type() != args.format_in.type)
		std::cerr << "warning: input file is of a different pixel type than declared format\n";

	if (zimg::pixel_size(src_frame.pixel_type()) != zimg::pixel_size(args.format_in.type))
		throw std::logic_error{ "pixel sizes not compatible" };

	ImageFrame dst_frame{ src_frame.width(), src_frame.height(), args.format_out.type, src_frame.planes(), is_yuv, src_frame.subsample_w(), src_frame.subsample_h() };

	std::unique_ptr<zimg::graph::ImageFilter> filter;
	std::unique_ptr<zimg::graph::ImageFilter> filter_uv;

	auto conv = zimg::depth::DepthConversion{ src_frame.width(), src_frame.height() }.
		set_pixel_in(args.format_in).
		set_pixel_out(args.format_out).
		set_dither_type(args.dither).
		set_cpu(args.cpu);

	filter = conv.create();

	if (src_frame.planes() >= 3 && is_yuv) {
		zimg::PixelFormat format_in_uv = args.format_in;
		zimg::PixelFormat format_out_uv = args.format_out;

		format_in_uv.chroma = true;
		format_out_uv.chroma = true;

		filter_uv = conv.set_pixel_in(format_in_uv).set_pixel_out(format_out_uv).create();
	}

	execute(filter.get(), filter_uv.get(), &src_frame, &dst_frame, args.times);

	if (args.visualise_path)
		imageframe::write_to_pathspec(dst_frame, args.visualise_path, "bmp", args.format_out.depth, true);

	imageframe::write_to_pathspec(dst_frame, args.outpath, "i444", args.format_out.fullrange);
	return 0;
}
