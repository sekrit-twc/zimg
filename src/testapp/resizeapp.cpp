#include <cmath>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include "common/cpuinfo.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "resize/filter.h"
#include "resize/resize.h"

#include "apps.h"
#include "argparse.h"
#include "frame.h"
#include "pair_filter.h"
#include "table.h"
#include "timer.h"
#include "utils.h"

namespace {;

bool is_set_pixel_format(const zimg::PixelFormat &format)
{
	return format != zimg::PixelFormat{};
}

int decode_filter(const ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	try {
		std::unique_ptr<zimg::resize::Filter> *filter = static_cast<std::unique_ptr<zimg::resize::Filter> *>(out);
		std::regex filter_regex{ R"(^(point|bilinear|bicubic|spline16|spline36|lanczos)(?::([\w.+-]+)(?::([\w.+-]+))?)?$)" };
		std::cmatch match;
		std::string filter_str;
		double param_a = NAN;
		double param_b = NAN;

		if (!std::regex_match(*argv, match, filter_regex))
			throw std::runtime_error{ "bad filter string" };

		filter_str = match[1];

		if (match.size() >= 2 && match[2].length())
			param_a = std::stod(match[2]);
		if (match.size() >= 3 && match[3].length())
			param_b = std::stod(match[3]);

		*filter = g_resize_table[filter_str.c_str()](param_a, param_b);
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return -1;
	}

	return 1;
}


struct Arguments {
	const char *inpath;
	const char *outpath;
	unsigned width_in;
	unsigned height_in;
	unsigned width_out;
	unsigned height_out;
	std::unique_ptr<zimg::resize::Filter> filter;
	double param_a;
	double param_b;
	double shift_w;
	double shift_h;
	double subwidth;
	double subheight;
	zimg::PixelFormat working_format;
	const char *visualise_path;
	unsigned times;
	zimg::CPUClass cpu;
};

const ArgparseOption program_switches[] = {
	{ OPTION_UINTEGER, "w",     "width-in",     offsetof(Arguments, width_in),       nullptr, "image width" },
	{ OPTION_UINTEGER, "h",     "height-in",    offsetof(Arguments, height_in),      nullptr, "image height"},
	{ OPTION_USER,     nullptr, "filter",       offsetof(Arguments, filter),         decode_filter, "select resampling filter" },
	{ OPTION_FLOAT,    nullptr, "shift-w",      offsetof(Arguments, shift_w),        nullptr, "subpixel shift" },
	{ OPTION_FLOAT,    nullptr, "shift-h",      offsetof(Arguments, shift_h),        nullptr, "subpixel shift" },
	{ OPTION_FLOAT,    nullptr, "sub-width",    offsetof(Arguments, subwidth),       nullptr, "active image width" },
	{ OPTION_FLOAT,    nullptr, "sub-height",   offsetof(Arguments, subheight),      nullptr, "active image height" },
	{ OPTION_USER,     nullptr, "format",       offsetof(Arguments, working_format), arg_decode_pixfmt, "working pixel format" },
	{ OPTION_STRING,   nullptr, "visualise",    offsetof(Arguments, visualise_path), nullptr, "path to BMP file for visualisation" },
	{ OPTION_UINTEGER, nullptr, "times",        offsetof(Arguments, times),          nullptr, "number of benchmark cycles" },
	{ OPTION_USER,     nullptr, "cpu",          offsetof(Arguments, cpu),            arg_decode_cpu, "select CPU type" },
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING,   nullptr, "inpath",     offsetof(Arguments, inpath),     nullptr, "input path specifier" },
	{ OPTION_STRING,   nullptr, "outpath",    offsetof(Arguments, outpath),    nullptr, "output path specifier" },
	{ OPTION_UINTEGER, nullptr, "width-out",  offsetof(Arguments, width_out),  nullptr, "output width" },
	{ OPTION_UINTEGER, nullptr, "height-out", offsetof(Arguments, height_out), nullptr, "output height" },
};

const char help_str[] =
"Resampling filter specifier: filter[:param_a[:param_b]]\n"
"filter: point, bilinear, bicubic, spline16, spline36, lanczos\n"
"\n"
PIXFMT_SPECIFIER_HELP_STR
"\n"
PATH_SPECIFIER_HELP_STR;

const ArgparseCommandLine program_def = {
	program_switches,
	sizeof(program_switches) / sizeof(program_switches[0]),
	program_positional,
	sizeof(program_positional) / sizeof(program_positional[0]),
	"resize",
	"resize images",
	help_str
};

double ns_per_sample(const ImageFrame &frame, double seconds)
{
	double samples = static_cast<double>(static_cast<size_t>(frame.width()) * frame.height() * 3);
	return seconds * 1e9 / samples;
}

void execute(const zimg::graph::ImageFilter *filter, const ImageFrame *src_frame, ImageFrame *dst_frame, unsigned times)
{
	auto results = measure_benchmark(times, FilterExecutor{ filter, nullptr, src_frame, dst_frame }, [](unsigned n, double d)
	{
		std::cout << '#' << n << ": " << d << '\n';
	});

	std::cout << "avg: " << results.first << " (" << ns_per_sample(*dst_frame, results.first) << " ns/sample)\n";
	std::cout << "min: " << results.second << " (" << ns_per_sample(*dst_frame, results.second) << " ns/sample)\n";
}

} // namespace


int resize_main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	args.filter = g_resize_table["bicubic"](NAN, NAN);
	args.param_a = NAN;
	args.param_b = NAN;
	args.shift_w = NAN;
	args.shift_h = NAN;
	args.subwidth = NAN;
	args.subheight = NAN;
	args.times = 1;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)))
		return ret == ARGPARSE_HELP ? 0 : ret;

	if (std::isnan(args.shift_w))
		args.shift_w = 0.0;
	if (std::isnan(args.shift_h))
		args.shift_h = 0.0;
	if (!is_set_pixel_format(args.working_format))
		args.working_format = zimg::PixelType::FLOAT;

	ImageFrame src_frame = imageframe::read_from_pathspec(args.inpath, "i444s", args.width_in, args.height_in, args.working_format.type, false);

	if (std::isnan(args.subwidth))
		args.subwidth = src_frame.width();
	if (std::isnan(args.subheight))
		args.subheight = src_frame.height();

	if (src_frame.subsample_w() || src_frame.subsample_h())
		throw std::logic_error{ "can only resize greyscale/4:4:4 images" };

	ImageFrame dst_frame{ args.width_out, args.height_out, src_frame.pixel_type(), src_frame.planes(), src_frame.is_yuv() };

	auto filter_pair = zimg::resize::ResizeConversion{ src_frame.width(), src_frame.height(), src_frame.pixel_type() }.
		set_depth(args.working_format.depth).
		set_filter(args.filter.get()).
		set_dst_width(dst_frame.width()).
		set_dst_height(dst_frame.height()).
		set_shift_w(args.shift_w).
		set_shift_h(args.shift_h).
		set_subwidth(args.subwidth).
		set_subheight(args.subheight).
		set_cpu(args.cpu).
		create();

	if (filter_pair.second)
		filter_pair.first = ztd::make_unique<PairFilter>(std::move(filter_pair.first), std::move(filter_pair.second));

	execute(filter_pair.first.get(), &src_frame, &dst_frame, args.times);

	if (args.visualise_path)
		imageframe::write_to_pathspec(dst_frame, args.visualise_path, "bmp", true);

	imageframe::write_to_pathspec(dst_frame, args.outpath, "i444s", false);
	return 0;
}
