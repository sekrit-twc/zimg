#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Resize/filter.h"
#include "Resize/resize.h"
#include "apps.h"
#include "argparse.h"
#include "frame.h"
#include "pair_filter.h"
#include "timer.h"
#include "utils.h"

namespace {;

bool is_set_pixel_format(const zimg::PixelFormat &format)
{
	return format != zimg::PixelFormat{};
}

zimg::resize::Filter *create_filter(const char *filter, double param_a, double param_b)
{
	if (!strcmp(filter, "point"))
		return new zimg::resize::PointFilter{};
	else if (!strcmp(filter, "bilinear"))
		return new zimg::resize::BilinearFilter{};
	else if (!strcmp(filter, "bicubic"))
		return new zimg::resize::BicubicFilter{
			std::isnan(param_a) ? 1.0 / 3.0 : param_a,
			std::isnan(param_a) ? 1.0 / 3.0 : param_b
	};
	else if (!strcmp(filter, "spline16"))
		return new zimg::resize::Spline16Filter{};
	else if (!strcmp(filter, "spline36"))
		return new zimg::resize::Spline36Filter{};
	else if (!strcmp(filter, "lanczos"))
		return new zimg::resize::LanczosFilter{ std::isnan(param_a) ? 4 : (int)param_a };
	else
		return nullptr;
}

int decode_filter(const ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	auto *filter = reinterpret_cast<std::unique_ptr<zimg::resize::Filter> *>(out);

	try {
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

		filter->reset(create_filter(filter_str.c_str(), param_a, param_b));
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

void execute(const zimg::IZimgFilter *filter, const ImageFrame *src_frame, ImageFrame *dst_frame, unsigned times)
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

	args.filter.reset(create_filter("bicubic", NAN, NAN));
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
		args.working_format = zimg::default_pixel_format(zimg::PixelType::FLOAT);

	ImageFrame src_frame = imageframe::read_from_pathspec(args.inpath, "i444s", args.width_in, args.height_in, args.working_format.type, false);

	if (std::isnan(args.subwidth))
		args.subwidth = src_frame.width();
	if (std::isnan(args.subheight))
		args.subheight = src_frame.height();

	if (src_frame.subsample_w() || src_frame.subsample_h())
		throw std::logic_error{ "can only resize greyscale/4:4:4 images" };

	ImageFrame dst_frame{ args.width_out, args.height_out, src_frame.pixel_type(), src_frame.planes(), src_frame.is_yuv() };

	auto filter_pair = zimg::resize::create_resize(*args.filter, src_frame.pixel_type(), args.working_format.depth,
	                                               src_frame.width(), src_frame.height(), dst_frame.width(), dst_frame.height(),
	                                               args.shift_w, args.shift_h, args.subwidth, args.subheight, args.cpu);

	std::unique_ptr<zimg::IZimgFilter> filter_a{ filter_pair.first };
	std::unique_ptr<zimg::IZimgFilter> filter_b{ filter_pair.second };

	if (filter_b) {
		std::unique_ptr<zimg::IZimgFilter> pair{ new PairFilter{ filter_a.get(), filter_b.get() } };
		filter_a.release();
		filter_b.release();
		filter_a = std::move(pair);
	}

	execute(filter_a.get(), &src_frame, &dst_frame, args.times);

	if (args.visualise_path)
		imageframe::write_to_pathspec(dst_frame, args.visualise_path, "bmp", true);

	imageframe::write_to_pathspec(dst_frame, args.outpath, "i444s", false);
	return 0;
}
