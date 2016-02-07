#include <iostream>
#include <memory>
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "unresize/unresize.h"

#include "apps.h"
#include "argparse.h"
#include "frame.h"
#include "pair_filter.h"
#include "timer.h"
#include "utils.h"

namespace {

bool is_set_pixel_format(const zimg::PixelFormat &format)
{
	return format != zimg::PixelFormat{};
}


struct Arguments {
	const char *inpath;
	const char *outpath;
	unsigned width_in;
	unsigned height_in;
	unsigned width_out;
	unsigned height_out;
	double shift_w;
	double shift_h;
	zimg::PixelFormat working_format;
	const char *visualise_path;
	unsigned times;
	zimg::CPUClass cpu;
};

const ArgparseOption program_switches[] = {
	{ OPTION_UINTEGER, "w",     "width-in",     offsetof(Arguments, width_in),       nullptr, "image width" },
	{ OPTION_UINTEGER, "h",     "height-in",    offsetof(Arguments, height_in),      nullptr, "image height"},
	{ OPTION_FLOAT,    nullptr, "shift-w",      offsetof(Arguments, shift_w),        nullptr, "subpixel shift" },
	{ OPTION_FLOAT,    nullptr, "shift-h",      offsetof(Arguments, shift_h),        nullptr, "subpixel shift" },
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

const ArgparseCommandLine program_def = {
	program_switches,
	sizeof(program_switches) / sizeof(program_switches[0]),
	program_positional,
	sizeof(program_positional) / sizeof(program_positional[0]),
	"unresize",
	"unresize images",
	PIXFMT_SPECIFIER_HELP_STR "\n" PATH_SPECIFIER_HELP_STR
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


int unresize_main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	args.times = 1;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)))
		return ret == ARGPARSE_HELP ? 0 : ret;

	if (!is_set_pixel_format(args.working_format))
		args.working_format = zimg::PixelType::FLOAT;

	try {
		ImageFrame src_frame = imageframe::read(args.inpath, "i444s", args.width_in, args.height_in, args.working_format.type, false);

		if (src_frame.subsample_w() || src_frame.subsample_h())
			throw std::runtime_error{ "can only unresize greyscale/4:4:4 images" };

		ImageFrame dst_frame{ args.width_out, args.height_out, src_frame.pixel_type(), src_frame.planes(), src_frame.is_yuv() };

		auto filter_pair = zimg::unresize::UnresizeConversion{ src_frame.width(), src_frame.height(), src_frame.pixel_type() }.
			set_orig_width(dst_frame.width()).
			set_orig_height(dst_frame.height()).
			set_shift_w(args.shift_w).
			set_shift_h(args.shift_h).
			set_cpu(args.cpu).
			create();

		if (filter_pair.second)
			filter_pair.first = ztd::make_unique<PairFilter>(std::move(filter_pair.first), std::move(filter_pair.second));

		execute(filter_pair.first.get(), &src_frame, &dst_frame, args.times);

		if (args.visualise_path)
			imageframe::write(dst_frame, args.visualise_path, "bmp", true);

		imageframe::write(dst_frame, args.outpath, "i444s", false);
	} catch (const zimg::error::Exception &e) {
		std::cerr << e.what() << '\n';
		return 2;
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 2;
	}

	return 0;
}
