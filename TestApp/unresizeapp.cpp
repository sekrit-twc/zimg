#include <iostream>
#include "Common/align.h"
#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "Unresize/plane.h"
#include "Unresize/unresize.h"
#include "apps.h"
#include "argparse.h"
#include "frame.h"
#include "timer.h"

namespace {;

bool is_set_pixel_format(const zimg::PixelFormat &format)
{
	return format != zimg::PixelFormat{};
}

zimg::unresize::ImagePlane<void> frame_to_plane(ImageFrame &frame, unsigned plane)
{
	zimg::PixelFormat format = zimg::default_pixel_format(frame.pixel_type());
	format.chroma = (plane == 1 || plane == 2) && frame.is_yuv();

	auto buf = frame.as_write_buffer();
	return{ buf.data[plane], (int)frame.width(), (int)frame.height(), (int)(buf.stride[plane] / zimg::pixel_size(format.type)), format };
}

zimg::unresize::ImagePlane<const void> frame_to_plane(const ImageFrame &frame, unsigned plane)
{
	return frame_to_plane(const_cast<ImageFrame &>(frame), plane);
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
	{ OPTION_USER,     nullptr, "cpu",          offsetof(Arguments, cpu),            nullptr, "select CPU type" },
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

void execute(const zimg::unresize::Unresize &filter, const ImageFrame *src_frame, ImageFrame *dst_frame, unsigned times)
{
	zimg::PixelType pixel_type = src_frame->pixel_type();
	zimg::AlignedVector<char> tmp(filter.tmp_size(pixel_type) * zimg::pixel_size(pixel_type));

	auto exec_func = [&]
	{
		for (unsigned p = 0; p < src_frame->planes(); ++p) {
			auto src_plane = frame_to_plane(*src_frame, p);
			auto dst_plane = frame_to_plane(*dst_frame, p);

			filter.process(src_plane, dst_plane, tmp.data());
		}
	};

	auto results = measure_benchmark(times, exec_func, [](unsigned n, double d)
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
		args.working_format = zimg::default_pixel_format(zimg::PixelType::FLOAT);

	ImageFrame src_frame = imageframe::read_from_pathspec(args.inpath, "i444s", args.width_in, args.height_in, args.working_format.type, false);

	if (src_frame.subsample_w() || src_frame.subsample_h())
		throw std::logic_error{ "can only unresize greyscale/4:4:4 images" };

	ImageFrame dst_frame{ args.width_out, args.height_out, src_frame.pixel_type(), src_frame.planes(), src_frame.is_yuv() };

	zimg::unresize::Unresize filter{
		(int)src_frame.width(), (int)src_frame.height(), (int)dst_frame.width(), (int)dst_frame.height(),
		(float)args.shift_w, (float)args.shift_h, args.cpu
	};
	execute(filter, &src_frame, &dst_frame, args.times);

	if (args.visualise_path)
		imageframe::write_to_pathspec(dst_frame, args.visualise_path, "bmp", true);

	imageframe::write_to_pathspec(dst_frame, args.outpath, "i444s", false);
	return 0;
}
