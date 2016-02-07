#include <cstddef>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include "common/except.h"
#include "common/pixel.h"
#include "graph/image_filter.h"
#include "colorspace/colorspace.h"

#include "apps.h"
#include "argparse.h"
#include "frame.h"
#include "table.h"
#include "timer.h"
#include "utils.h"

namespace {

int decode_colorspace(const ArgparseOption *, void *out, int argc, char **argv)
{
	if (argc < 1)
		return -1;

	zimg::colorspace::ColorspaceDefinition *csp = static_cast<zimg::colorspace::ColorspaceDefinition *>(out);

	try {
		std::regex csp_regex{ R"(^(\w+):(\w+):(\w+)$)" };
		std::cmatch match;

		if (!std::regex_match(*argv, match, csp_regex))
			throw std::runtime_error{ "bad colorspace string" };

		csp->matrix = g_matrix_table[match[1].str().c_str()];
		csp->transfer = g_transfer_table[match[2].str().c_str()];
		csp->primaries = g_primaries_table[match[3].str().c_str()];
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return -1;
	}

	return 1;
}


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

	std::cout << "avg: " << results.first << " (" << ns_per_sample(*src_frame, results.first) << " ns/sample)\n";
	std::cout << "min: " << results.second << " (" << ns_per_sample(*src_frame, results.second) << " ns/sample)\n";
}


struct Arguments {
	const char *inpath;
	const char *outpath;
	unsigned width;
	unsigned height;
	zimg::colorspace::ColorspaceDefinition csp_in;
	zimg::colorspace::ColorspaceDefinition csp_out;
	int fullrange_in;
	int fullrange_out;
	const char *visualise_path;
	unsigned times;
	zimg::CPUClass cpu;
};

const ArgparseOption program_switches[] = {
	{ OPTION_UINTEGER, "w",     "width",     offsetof(Arguments, width),           nullptr, "image width" },
	{ OPTION_UINTEGER, "h",     "height",    offsetof(Arguments, height),          nullptr, "image height" },
	{ OPTION_FALSE,    nullptr, "tv-in",     offsetof(Arguments, fullrange_in),    nullptr, "input is TV range" },
	{ OPTION_TRUE,     nullptr, "pc-in",     offsetof(Arguments, fullrange_in),    nullptr, "input is PC range" },
	{ OPTION_FALSE,    nullptr, "tv-out",    offsetof(Arguments, fullrange_out),   nullptr, "output is TV range" },
	{ OPTION_TRUE,     nullptr, "pc-out",    offsetof(Arguments, fullrange_out),   nullptr, "output is PC range" },
	{ OPTION_STRING,   nullptr, "visualise", offsetof(Arguments, visualise_path),  nullptr, "path to BMP file for visualisation" },
	{ OPTION_UINTEGER, nullptr, "times",     offsetof(Arguments, times),           nullptr, "number of benchmark cycles" },
	{ OPTION_USER,     nullptr, "cpu",       offsetof(Arguments, cpu),             arg_decode_cpu, "select CPU type" },
};

const ArgparseOption program_positional[] = {
	{ OPTION_STRING,   nullptr,   "inpath",         offsetof(Arguments, inpath),  nullptr, "input path specifier" },
	{ OPTION_STRING,   nullptr,   "outpath",        offsetof(Arguments, outpath), nullptr, "output path specifier" },
	{ OPTION_USER,     "csp-in",  "colorspace-in",  offsetof(Arguments, csp_in),  decode_colorspace, "input colorspace specifier" },
	{ OPTION_USER,     "csp-out", "colorspace-out", offsetof(Arguments, csp_out), decode_colorspace, "output colorspace specifier" },
};

const char help_str[] =
"Colorspace specifier format: matrix:transfer:primaries\n"
"matrix:    unspec, rgb, 601, 709, 2020_ncl, 2020_cl\n"
"transfer:  unspec, linear, 709\n"
"primaries: unspec, smpte_c, 709, 2020\n"
"\n"
PATH_SPECIFIER_HELP_STR;

const ArgparseCommandLine program_def = {
	program_switches,
	sizeof(program_switches) / sizeof(program_switches[0]),
	program_positional,
	sizeof(program_positional) / sizeof(program_positional[0]),
	"colorspace",
	"convert images between colorspaces",
	help_str
};

} // namespace


int colorspace_main(int argc, char **argv)
{
	Arguments args{};
	int ret;

	args.times = 1;

	if ((ret = argparse_parse(&program_def, &args, argc, argv)))
		return ret == ARGPARSE_HELP ? 0 : ret;

	bool yuv_in = args.csp_in.matrix != zimg::colorspace::MatrixCoefficients::MATRIX_RGB;
	bool yuv_out = args.csp_out.matrix != zimg::colorspace::MatrixCoefficients::MATRIX_RGB;

	try {
		ImageFrame src_frame = imageframe::read(args.inpath, "i444s", args.width, args.height, zimg::PixelType::FLOAT, !!args.fullrange_in);
		ImageFrame dst_frame{ src_frame.width(), src_frame.height(), zimg::PixelType::FLOAT, 3, yuv_out };

		if (src_frame.is_yuv() != yuv_in)
			std::cerr << "warning: input file is of different color family than declared format\n";

		auto convert = zimg::colorspace::ColorspaceConversion{ src_frame.width(), src_frame.height() }.
			set_csp_in(args.csp_in).
			set_csp_out(args.csp_out).
			set_cpu(args.cpu).
			create();

		execute(convert.get(), &src_frame, &dst_frame, args.times);

		if (args.visualise_path)
			imageframe::write(dst_frame, args.visualise_path, "bmp", true);

		imageframe::write(dst_frame, args.outpath, "i444s", !!args.fullrange_out);
	} catch (const zimg::error::Exception &e) {
		std::cerr << e.what() << '\n';
		return 2;
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 2;
	}

	return 0;
}
