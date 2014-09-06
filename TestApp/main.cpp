#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include "align.h"
#include "bitmap.h"
#include "filter.h"
#include "resize.h"
#include "timer.h"

using namespace resize;

namespace {;

Filter *select_filter(const char *filter)
{
	if (!strcmp(filter, "point"))
		return new PointFilter{};
	else if (!strcmp(filter, "bilinear"))
		return new BilinearFilter{};
	else if (!strcmp(filter, "bicubic"))
		return new BicubicFilter(1.0 / 3.0, 1.0 / 3.0);
	else if (!strcmp(filter, "lanczos"))
		return new LanczosFilter{ 4 };
	else if (!strcmp(filter, "spline16"))
		return new Spline16Filter{};
	else if (!strcmp(filter, "spline36"))
		return new Spline36Filter{};
	else
		return new BilinearFilter{};
}

void load_plane_u8(const uint8_t *src, float *dst, int width, int height, int src_stride, int dst_stride)
{
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			dst[i * dst_stride + j] = (float)src[i * src_stride + j] / (float)UINT8_MAX;
		}
	}
}

void store_plane_u8(const float *src, uint8_t *dst, int width, int height, int src_stride, int dst_stride)
{
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			float x = std::round(src[i * src_stride + j] * (float)UINT8_MAX);
			dst[i * dst_stride + j] = (uint8_t)std::min(std::max(x, 0.f), (float)UINT8_MAX);
		}
	}
}

void usage()
{
	std::cout << "TestApp infile outfile w h [--filter filter] [--shift-w shift] [--shift-h shift] [--sub-w w] [--sub-h h] [--times n] [--x86 | --no-x86]";
	std::cout << "    infile              input BMP file\n";
	std::cout << "    outfile             output BMP file\n";
	std::cout << "    width               output width\n";
	std::cout << "    height              output height\n";
	std::cout << "    --filter            resampling filter\n";
	std::cout << "    --shift-w           horizontal shift\n";
	std::cout << "    --shift-h           vertical shift\n";
	std::cout << "    --sub-w             subwindow width\n";
	std::cout << "    --sub-h             subwindow height\n";
	std::cout << "    --times             number of cycles\n";
	std::cout << "    --x86 / --no-x86    toggle x86 optimizations\n";
}

} // namespace


int main(int argc, const char **argv)
{
	if (argc < 5) {
		usage();
		return -1;
	}

	const char *ifile = argv[1];
	const char *ofile = argv[2];
	int width = std::atoi(argv[3]);
	int height = std::atoi(argv[4]);
	const char *filter_str = "bilinear";
	double shift_w = 0.0;
	double shift_h = 0.0;
	double sub_w = -1.0;
	double sub_h = -1.0;
	int times = 1;
	bool x86 = false;

	for (int i = 5; i < argc; ++i) {
		if (!strcmp(argv[i], "--filter") && i + 1 < argc) {
			filter_str = argv[i + 1];
			++i;
		} else if (!strcmp(argv[i], "--shift-w") && i + 1 < argc) {
			shift_w = std::atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--shift-h") && i + 1 < argc) {
			shift_h = std::atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--sub-w") && i + 1 < argc) {
			sub_w = std::atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--sub-h") && i + 1 < argc) {
			sub_h = std::atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--times") && i + 1 < argc) {
			times = std::atoi(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--x86")) {
			x86 = true;
		} else if (!strcmp(argv[i], "--no-x86")) {
			x86 = false;
		} else {
			std::cerr << "unknown argument: " << argv[i] << '\n';
		}
	}

	try {
		Bitmap in = read_bitmap(ifile);
		Bitmap out{ width, height, in.planes() == 4 };

		if (sub_w < 0.0)
			sub_w = in.width();
		if (sub_h < 0.0)
			sub_h = in.height();

		int in_stride = align(in.width(), 8);
		int out_stride = align(width, 8);

		AlignedVector<float> in_plane(in_stride * in.height() * in.planes());
		AlignedVector<float> out_plane(out_stride * height * in.planes());

		for (int p = 0; p < in.planes(); ++p) {
			load_plane_u8(in.data(p), in_plane.data() + in_stride * in.height() * p, in.width(), in.height(), in.stride(), in_stride);
		}

		std::unique_ptr<Filter> filter{ select_filter(filter_str) };
		Resize resize{ *filter, in.width(), in.height(), width, height, shift_w, shift_h, sub_w, sub_h, x86 };

		AlignedVector<float> tmp(resize.tmp_size());

		double min_time = INFINITY;
		double avg_time = 0.0;
		for (int i = 0; i < times; ++i) {
			Timer watch;
			double elapsed = 0.0;

			watch.start();
			for (int p = 0; p < in.planes(); ++p) {
				resize.process(in_plane.data() + in_stride * in.height() * p, out_plane.data() + out_stride * height * p, tmp.data(), in_stride, out_stride);
			}
			watch.stop();
			elapsed = watch.elapsed();

			std::cout << '#' << i << ": " << elapsed << '\n';
			avg_time += elapsed / times;
			min_time = std::min(elapsed, min_time);
		}
		std::cout << "average: " << avg_time << '\n';
		std::cout << "min: " << min_time << '\n';

		for (int p = 0; p < in.planes(); ++p) {
			store_plane_u8(out_plane.data() + out_stride * height * p, out.data(p), width, height, out_stride, out.stride());
		}

		write_bitmap(out, ofile);
	} catch (std::exception &e) {
		std::cerr << e.what() << '\n';
		return -1;
	}

	return 0;
}
