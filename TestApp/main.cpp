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

uint8_t clamp_float(float x)
{
	return (uint8_t)std::min(std::max(x, 0.f), (float)UINT8_MAX);
}

} // namespace


int main(int argc, const char **argv)
{
	if (argc < 5) {
		printf("%s: infile outfile width height [filter] [shift_w] [shift_h] [subwidth] [subheight]\n", argv[0]);
		return -1;
	}

	const char *infile = argv[1];
	const char *outfile = argv[2];
	int width = std::atoi(argv[3]);
	int height = std::atoi(argv[4]);
	std::unique_ptr<Filter> filter{ select_filter(argc > 5 ? argv[5] : "bilinear") };
	double shift_w = argc > 6 ? std::atof(argv[6]) : 0.0;
	double shift_h = argc > 7 ? std::atof(argv[7]) : 0.0;

	try {
		Bitmap in = read_bitmap(infile);
		Bitmap out{ width, height, in.planes() == 4 };

		double subwidth = argc > 8 ? std::atof(argv[8]) : in.width();
		double subheight = argc > 9 ? std::atof(argv[9]) : in.height();

		Resize resize{ *filter, in.width(), in.height(), width, height, shift_w, shift_h, subwidth, subheight, false };

		int src_stride = align(in.width(), 8);
		int dst_stride = align(width, 8);

		AlignedVector<float> src_image(src_stride * in.height());
		AlignedVector<float> dst_image(dst_stride * height);

		AlignedVector<float> tmp(resize.tmp_size());
		Timer t;

		t.start();
		for (int p = 0; p < in.planes(); ++p) {
			for (int i = 0; i < in.height(); ++i) {
				for (int j = 0; j < in.width(); ++j) {
					src_image[i * src_stride + j] = in.data(p)[i * in.stride() + j];
				}
			}

			resize.process(src_image.data(), dst_image.data(), tmp.data(), src_stride, dst_stride);

			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j) {
					out.data(p)[i * out.stride() + j] = clamp_float(dst_image[i * dst_stride + j]);
				}
			}
		}
		t.stop();

		std::cout << "elapsed: " << t.elapsed() << '\n';

		write_bitmap(out, outfile);
	} catch (std::exception &e) {
		fprintf(stderr, "exception: %s\n", e.what());
		return -1;
	}

	return 0;
}
