#include <cstdlib>
#include <cstring>
#include <iostream>
#include "Unresize/unresize.h"
#include "apps.h"
#include "bitmap.h"

using namespace zimg;

namespace {;

void process(Bitmap &dst, const Bitmap &src, float shift_w, float shift_h, int times, bool x86)
{
	unresize::Unresize u(src.width(), src.height(), dst.width(), dst.height(), shift_w, shift_h, x86);

	int src_stride = width_to_stride(src.width(), PixelType::BYTE);
	int dst_stride = width_to_stride(dst.width(), PixelType::BYTE);
	int src_plane_size = src_stride * src.height();
	int dst_plane_size = dst_stride * dst.height();

	auto src_planes = allocate_frame(src_stride, src.height(), 3, PixelType::BYTE);
	auto dst_planes = allocate_frame(dst_stride, dst.height(), 3, PixelType::BYTE);
	auto tmp = allocate_buffer(u.tmp_size(PixelType::BYTE, PixelType::BYTE), PixelType::FLOAT);

	for (int p = 0; p < 3; ++p) {
		const uint8_t *src_p = src.data(p);
		uint8_t *src_plane_p = (uint8_t *)(src_planes.data() + p * src_plane_size);

		for (int i = 0; i < src.height(); ++i) {
			std::copy(src_p, src_p + src.width(), src_plane_p);
			src_p += src.stride();
			src_plane_p += src_stride;
		}
	}

	measure_time(times, [&]()
	{
		for (int p = 0; p < 3; ++p) {
			const uint8_t *src_p = (const uint8_t *)(src_planes.data() + p * src_plane_size);
			uint8_t *dst_p = (uint8_t *)(dst_planes.data() + p * dst_plane_size);

			u.process(src_p, dst_p, (float *)tmp.data(), src_stride, dst_stride, PixelType::BYTE, PixelType::BYTE);
		}
	});

	for (int p = 0; p < 3; ++p) {
		const uint8_t *dst_plane_p = (const uint8_t *)(dst_planes.data() + p * dst_plane_size);
		uint8_t *dst_p = dst.data(p);

		for (int i = 0; i < dst.height(); ++i) {
			std::copy(dst_plane_p, dst_plane_p + dst.width(), dst_p);
			dst_p += dst.stride();
			dst_plane_p += dst_stride;
		}
	}
}

void usage()
{
	std::cerr << "unresize infile outfile width height [--shift-w shift] [--shift-h shift] [--times n] [--x86 | --no-x86]\n";
	std::cerr << "    infile              input BMP file\n";
	std::cerr << "    outfile             output BMP file\n";
	std::cerr << "    width               output width\n";
	std::cerr << "    height              output height\n";
	std::cerr << "    --shift-w           horizontal shift\n";
	std::cerr << "    --shift-h           vertical shift\n";
	std::cerr << "    --times             number of cycles\n";
	std::cerr << "    --x86 / --no-x86    toggle x86 optimizations\n";
}

} // namespace


int unresize_main(int argc, const char **argv)
{
	if (argc < 5) {
		usage();
		return -1;
	}

	const char *ifile = argv[1];
	const char *ofile = argv[2];
	int dst_width = atoi(argv[3]);
	int dst_height = atoi(argv[4]);

	float shift_w = 0;
	float shift_h = 0;
	int times = 1;
	bool x86 = false;

	for (int i = 5; i < argc; ++i) {
		if (!strcmp(argv[i], "--shift-w") && i + 1 < argc) {
			shift_w = (float)atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--shift-h") && i + 1 < argc) {
			shift_h = (float)atof(argv[i + 1]);
			++i;
		} else if (!strcmp(argv[i], "--times") && i + 1 < argc) {
			times = atoi(argv[i + 1]);
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
		Bitmap bmp = read_bitmap(ifile);
		Bitmap out(dst_width, dst_height, false);

		process(out, bmp, shift_w, shift_h, times, x86);
		write_bitmap(out, ofile);
	} catch (std::exception &e) {
		std::cerr << e.what() << '\n';
		return -1;
	}

	return 0;
}
