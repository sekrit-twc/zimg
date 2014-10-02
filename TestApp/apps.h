#pragma once

#ifndef APPS_H_
#define APPS_H_

#include <cstddef>
#include <functional>
#include "Common/align.h"
#include "Common/pixel.h"

enum class OptionType {
	OPTION_INTEGER,
	OPTION_FLOAT,
	OPTION_STRING,
	OPTION_FALSE,
	OPTION_TRUE,
	OPTION_CPUCLASS,
	OPTION_PIXELTYPE,
	OPTION_SPECIAL
};

struct AppOption {
	const char *name;
	OptionType type;
	ptrdiff_t offset;
	int (*func)(const char **opt, const char **lastopt, void *p, void *user);
};

int width_to_stride(int width, zimg::PixelType type);

size_t image_plane_size(int stride, int height, zimg::PixelType type);

zimg::AlignedVector<char> allocate_buffer(size_t count, zimg::PixelType type);

zimg::AlignedVector<char> allocate_frame(int stride, int height, int planes, zimg::PixelType type);

void convert_from_byte(zimg::PixelType dst_pixel, const uint8_t *src, void *dst, int width, int height, int src_stride, int dst_stride);

void convert_to_byte(zimg::PixelType src_pixel, const void *src, uint8_t *dst, int width, int height, int src_stride, int dst_stride);

void measure_time(int times, std::function<void(void)> f);

void parse_opts(const char **first, const char **last, const AppOption *options_first, const AppOption *options_last, void *dst, void *user);

int resize_main(int argc, const char **argv);

int unresize_main(int argc, const char **argv);

int colorspace_main(int argc, const char **argv);

#endif // APPS_H_
