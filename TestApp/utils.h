#pragma once

#ifndef UTILS_H_
#define UTILS_H_

#include <cstddef>
#include <functional>
#include "Common/align.h"

namespace zimg {;

enum class CPUClass;
enum class PixelType;

class IZimgFilter;

} // namespace zimg


class Frame;

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
	int(*func)(const char **opt, const char **lastopt, void *p, void *user);
};

zimg::CPUClass select_cpu(const char *cpu);

zimg::PixelType select_pixel_type(const char *pixtype);

zimg::AlignedVector<char> allocate_buffer(size_t count, zimg::PixelType type);

void convert_frame(const Frame &in, Frame &out, zimg::PixelType pxl_in, zimg::PixelType pxl_out, bool fullrange, bool yuv);

zimg::AlignedVector<char> alloc_filter_tmp(const zimg::IZimgFilter &filter, const Frame &in, Frame &out);

void apply_filter(const zimg::IZimgFilter &filter, const Frame &in, Frame &out, void *alloc_pool, int plane);

void measure_time(int times, std::function<void(void)> f);

void parse_opts(const char **first, const char **last, const AppOption *options_first, const AppOption *options_last, void *dst, void *user);

#endif // UTILS_H_
