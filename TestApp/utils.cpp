#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "Common/align.h"
#include "Common/cpuinfo.h"
#include "Common/pixel.h"
#include "apps.h"
#include "floatutil.h"
#include "timer.h"

using namespace zimg;

namespace {;

CPUClass select_cpu(const char *cpu)
{
#ifdef ZIMG_X86
	if (!strcmp(cpu, "auto"))
		return CPUClass::CPU_X86_AUTO;
	else if (!strcmp(cpu, "sse2"))
		return CPUClass::CPU_X86_SSE2;
	else if (!strcmp(cpu, "avx2"))
		return CPUClass::CPU_X86_AVX2;
	else
		return CPUClass::CPU_NONE;
#else
	return CPUClass::CPU_NONE;
#endif // ZIMG_X86
}

PixelType select_pixel_type(const char *pixtype)
{
	if (!strcmp(pixtype, "u8"))
		return PixelType::BYTE;
	else if (!strcmp(pixtype, "u16"))
		return PixelType::WORD;
	else if (!strcmp(pixtype, "f16"))
		return PixelType::HALF;
	else if (!strcmp(pixtype, "f32"))
		return PixelType::FLOAT;
	else
		throw std::invalid_argument{ "unknown pixel type" };
}

uint16_t u8_to_u16(uint8_t x) { return x << 8; }
float    u8_to_f32(uint8_t x) { return normalize_float(x, UINT8_MAX); }
uint16_t u8_to_f16(uint8_t x) { return float_to_half(u8_to_f32(x)); }

uint8_t u16_to_u8(uint16_t x) { return (uint8_t)(x >> 8); }
uint8_t f32_to_u8(float x)    { return (uint8_t)std::round(denormalize_float(x, UINT8_MAX)); }
uint8_t f16_to_u8(uint16_t x) { return f32_to_u8(half_to_float(x)); }

int required_args(OptionType type)
{
	if (type == OptionType::OPTION_SPECIAL)
		return 1;
	else
		return 2;
}

} // namespace


int width_to_stride(int width, zimg::PixelType type)
{
	return align(width, ALIGNMENT / pixel_size(type));
}

size_t image_plane_size(int stride, int height, zimg::PixelType type)
{
	return (size_t)stride * height * pixel_size(type);
}

zimg::AlignedVector<char> allocate_buffer(size_t count, zimg::PixelType type)
{
	return zimg::AlignedVector<char>(count * pixel_size(type));
}

zimg::AlignedVector<char> allocate_frame(int stride, int height, int planes, zimg::PixelType type)
{
	return allocate_buffer((size_t)stride * height * planes, type);
}

void convert_from_byte(zimg::PixelType dst_pixel, const uint8_t *src, void *dst, int width, int height, int src_stride, int dst_stride)
{
	char *dst_byteptr = (char *)dst;
	int dst_bytestride = dst_stride * pixel_size(dst_pixel);

	for (int i = 0; i < height; ++i) {
		if (dst_pixel == PixelType::BYTE)
			std::copy(src, src + width, (uint8_t *)dst_byteptr);
		else if (dst_pixel == PixelType::WORD)
			std::transform(src, src + width, (uint16_t *)dst_byteptr, u8_to_u16);
		else if (dst_pixel == PixelType::HALF)
			std::transform(src, src + width, (uint16_t *)dst_byteptr, u8_to_f16);
		else if (dst_pixel == PixelType::FLOAT)
			std::transform(src, src + width, (float *)dst_byteptr, u8_to_f32);
		else
			throw std::invalid_argument{ "unknown pixel type" };

		src += src_stride;
		dst_byteptr += dst_bytestride;
	}
}

void convert_to_byte(zimg::PixelType src_pixel, const void *src, uint8_t *dst, int width, int height, int src_stride, int dst_stride)
{
	const char *src_byteptr = (const char *)src;
	int src_bytestride = src_stride * pixel_size(src_pixel);

	for (int i = 0; i < height; ++i) {
		if (src_pixel == PixelType::BYTE)
			std::copy((const uint8_t *)src_byteptr, (const uint8_t *)src_byteptr + width, dst);
		else if (src_pixel == PixelType::WORD)
			std::transform((const uint16_t *)src_byteptr, (const uint16_t *)src_byteptr + width, dst, u16_to_u8);
		else if (src_pixel == PixelType::HALF)
			std::transform((const uint16_t *)src_byteptr, (const uint16_t *)src_byteptr + width, dst, f16_to_u8);
		else if (src_pixel == PixelType::FLOAT)
			std::transform((const float *)src_byteptr, (const float *)src_byteptr + width, dst, f32_to_u8);
		else
			throw std::invalid_argument{ "unknown pixel type" };

		src_byteptr += src_bytestride;
		dst += dst_stride;
	}
}

void measure_time(int times, std::function<void(void)> f)
{
	Timer timer;
	double min_time = INFINITY;
	double avg_time = 0.0;

	for (int n = 0; n < times; ++n) {
		double elapsed;

		timer.start();
		f();
		timer.stop();

		elapsed = timer.elapsed();
		std::cout << '#' << n << ": " << elapsed << '\n';

		avg_time += elapsed / times;
		min_time = min_time < elapsed ? min_time : elapsed;
	}
	std::cout << "average: " << avg_time << '\n';
	std::cout << "min: " << min_time << '\n';
}

void parse_opts(const char **first, const char **last, const AppOption *options_first, const AppOption *options_last, void *dst, void *user)
{
	std::unordered_map<std::string, const AppOption *> option_map;
	char *dst_byteptr = reinterpret_cast<char *>(dst);

	while (options_first != options_last) {
		option_map[options_first->name] = options_first;
		++options_first;
	}

	while (first < last) {
		std::string o{ *first };

		if (o.find_first_of("--") == 0) {
			o = o.substr(2);
		} else if (o.find_first_of("-") == 0) {
			o = o.substr(1);
		} else {
			std::cerr << "not an option: " << o << '\n';
			throw std::invalid_argument{ "not an option" };
		}

		auto it = option_map.find(o);
		if (it == option_map.end()) {
			std::cerr << "unknown option: " << o << '\n';
			throw std::invalid_argument{ "unknown option" };
		}

		const AppOption *cur = it->second;
		char *dst_out = dst_byteptr + cur->offset;

		if (last - first < required_args(cur->type)) {
			std::cerr << "insufficient arguments to option: " << o << '\n';
			throw std::invalid_argument{ "insufficient arguments" };
		}

		switch (cur->type) {
		case OptionType::OPTION_INTEGER:
			*(int *)dst_out = std::stoi(first[1]);
			first += 2;
			break;
		case OptionType::OPTION_FLOAT:
			*(double *)dst_out = std::stod(first[1]);
			first += 2;
			break;
		case OptionType::OPTION_STRING:
			*(const char **)dst_out = first[1];
			first += 2;
			break;
		case OptionType::OPTION_CPUCLASS:
			*(CPUClass *)dst_out = select_cpu(first[1]);
			first += 2;
			break;
		case OptionType::OPTION_PIXELTYPE:
			*(PixelType *)dst_out = select_pixel_type(first[1]);
			first += 2;
			break;
		case OptionType::OPTION_SPECIAL:
			first += cur->func(first, last, dst, user);
			break;
		default:
			throw std::invalid_argument{ "unknown option type" };
		}
	}
}
