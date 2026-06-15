#pragma once

#ifndef FRAME_H_
#define FRAME_H_

#include <array>
#include "common/alloc.h"
#include "graphengine/types.h"

#define PATH_SPECIFIER_HELP_STR \
"Path specifier: spec@path\n" \
"BYTE:  bmp, grey, yuy2, yv12, yv16, yv24, i420, i422, i444, rgbp, gbrp\n" \
"WORD:  greyw, yv12w, yv16w, yv24w, i420w, i422w, i444w, rgbpw, gbrpw\n" \
"HALF:  greyh, i420h, i422h, i444h, rgbph\n" \
"FLOAT: greys, i420s, i422s, i444s, rgbps\n"

namespace zimg {

enum class PixelType;

} // namespace zimg


class ImageFrame {
	zimg::AlignedVector<unsigned char> m_vector[4];
	ptrdiff_t m_offset[4];
	unsigned m_width;
	unsigned m_height;
	zimg::PixelType m_pixel;
	unsigned m_planes;
	unsigned m_subsample_w;
	unsigned m_subsample_h;
	bool m_yuv;
public:
	ImageFrame(unsigned width, unsigned height, zimg::PixelType pixel, unsigned planes,
	           bool yuv = false, unsigned subsample_w = 0, unsigned subsample_h = 0);

	unsigned width(unsigned plane = 0) const noexcept
	{
		return m_width >> ((plane == 1 || plane == 2) ? m_subsample_w : 0);
	}

	unsigned height(unsigned plane = 0) const noexcept
	{
		return m_height >> ((plane == 1 || plane == 2) ? m_subsample_h : 0);
	}

	zimg::PixelType pixel_type() const noexcept { return m_pixel; }

	unsigned planes() const noexcept { return m_planes; }

	unsigned subsample_w() const noexcept { return m_subsample_w; }

	unsigned subsample_h() const noexcept { return m_subsample_h; }

	bool is_yuv() const noexcept { return m_yuv; }

	graphengine::BufferDescriptor as_buffer(unsigned plane) const noexcept;

	std::array<graphengine::BufferDescriptor, 4> as_buffer() const noexcept;
};


namespace imageframe {

ImageFrame read(const char *pathspec, const char *assumed, unsigned width, unsigned height);

ImageFrame read(const char *pathspec, const char *assumed, unsigned width, unsigned height, zimg::PixelType type, bool fullrange);

void write(const ImageFrame &frame, const char *pathspec, const char *assumed, bool fullrange = false);

void write(const ImageFrame &frame, const char *pathspec, const char *assumed, unsigned depth_in, bool fullrange);

} // namespace imageframe

#endif // FRAME_H_
