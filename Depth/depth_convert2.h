#pragma once

#ifndef ZIMG_DEPTH_DEPTH_CONVERT2_H_
#define ZIMG_DEPTH_DEPTH_CONVERT2_H_

#include "Common/zfilter.h"

namespace zimg {;

struct PixelFormat;

enum class PixelType;
enum class CPUClass;

namespace depth {;

class DepthConvert2 : public ZimgFilter {
public:
	typedef void (*func_type)(const void *src, void *dst, float scale, float offset, unsigned width);
	typedef void (*f16c_func_type)(const void *src, void *dst, unsigned width);
private:
	func_type m_func;
	f16c_func_type m_f16c;

	PixelType m_pixel_in;
	PixelType m_pixel_out;
	float m_scale;
	float m_offset;
public:
	DepthConvert2(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

	zimg_filter_flags get_flags() const override;

	size_t get_tmp_size(unsigned left, unsigned right) const override;

	void process(void *ctx, const zimg_image_buffer *src, const zimg_image_buffer *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DEPTH_CONVERT2_H_
