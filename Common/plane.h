#pragma once

#ifndef ZIMG_COMMON_PLANE_H_
#define ZIMG_COMMON_PLANE_H_

#include <algorithm>
#include "pixel.h"

namespace zimg {;

template <class T>
class ImagePlane {
	void *m_data;
	int m_width;
	int m_height;
	int m_stride;
	PixelFormat m_format;
public:
	ImagePlane() = default;

	ImagePlane(const T *data, int width, int height, int stride, PixelType type) :
		ImagePlane(data, width, height, stride, default_pixel_format(type))
	{}

	ImagePlane(const T *data, int width, int height, int stride, const PixelFormat &format) :
		m_data{ const_cast<T *>(data) }, m_width{ width }, m_height{ height }, m_stride{ stride }, m_format(format)
	{}

	operator ImagePlane<void>()
	{
		return{ m_data, m_width, m_height, m_stride, m_format };
	}

	int width() const
	{
		return m_width;
	}

	int height() const
	{
		return m_height;
	}

	int stride() const
	{
		return m_stride;
	}

	const PixelFormat &format() const
	{
		return m_format;
	}

	T *data()
	{
		return reinterpret_cast<T *>(m_data);
	}

	const T *data() const
	{
		return cdata();
	}

	const T *cdata() const
	{
		return reinterpret_cast<const T *>(m_data);
	}
};

template <class T, class U>
ImagePlane<T> plane_cast(const ImagePlane<U> &x)
{
	return{ reinterpret_cast<const T *>(x.data()), x.width(), x.height(), x.stride(), x.format() };
}

template <class T>
inline void copy_image_plane(const ImagePlane<T> &src, ImagePlane<T> &dst)
{
	int pxsize = pixel_size(src.format().type);
	ptrdiff_t byte_width = pxsize * src.width();
	ptrdiff_t src_stride = pxsize * src.stride();
	ptrdiff_t dst_stride = pxsize * dst.stride();

	const char *src_p = reinterpret_cast<const char *>(src.cdata());
	char *dst_p = reinterpret_cast<char *>(dst.data());

	for (int i = 0; i < src.height(); ++i) {
		std::copy_n(src_p, byte_width, dst_p);

		src_p += src_stride;
		dst_p += dst_stride;
	}
}

} // namespace zimg

#endif // ZIMG_COMMON_PLANE_H_
