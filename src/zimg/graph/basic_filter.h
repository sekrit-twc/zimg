#pragma once

#ifndef ZIMG_GRAPH_BASIC_FILTER_H_
#define ZIMG_GRAPH_BASIC_FILTER_H_

#include <cstdint>
#include "image_filter.h"

namespace zimg {

enum class PixelType;

namespace graph {

// Converts greyscale to RGB image by replicating the luma plane.
//
// For any YUV system, a greyscale image is encoded by U=0 and V=0, which also
// implies R=G=B. Since Y is a weighted sum of R, G, and B, this also implies
// R=G=B=Y.
class RGBExtendFilter : public ImageFilterBase {
	image_attributes m_attr;
public:
	RGBExtendFilter(unsigned width, unsigned height, PixelType type);

	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	void process(void *, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *, unsigned i, unsigned left, unsigned right) const override;
};

// Initializes a plane to a constant value.
class ValueInitializeFilter : public ImageFilterBase {
public:
	union value_type {
		uint8_t b;
		uint16_t w;
		float f;
	};
private:
	image_attributes m_attr;
	value_type m_value;

	void fill_b(void *ptr, size_t n) const;
	void fill_w(void *ptr, size_t n) const;
	void fill_f(void *ptr, size_t n) const;
public:
	ValueInitializeFilter(unsigned width, unsigned height, PixelType type, value_type val);

	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	void process(void *, const ImageBuffer<const void> *, const ImageBuffer<void> *dst, void *, unsigned i, unsigned left, unsigned right) const override;
};

// Premultiplies an image.
class PremultiplyFilter : public ImageFilterBase {
	unsigned m_width;
	unsigned m_height;
	bool m_color;
public:
	PremultiplyFilter(unsigned width, unsigned height, bool color);

	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	void process(void *, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *, unsigned i, unsigned left, unsigned right) const override;
};

// Unpremultiplies an image.
class UnpremultiplyFilter : public ImageFilterBase {
	unsigned m_width;
	unsigned m_height;
	bool m_color;
public:
	UnpremultiplyFilter(unsigned width, unsigned height, bool color);

	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	void process(void *, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *, unsigned i, unsigned left, unsigned right) const override;
};

} // namespace graph
} // namespace zimg

#endif // ZIMG_GRAPH_BASIC_FILTER_H_
