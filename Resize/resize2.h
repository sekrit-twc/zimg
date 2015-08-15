#pragma once

#ifndef ZIMG_RESIZE_RESIZE2_H_
#define ZIMG_RESIZE_RESIZE2_H_

#include "Common/zfilter.h"

namespace zimg {;

enum class CPUClass;
enum class PixelType;

class IZimgFilter;

namespace resize {;

class Filter;

class Resize2 : public IZimgFilter {
	std::shared_ptr<IZimgFilter> m_impl;
public:
	Resize2() = default;

	Resize2(const Filter &filter, PixelType type, int src_width, int src_height, int dst_width, int dst_height,
	        double shift_w, double shift_h, double subwidth, double subheight, CPUClass cpu);

	ZimgFilterFlags get_flags() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override;

	unsigned get_simultaneous_lines() const override;

	unsigned get_max_buffering() const override;

	size_t get_context_size() const override;

	size_t get_tmp_size(unsigned left, unsigned right) const override;

	void init_context(void *ctx) const override;

	void process(void *ctx, const ZimgImageBuffer *src, const ZimgImageBuffer *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE2_H_
