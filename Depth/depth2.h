#pragma once

#ifndef ZIMG_DEPTH_DEPTH2_H_
#define ZIMG_DEPTH_DEPTH2_H_

#include <memory>
#include "Common/zfilter.h"

namespace zimg {;

enum class CPUClass;
struct PixelFormat;

namespace depth {;

#ifndef ZIMG_DEPTH_DEPTH_H_
enum class DitherType {
	DITHER_NONE,
	DITHER_ORDERED,
	DITHER_RANDOM,
	DITHER_ERROR_DIFFUSION
};
#endif

class Depth2 : public IZimgFilter {
	std::shared_ptr<IZimgFilter> m_impl;
public:
	Depth2() = default;

	Depth2(DitherType type, unsigned width, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu);

	ZimgFilterFlags get_flags() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override;

	unsigned get_simultaneous_lines() const override;

	unsigned get_max_buffering() const override;

	size_t get_context_size() const override;

	size_t get_tmp_size(unsigned left, unsigned right) const override;

	void init_context(void *ctx) const override;

	void process(void *ctx, const ZimgImageBuffer src[3], const ZimgImageBuffer dst[3], void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DEPTH2_H_
