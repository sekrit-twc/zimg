#pragma once

#ifndef ZIMG_PAIR_FILTER_H_
#define ZIMG_PAIR_FILTER_H_

#include <memory>
#include "graph/zfilter.h"

enum class PixelType;

class PairFilter final : public zimg::graph::IZimgFilter {
	struct cache_context;
private:
	std::unique_ptr<zimg::graph::IZimgFilter> m_first;
	std::unique_ptr<zimg::graph::IZimgFilter> m_second;

	filter_flags m_first_flags;
	filter_flags m_second_flags;

	image_attributes m_first_attr;
	image_attributes m_second_attr;

	unsigned m_first_step;
	unsigned m_second_step;
	unsigned m_second_buffering;

	bool m_has_state;
	bool m_in_place;
	bool m_color;

	ptrdiff_t get_cache_stride() const;

	unsigned get_cache_line_count() const;

	size_t get_cache_size_one_plane() const;

	unsigned get_num_planes() const;
public:
	PairFilter(zimg::graph::IZimgFilter *first, zimg::graph::IZimgFilter *second);

	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override;

	unsigned get_simultaneous_lines() const override;

	unsigned get_max_buffering() const override;

	size_t get_context_size() const override;

	size_t get_tmp_size(unsigned left, unsigned right) const override;

	void init_context(void *ctx) const override;

	void process(void *ctx, const zimg::graph::ZimgImageBufferConst &src, const zimg::graph::ZimgImageBuffer &dst, void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

#endif // ZIMG_PAIR_FILTER_H_
