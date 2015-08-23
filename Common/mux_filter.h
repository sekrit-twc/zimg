#pragma once

#include <memory>
#include "zfilter.h"

namespace zimg {;

class MuxFilter final : public IZimgFilter {
	std::unique_ptr<IZimgFilter> m_filter;
	std::unique_ptr<IZimgFilter> m_filter_uv;
	ZimgFilterFlags m_flags;
public:
	MuxFilter(IZimgFilter *filter, IZimgFilter *filter_uv);

	ZimgFilterFlags get_flags() const override;

	image_attributes get_image_attributes() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override;

	unsigned get_simultaneous_lines() const override;

	unsigned get_max_buffering() const override;

	size_t get_context_size() const override;

	size_t get_tmp_size(unsigned left, unsigned right) const override;

	void init_context(void *ctx) const override;

	void process(void *ctx, const ZimgImageBuffer *src, const ZimgImageBuffer *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

} // namespace zimg
