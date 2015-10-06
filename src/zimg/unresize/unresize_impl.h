#pragma once

#ifndef ZIMG_UNRESIZE_UNRESIZE_IMPL_H_
#define ZIMG_UNRESIZE_UNRESIZE_IMPL_H_

#include <memory>
#include "graph/image_filter.h"
#include "bilinear.h"

namespace zimg {;

enum class CPUClass;
enum class PixelType;

namespace unresize {;

class UnresizeImplH : public graph::ImageFilterBase {
protected:
	BilinearContext m_context;
	image_attributes m_attr;

	UnresizeImplH(const BilinearContext &context, const image_attributes &attr);
public:
	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override;

	unsigned get_max_buffering() const override;
};

class UnresizeImplV : public graph::ImageFilterBase {
protected:
	BilinearContext m_context;
	image_attributes m_attr;

	UnresizeImplV(const BilinearContext &context, const image_attributes &attr);
public:
	filter_flags get_flags() const override;

	image_attributes get_image_attributes() const override;

	pair_unsigned get_required_row_range(unsigned i) const override;

	pair_unsigned get_required_col_range(unsigned left, unsigned right) const override;

	unsigned get_simultaneous_lines() const override;

	unsigned get_max_buffering() const override;
};

struct UnresizeImplBuilder {
	unsigned up_width;
	unsigned up_height;
	PixelType type;

#include "common/builder.h"
	BUILDER_MEMBER(bool, horizontal);
	BUILDER_MEMBER(unsigned, orig_dim);
	BUILDER_MEMBER(double, shift);
	BUILDER_MEMBER(CPUClass, cpu);
#undef BUILDER_MEMBER

	UnresizeImplBuilder(unsigned up_width, unsigned up_height, PixelType type);

	std::unique_ptr<graph::ImageFilter> create() const;
};

} // namespace unresize
} // namespace zimg

#endif // ZIMG_UNRESIZE_UNRESIZE_IMPL_H_
