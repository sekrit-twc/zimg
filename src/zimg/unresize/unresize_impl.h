#pragma once

#ifndef ZIMG_UNRESIZE_UNRESIZE_IMPL_H_
#define ZIMG_UNRESIZE_UNRESIZE_IMPL_H_

#include <memory>
#include "graph/filter_base.h"
#include "bilinear.h"

namespace zimg {

enum class CPUClass;
enum class PixelType;

namespace unresize {

class UnresizeImplH : public graph::FilterBase {
protected:
	BilinearContext m_context;

	UnresizeImplH(const BilinearContext &context, unsigned width, unsigned height, PixelType type);
public:
	pair_unsigned get_row_deps(unsigned i) const noexcept override;

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override;
};

class UnresizeImplV : public graph::FilterBase {
protected:
	BilinearContext m_context;

	UnresizeImplV(const BilinearContext &context, unsigned width, unsigned height, PixelType type);
public:
	pair_unsigned get_row_deps(unsigned i) const noexcept override;

	pair_unsigned get_col_deps(unsigned left, unsigned right) const noexcept override;
};

struct UnresizeImplBuilder {
	unsigned up_width;
	unsigned up_height;
	PixelType type;

#include "common/builder.h"
	BUILDER_MEMBER(bool, horizontal)
	BUILDER_MEMBER(unsigned, orig_dim)
	BUILDER_MEMBER(double, shift)
	BUILDER_MEMBER(CPUClass, cpu)
#undef BUILDER_MEMBER

	UnresizeImplBuilder(unsigned up_width, unsigned up_height, PixelType type);

	std::unique_ptr<graphengine::Filter> create() const;
};

} // namespace unresize
} // namespace zimg

#endif // ZIMG_UNRESIZE_UNRESIZE_IMPL_H_
