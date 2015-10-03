#pragma once

#ifndef ZIMG_COLORSPACE_COLORSPACE_H_
#define ZIMG_COLORSPACE_COLORSPACE_H_

#include <memory>
#include <vector>
#include "graph/zfilter.h"
#include "operation.h"

namespace zimg {;
namespace colorspace {;

struct ColorspaceDefinition;

class ColorspaceConversion final : public graph::ZimgFilter {
	std::vector<std::shared_ptr<Operation>> m_operations;
	unsigned m_width;
	unsigned m_height;
public:
	ColorspaceConversion() = default;

	ColorspaceConversion(unsigned width, unsigned height, const ColorspaceDefinition &in, const ColorspaceDefinition &out, CPUClass cpu);

	image_attributes get_image_attributes() const override;

	filter_flags get_flags() const override;

	void process(void *ctx, const graph::ZimgImageBufferConst &src, const graph::ZimgImageBuffer &dst, void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_COLORSPACE2_H_
