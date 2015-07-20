#pragma once

#ifndef ZIMG_COLORSPACE_COLORSPACE2_H_
#define ZIMG_COLORSPACE_COLORSPACE2_H_

#include <memory>
#include <vector>
#include "Common/zfilter.h"
#include "operation.h"

namespace zimg {;
namespace colorspace {;

struct ColorspaceDefinition;

class ColorspaceConversion2 final : public ZimgFilter {
	std::vector<std::shared_ptr<Operation>> m_operations;
public:
	ColorspaceConversion2() = default;

	ColorspaceConversion2(const ColorspaceDefinition &in, const ColorspaceDefinition &out, CPUClass cpu);

	zimg_filter_flags get_flags() const override;

	void process(void *ctx, const zimg_image_buffer *src, const zimg_image_buffer *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override;
};

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_COLORSPACE2_H_
