#include <algorithm>
#include "Common/except.h"
#include "Common/linebuffer.h"
#include "Common/pixel.h"
#include "colorspace.h"
#include "colorspace_param.h"
#include "colorspace2.h"
#include "graph.h"

namespace zimg {;
namespace colorspace {;

ColorspaceConversion2::ColorspaceConversion2(unsigned width, unsigned height, const ColorspaceDefinition &in, const ColorspaceDefinition &out, CPUClass cpu)
try :
	m_width{ width },
	m_height{ height }
{
	for (const auto &func : get_operation_path(in, out)) {
		m_operations.emplace_back(func(cpu));
	}
} catch (const std::bad_alloc &) {
	throw error::OutOfMemory{};
}

ZimgFilterFlags ColorspaceConversion2::get_flags() const
{
	ZimgFilterFlags flags{};

	flags.same_row = true;
	flags.in_place = true;
	flags.color = true;

	return flags;
}

IZimgFilter::image_attributes ColorspaceConversion2::get_image_attributes() const
{
	return{ m_width, m_height, PixelType::FLOAT };
}

void ColorspaceConversion2::process(void *, const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *, unsigned i, unsigned left, unsigned right) const
{
	const float *src_ptr[3];
	float *dst_ptr[3];

	for (unsigned p = 0; p < 3; ++p) {
		src_ptr[p] = LineBuffer<const float>{ src, p }[i];
		dst_ptr[p] = LineBuffer<float>{ dst, p }[i];
	}

	if (m_operations.empty()) {
		for (unsigned p = 0; p < 3; ++p) {
			std::copy(src_ptr[p] + left, src_ptr[p] + right, dst_ptr[p] + left);
		}
	} else {
		m_operations[0]->process(src_ptr, dst_ptr, left, right);

		for (size_t i = 1; i < m_operations.size(); ++i) {
			m_operations[i]->process(dst_ptr, dst_ptr, left, right);
		}
	}
}

} // namespace colorspace
} // namespace zimg
