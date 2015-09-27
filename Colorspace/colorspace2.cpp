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
	float *buf[3];
	unsigned count = right - left;

	for (unsigned p = 0; p < 3; ++p) {
		LineBuffer<const float> src_buf{ src, p };
		LineBuffer<float> dst_buf{ dst, p };

		const float *src_p = src_buf[i] + left;
		float *dst_p = dst_buf[i] + left;

		if (src_p != dst_p)
			std::copy_n(src_p, count, dst_p);

		buf[p] = dst_p;
	}

	for (auto &o : m_operations) {
		o->process(buf, count);
	}
}

} // namespace colorspace
} // namespace zimg
