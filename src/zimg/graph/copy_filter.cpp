#include <algorithm>
#include "common/pixel.h"
#include "copy_filter.h"

namespace zimg {
namespace graph {

CopyFilter::CopyFilter(unsigned width, unsigned height, PixelType type, bool color) :
	m_attr{ width, height, type },
	m_color{ color }
{}

auto CopyFilter::get_flags() const -> filter_flags
{
	filter_flags flags{};

	flags.same_row = true;
	flags.in_place = true;
	flags.color = m_color;

	return flags;
}

auto CopyFilter::get_image_attributes() const -> image_attributes
{
	return m_attr;
}

void CopyFilter::process(void *, const ImageBuffer<const void> src[], const ImageBuffer<void> dst[], void *, unsigned i, unsigned left, unsigned right) const
{
	unsigned left_bytes = left * pixel_size(m_attr.type);
	unsigned right_bytes = right * pixel_size(m_attr.type);

	for (unsigned p = 0; p < (m_color ? 3U : 1U); ++p) {
		const uint8_t *src_p = static_cast<const uint8_t *>(src[p][i]);
		uint8_t *dst_p = static_cast<uint8_t *>(dst[p][i]);

		std::copy(src_p + left_bytes, src_p + right_bytes, dst_p + left_bytes);
	}
}

} // namespace graph
} // namespace zimg
