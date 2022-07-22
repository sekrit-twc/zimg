#include "common/pixel.h"
#include "filter_base.h"

namespace zimg {
namespace graph {

PointFilter::PointFilter(unsigned width, unsigned height, PixelType type)
{
	m_desc.format = { width, height, pixel_size(type) };
	m_desc.step = 1;
}

} // namespace graph
} // namespace zimg
