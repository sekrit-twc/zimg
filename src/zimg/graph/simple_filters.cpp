#include <algorithm>
#include <cstdint>
#include "common/pixel.h"
#include "simple_filters.h"

namespace zimg::graph {

CopyRectFilter::CopyRectFilter(unsigned left, unsigned top, unsigned width, unsigned height, PixelType type) :
	m_left{ left },
	m_top{ top }
{
	m_desc.format = { width, height, pixel_size(type) };
	m_desc.num_deps = 1;
	m_desc.num_planes = 1;
	m_desc.step = 1;
}

void CopyRectFilter::process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
                             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept
{
	const unsigned char *src_p = in->get_line<unsigned char>(m_top + i) + (static_cast<size_t>(left) + static_cast<size_t>(m_left)) * m_desc.format.bytes_per_sample;
	unsigned char *dst_p = out->get_line<unsigned char>(i) + static_cast<size_t>(left) * m_desc.format.bytes_per_sample;
	std::copy_n(src_p, static_cast<size_t>(right - left) * m_desc.format.bytes_per_sample, dst_p);
}


ValueInitializeFilter::ValueInitializeFilter(unsigned width, unsigned height, PixelType type, value_type val) :
	PointFilter(width, height, type),
	m_value(val)
{
	m_desc.num_deps = 0;
	m_desc.num_planes = 1;
}

void ValueInitializeFilter::fill_b(void *ptr, size_t n) const
{
	std::fill_n(static_cast<uint8_t *>(ptr), n, m_value.b);
}

void ValueInitializeFilter::fill_w(void *ptr, size_t n) const
{
	std::fill_n(static_cast<uint16_t *>(ptr), n, m_value.w);
}

void ValueInitializeFilter::fill_f(void *ptr, size_t n) const
{
	std::fill_n(static_cast<float *>(ptr), n, m_value.f);
}

void ValueInitializeFilter::process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
                                    unsigned i, unsigned left, unsigned right, void *, void *) const noexcept
{
	unsigned char *dst_p = out->get_line<unsigned char>(i) + static_cast<size_t>(left) * m_desc.format.bytes_per_sample;

	switch (m_desc.format.bytes_per_sample) {
	case sizeof(uint8_t):
		fill_b(dst_p, right - left);
		break;
	case sizeof(uint16_t):
		fill_w(dst_p, right - left);
		break;
	case sizeof(float):
		fill_f(dst_p, right - left);
		break;
	}
}


PremultiplyFilter::PremultiplyFilter(unsigned width, unsigned height) : PointFilter(width, height, PixelType::FLOAT)
{
	m_desc.num_deps = 2;
	m_desc.num_planes = 1;
	m_desc.flags.in_place = 1;
}

void PremultiplyFilter::process(const graphengine::BufferDescriptor in[2], const graphengine::BufferDescriptor *out,
                                unsigned i, unsigned left, unsigned right, void *, void *) const noexcept
{
	const float *src_p = in[0].get_line<float>(i);
	const float *alpha = in[1].get_line<float>(i);
	float *dst_p = out->get_line<float>(i);

	for (unsigned j = left; j < right; ++j) {
		float a = alpha[j];
		a = std::clamp(a, 0.0f, 1.0f);
		dst_p[j] = src_p[j] * alpha[j];
	}
}


UnpremultiplyFilter::UnpremultiplyFilter(unsigned width, unsigned height) : PointFilter(width, height, PixelType::FLOAT)
{
	m_desc.num_deps = 2;
	m_desc.num_planes = 1;
	m_desc.flags.in_place = 1;
}

void UnpremultiplyFilter::process(const graphengine::BufferDescriptor in[2], const graphengine::BufferDescriptor *out,
                                  unsigned i, unsigned left, unsigned right, void *, void *) const noexcept
{
	const float *src_p = in[0].get_line<float>(i);
	const float *alpha = in[1].get_line<float>(i);
	float *dst_p = out->get_line<float>(i);

	for (unsigned j = left; j < right; ++j) {
		float a = alpha[j];
		a = std::clamp(a, 0.0f, 1.0f);
		dst_p[j] = a == 0.0f ? 0.0f : src_p[j] / a;
	}
}

} // namespace zimg::graph
