#include <algorithm>
#include <cstdint>
#include "common/pixel.h"
#include "basic_filter2.h"

namespace zimg {
namespace graph {

CopyFilter_GE::CopyFilter_GE(unsigned width, unsigned height, PixelType type) : m_desc{}
{
	m_desc.format = { width, height, pixel_size(type) };
	m_desc.num_deps = 1;
	m_desc.num_planes = 1;
	m_desc.step = 1;
	m_desc.flags.in_place = 1;
}

void CopyFilter_GE::process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
	                     unsigned i, unsigned left, unsigned right, void *, void *) const noexcept
{
	const uint8_t *src_p = in->get_line<uint8_t>(i);
	uint8_t *dst_p = out->get_line<uint8_t>(i);
	size_t left_byte = left * m_desc.format.bytes_per_sample;
	size_t right_byte = right * m_desc.format.bytes_per_sample;
	std::copy_n(src_p + left_byte, right_byte - left_byte, dst_p + left_byte);
}


ValueInitializeFilter_GE::ValueInitializeFilter_GE(unsigned width, unsigned height, PixelType type, value_type val) :
	m_desc{},
	m_value(val)
{
	m_desc.format = { width, height, pixel_size(type) };
	m_desc.num_deps = 0;
	m_desc.num_planes = 1;
	m_desc.step = 1;
}

void ValueInitializeFilter_GE::fill_b(void *ptr, size_t n) const
{
	std::fill_n(static_cast<uint8_t *>(ptr), n, m_value.b);
}

void ValueInitializeFilter_GE::fill_w(void *ptr, size_t n) const
{
	std::fill_n(static_cast<uint16_t *>(ptr), n, m_value.w);
}

void ValueInitializeFilter_GE::fill_f(void *ptr, size_t n) const
{
	std::fill_n(static_cast<float *>(ptr), n, m_value.f);
}

void ValueInitializeFilter_GE::process(const graphengine::BufferDescriptor *in, const graphengine::BufferDescriptor *out,
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


PremultiplyFilter_GE::PremultiplyFilter_GE(unsigned width, unsigned height) : m_desc{}
{
	m_desc.format = { width, height, pixel_size(PixelType::FLOAT) };
	m_desc.num_deps = 2;
	m_desc.num_planes = 1;
	m_desc.step = 1;
	m_desc.flags.in_place = 1;
}

void PremultiplyFilter_GE::process(const graphengine::BufferDescriptor in[2], const graphengine::BufferDescriptor *out,
                                unsigned i, unsigned left, unsigned right, void *, void *) const noexcept
{
	const float *src_p = in[0].get_line<float>(i);
	const float *alpha = in[1].get_line<float>(i);
	float *dst_p = out->get_line<float>(i);

	for (unsigned j = left; j < right; ++j) {
		float a = alpha[j];
		a = std::min(std::max(a, 0.0f), 1.0f);
		dst_p[j] = src_p[j] * alpha[j];
	}
}


UnpremultiplyFilter_GE::UnpremultiplyFilter_GE(unsigned width, unsigned height) : m_desc{}
{
	m_desc.format = { width, height, pixel_size(PixelType::FLOAT) };
	m_desc.num_deps = 2;
	m_desc.num_planes = 1;
	m_desc.step = 1;
	m_desc.flags.in_place = 1;
}

void UnpremultiplyFilter_GE::process(const graphengine::BufferDescriptor in[2], const graphengine::BufferDescriptor *out,
                                  unsigned i, unsigned left, unsigned right, void *, void *) const noexcept
{
	const float *src_p = in[0].get_line<float>(i);
	const float *alpha = in[1].get_line<float>(i);
	float *dst_p = out->get_line<float>(i);

	for (unsigned j = left; j < right; ++j) {
		float a = alpha[j];
		a = std::min(std::max(a, 0.0f), 1.0f);
		dst_p[j] = a == 0.0f ? 0.0f : src_p[j] / a;
	}
}

} // namespace graph
} // namespace zimg
