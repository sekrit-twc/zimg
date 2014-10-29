#include <algorithm>
#include <functional>
#include <unordered_map>
#include <vector>
#include "Common/align.h"
#include "Common/except.h"
#include "Common/pixel.h"
#include "Common/plane.h"
#include "colorspace.h"
#include "colorspace_param.h"
#include "graph.h"

namespace zimg {;
namespace colorspace {;

namespace {;

bool is_valid_csp(const ColorspaceDefinition &csp)
{
	return !(csp.matrix == MatrixCoefficients::MATRIX_2020_CL && csp.transfer == TransferCharacteristics::TRANSFER_LINEAR);
}

} // namespace


ColorspaceConversion::ColorspaceConversion(const ColorspaceDefinition &in, const ColorspaceDefinition &out, CPUClass cpu)
try :
	m_pixel_adapter{ create_pixel_adapter(cpu) }
{	
	if (!is_valid_csp(in) || !is_valid_csp(out))
		throw ZimgIllegalArgument{ "invalid colorspace definition" };

	for (const auto &func : get_operation_path(in, out)) {
		m_operations.emplace_back(func(cpu));
	}
} catch (const std::bad_alloc &) {
	throw ZimgOutOfMemory{};
}

void ColorspaceConversion::load_line(const void *src, float *dst, int width, PixelType type) const
{
	switch (type) {
	case PixelType::HALF:
		m_pixel_adapter->f16_to_f32((const uint16_t *)src, dst, width);
		break;
	case PixelType::FLOAT:
		std::copy_n((const float *)src, width, dst);
		break;
	default:
		throw ZimgUnsupportedError{ "unsupported pixel type" };
	}
}

void ColorspaceConversion::store_line(const float *src, void *dst, int width, PixelType type) const
{
	switch (type) {
	case PixelType::HALF:
		m_pixel_adapter->f16_from_f32(src, (uint16_t *)dst, width);
		break;
	case PixelType::FLOAT:
		std::copy_n(src, width, (float *)dst);
		break;
	default:
		throw ZimgUnsupportedError{ "unsupported pixel type" };
	}
}

size_t ColorspaceConversion::tmp_size(int width) const
{
	return 3 * align(width, AlignmentOf<float>::value);
}

void ColorspaceConversion::process(const ImagePlane<void> *src, ImagePlane<void> *dst, void *tmp) const
{
	PixelType src_type = src[0].format().type;
	PixelType dst_type = dst[0].format().type;
	size_t src_pxsize = pixel_size(src_type);
	size_t dst_pxsize = pixel_size(dst_type);

	int width = src[0].width();
	int height = src[1].height();

	ptrdiff_t tmp_stride = align(width, AlignmentOf<float>::value);
	float *tmp_f = (float *)tmp;

	const void *src_p[3] = { src[0].data(), src[1].data(), src[2].data() };
	void *dst_p[3] = { dst[0].data(), dst[1].data(), dst[2].data() };
	float *buf[3] = { tmp_f, tmp_f + tmp_stride, tmp_f + 2 * tmp_stride };

	for (int i = 0; i < height; ++i) {
		load_line(src_p[0], buf[0], width, src_type);
		load_line(src_p[1], buf[1], width, src_type);
		load_line(src_p[2], buf[2], width, src_type);

		for (auto &o : m_operations) {
			o->process(buf, width);
		}

		store_line(buf[0], dst_p[0], width, dst_type);
		store_line(buf[1], dst_p[1], width, dst_type);
		store_line(buf[2], dst_p[2], width, dst_type);

		src_p[0] = (const char *)src_p[0] + src[0].stride() * src_pxsize;
		src_p[1] = (const char *)src_p[1] + src[1].stride() * src_pxsize;
		src_p[2] = (const char *)src_p[2] + src[2].stride() * src_pxsize;
		dst_p[0] = (char *)dst_p[0] + dst[0].stride() * dst_pxsize;
		dst_p[1] = (char *)dst_p[1] + dst[1].stride() * dst_pxsize;
		dst_p[2] = (char *)dst_p[2] + dst[2].stride() * dst_pxsize;
	}
}

} // namespace colorspace
} // namespace zimg
