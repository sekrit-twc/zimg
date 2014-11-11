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
	return 3 * align((size_t)width, AlignmentOf<float>::value);
}

void ColorspaceConversion::process(const ImagePlane<const void> *src, const ImagePlane<void> *dst, void *tmp) const
{
	PixelType src_type = src[0].format().type;
	PixelType dst_type = dst[0].format().type;

	int width = src[0].width();
	int height = src[0].height();

	ptrdiff_t tmp_stride = align(width, AlignmentOf<float>::value);
	float *tmp_f = (float *)tmp;
	float *buf[3] = { tmp_f, tmp_f + tmp_stride, tmp_f + 2 * tmp_stride };

	for (int i = 0; i < height; ++i) {
		load_line(src[0][i], buf[0], width, src_type);
		load_line(src[1][i], buf[1], width, src_type);
		load_line(src[2][i], buf[2], width, src_type);

		for (auto &o : m_operations) {
			o->process(buf, width);
		}

		store_line(buf[0], dst[0][i], width, dst_type);
		store_line(buf[1], dst[1][i], width, dst_type);
		store_line(buf[2], dst[2][i], width, dst_type);
	}
}

} // namespace colorspace
} // namespace zimg
