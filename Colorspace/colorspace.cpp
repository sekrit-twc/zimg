#include <algorithm>
#include <functional>
#include <unordered_map>
#include <vector>
#include "Common/align.h"
#include "Common/except.h"
#include "colorspace.h"
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
	m_pixel_adapter{ create_pixel_adapter(cpu) },
	m_input_is_yuv{ in.matrix != MatrixCoefficients::MATRIX_RGB },
	m_output_is_yuv{ out.matrix != MatrixCoefficients::MATRIX_RGB }
{	
	if (!is_valid_csp(in) || !is_valid_csp(out))
		throw ZimgIllegalArgument{ "invalid colorspace definition" };

	for (const auto &func : get_operation_path(in, out)) {
		m_operations.emplace_back(func(cpu));
	}
} catch (const std::bad_alloc &) {
	throw ZimgOutOfMemory{};
}

void ColorspaceConversion::load_line(const void *src, float *dst, int width, bool tv, bool chroma, PixelType type) const
{
	switch (type) {
	case PixelType::WORD:
		m_pixel_adapter->u16_to_f32((const uint16_t *)src, dst, width, tv, chroma);
		break;
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

void ColorspaceConversion::store_line(const float *src, void *dst, int width, bool tv, bool chroma, PixelType type) const
{
	switch (type) {
	case PixelType::WORD:
		m_pixel_adapter->u16_from_f32(src, (uint16_t *)dst, width, tv, chroma);
		break;
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

void ColorspaceConversion::process(PixelType type, const void * const *src, void * const *dst, void *tmp, int width, int height, const int *src_stride, const int *dst_stride, bool tv_in, bool tv_out) const
{
	const void *src_p[3] = { src[0], src[1], src[2] };
	void *dst_p[3] = { dst[0], dst[1], dst[2] };
	float *buf[3];
	ptrdiff_t tmp_stride = align(width, AlignmentOf<float>::value);
	size_t pxsize = pixel_size(type);

	buf[0] = (float *)tmp;
	buf[1] = buf[0] + tmp_stride;
	buf[2] = buf[1] + tmp_stride;

	std::fill_n(buf[0], tmp_stride, 0.0f);
	std::fill_n(buf[1], tmp_stride, 0.0f);
	std::fill_n(buf[2], tmp_stride, 0.0f);

	for (int i = 0; i < height; ++i) {
		load_line(src_p[0], buf[0], width, tv_in, false, type);
		load_line(src_p[1], buf[1], width, tv_in, m_input_is_yuv, type);
		load_line(src_p[2], buf[2], width, tv_in, m_input_is_yuv, type);

		for (auto &o : m_operations) {
			o->process(buf, width);
		}

		store_line(buf[0], dst_p[0], width, tv_out, false, type);
		store_line(buf[1], dst_p[1], width, tv_out, m_output_is_yuv, type);
		store_line(buf[2], dst_p[2], width, tv_out, m_output_is_yuv, type);

		src_p[0] = (const char *)src_p[0] + src_stride[0] * pxsize;
		src_p[1] = (const char *)src_p[1] + src_stride[1] * pxsize;
		src_p[2] = (const char *)src_p[2] + src_stride[2] * pxsize;
		dst_p[0] = (char *)dst_p[0] + dst_stride[0] * pxsize;
		dst_p[1] = (char *)dst_p[1] + dst_stride[1] * pxsize;
		dst_p[2] = (char *)dst_p[2] + dst_stride[2] * pxsize;
	}
}

} // namespace colorspace
} // namespace zimg
