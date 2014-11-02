#include <cstddef>
#include <utility>
#include "Common/except.h"
#include "Common/plane.h"
#include "Common/pixel.h"
#include "depth.h"
#include "depth_convert.h"
#include "dither.h"

namespace zimg {;
namespace depth {;

namespace {;

template <class Func, class T, class U, class... Args>
void invoke_depth(const DepthConvert &depth, Func func, const ImagePlane<const T> &src, const ImagePlane<U> &dst, Args... arg)
{
	int src_stride = src.stride();
	int dst_stride = dst.stride();
	const T *src_p = src.data();
	U *dst_p = dst.data();

	for (int i = 0; i < src.height(); ++i) {
		(depth.*func)(src_p, dst_p, src.width(), std::forward<Args>(arg)...);

		src_p += src_stride;
		dst_p += dst_stride;
	}
}

void convert_dithered(const DitherConvert &dither, const ImagePlane<const void> &src, const ImagePlane<void> &dst, void *tmp)
{
	ImagePlane<const uint8_t> src_b = plane_cast<const uint8_t>(src);
	ImagePlane<const uint16_t> src_w = plane_cast<const uint16_t>(src);
	ImagePlane<const float> src_f = plane_cast<const float>(src);

	ImagePlane<uint8_t> dst_b = plane_cast<uint8_t>(dst);
	ImagePlane<uint16_t> dst_w = plane_cast<uint16_t>(dst);

	PixelType src_type = src.format().type;
	PixelType dst_type = dst.format().type;
	float *tmp_f = (float *)tmp;

	if (src_type == PixelType::BYTE && dst_type == PixelType::BYTE)
		dither.byte_to_byte(src_b, dst_b, tmp_f);
	else if (src_type == PixelType::BYTE && dst_type == PixelType::WORD)
		dither.byte_to_word(src_b, dst_w, tmp_f);
	else if (src_type == PixelType::WORD && dst_type == PixelType::BYTE)
		dither.word_to_byte(src_w, dst_b, tmp_f);
	else if (src_type == PixelType::WORD && dst_type == PixelType::WORD)
		dither.word_to_word(src_w, dst_w, tmp_f);
	else if (src_type == PixelType::HALF && dst_type == PixelType::BYTE)
		dither.half_to_byte(src_w, dst_b, tmp_f);
	else if (src_type == PixelType::HALF && dst_type == PixelType::WORD)
		dither.half_to_word(src_w, dst_w, tmp_f);
	else if (src_type == PixelType::FLOAT && dst_type == PixelType::BYTE)
		dither.float_to_byte(src_f, dst_b, tmp_f);
	else if (src_type == PixelType::FLOAT && dst_type == PixelType::WORD)
		dither.float_to_word(src_f, dst_w, tmp_f);
	else
		throw ZimgUnsupportedError{ "no conversion found between pixel types" };
}

void convert_depth(const DepthConvert &depth, const ImagePlane<const void> &src, const ImagePlane<void> &dst)
{
	ImagePlane<const uint8_t> src_b = plane_cast<const uint8_t>(src);
	ImagePlane<const uint16_t> src_w = plane_cast<const uint16_t>(src);
	ImagePlane<const float> src_f = plane_cast<const float>(src);

	//ImagePlane<uint8_t> dst_b = plane_cast<uint8_t>(dst);
	ImagePlane<uint16_t> dst_w = plane_cast<uint16_t>(dst);
	ImagePlane<float> dst_f = plane_cast<float>(dst);

	PixelType src_type = src.format().type;
	PixelType dst_type = dst.format().type;

	if (src_type == PixelType::BYTE && dst_type == PixelType::HALF)
		invoke_depth(depth, &DepthConvert::byte_to_half, src_b, dst_w, src.format());
	else if (src_type == PixelType::BYTE && dst_type == PixelType::FLOAT)
		invoke_depth(depth, &DepthConvert::byte_to_float, src_b, dst_f, src.format());
	else if (src_type == PixelType::WORD && dst_type == PixelType::HALF)
		invoke_depth(depth, &DepthConvert::word_to_half, src_w, dst_w, src.format());
	else if (src_type == PixelType::WORD && dst_type == PixelType::FLOAT)
		invoke_depth(depth, &DepthConvert::word_to_float, src_w, dst_f, src.format());
	else if (src_type == PixelType::HALF && dst_type == PixelType::FLOAT)
		invoke_depth(depth, &DepthConvert::half_to_float, src_w, dst_f);
	else if (src_type == PixelType::FLOAT && dst_type == PixelType::HALF)
		invoke_depth(depth, &DepthConvert::float_to_half, src_f, dst_w);
	else
		throw ZimgUnsupportedError{ "no conversion found between pixel types" };
}

} // namespace


Depth::Depth(DitherType type, CPUClass cpu) try :
	m_depth{ create_depth_convert(cpu) },
	m_dither{ create_dither_convert(type, cpu) },
	m_error_diffusion{ type == DitherType::DITHER_ERROR_DIFFUSION }
{
}
catch (const std::bad_alloc &)
{
	throw ZimgOutOfMemory{};
}

size_t Depth::tmp_size(int width) const
{
	return m_error_diffusion ? (width + 2) * 2 : 0;
}

void Depth::process(const ImagePlane<const void> &src, const ImagePlane<void> &dst, void *tmp) const
{
	if (dst.format().type >= PixelType::HALF)
		convert_depth(*m_depth, src, dst);
	else
		convert_dithered(*m_dither, src, dst, tmp);
}

} // namespace depth
} // namespace zimg
