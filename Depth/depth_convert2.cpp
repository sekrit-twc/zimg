#include <algorithm>
#include <cstdint>
#include <utility>
#include "Common/align.h"
#include "Common/except.h"
#include "Common/linebuffer.h"
#include "Common/pixel.h"
#include "depth_convert2.h"
#include "quantize.h"

namespace zimg {;
namespace depth {;

namespace {;

template <class T>
void integer_to_float(const void *src, void *dst, float scale, float offset, unsigned width)
{
	const T *src_p = reinterpret_cast<const T *>(src);
	float *dst_p = reinterpret_cast<float *>(dst);

	std::transform(src_p, src_p + width, dst_p, [=](T x){ return (float)x * scale + offset; });
}

void half_to_float_n(const void *src, void *dst, float, float, unsigned width)
{
	const uint16_t *src_p = reinterpret_cast<const uint16_t *>(src);
	float *dst_p = reinterpret_cast<float *>(dst);

	std::transform(src_p, src_p + width, dst_p, half_to_float);
}

void float_to_half_n(const void *src, void *dst, unsigned width)
{
	const float *src_p = reinterpret_cast<const float *>(src);
	uint16_t *dst_p = reinterpret_cast<uint16_t *>(dst);

	std::transform(src_p, src_p + width, dst_p, float_to_half);
}

std::pair<DepthConvert2::func_type, DepthConvert2::f16c_func_type> select_func(const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu)
{
	DepthConvert2::func_type func = nullptr;
	DepthConvert2::f16c_func_type f16c = nullptr;

	switch (pixel_in.type) {
	case PixelType::BYTE:
		func = integer_to_float<uint8_t>;
		break;
	case PixelType::WORD:
		func = integer_to_float<uint16_t>;
		break;
	case PixelType::HALF:
		func = half_to_float_n;
		break;
	default:
		break;
	}

	if (pixel_out.type == PixelType::HALF)
		f16c = float_to_half_n;

	if (pixel_in == pixel_out) {
		func = nullptr;
		f16c = nullptr;
	}

	return{ func, f16c };
}

} // namespace


DepthConvert2::DepthConvert2(unsigned width, unsigned height, const PixelFormat &pixel_in, const PixelFormat &pixel_out, CPUClass cpu) :
	m_func{},
	m_f16c{},
	m_pixel_in{ pixel_in.type },
	m_pixel_out{ pixel_out.type },
	m_scale{},
	m_offset{},
	m_width{ width },
	m_height{ height }
{
	if (pixel_out.type != PixelType::HALF && pixel_out.type != PixelType::FLOAT)
		throw zimg::error::InternalError{ "DepthConvert only converts to floating point types" };

	int32_t range = pixel_in.type < PixelType::HALF ? integer_range(pixel_in.depth, pixel_in.fullrange, pixel_in.chroma) : 1;
	int32_t offset = pixel_in.type < PixelType::HALF ? integer_offset(pixel_in.depth, pixel_in.fullrange, pixel_in.chroma) : 0;

	auto impl = select_func(pixel_in, pixel_out, cpu);

	m_func = impl.first;
	m_f16c = impl.second;

	m_scale = (float)(1.0 / range);
	m_offset = (float)(-offset * (1.0 / range));
}

ZimgFilterFlags DepthConvert2::get_flags() const
{
	ZimgFilterFlags flags{};

	flags.same_row = true;
	flags.in_place = !(m_func && m_f16c) && (pixel_size(m_pixel_in) >= pixel_size(m_pixel_out));

	return flags;
}

IZimgFilter::image_attributes DepthConvert2::get_image_attributes() const
{
	return{ m_width, m_height, m_pixel_out };
}

size_t DepthConvert2::get_tmp_size(unsigned left, unsigned right) const
{
	return (m_func && m_f16c) ? align(right - left, AlignmentOf<float>::value) * sizeof(float) : 0;
}

void DepthConvert2::process(void *, const ZimgImageBufferConst &src, const ZimgImageBuffer &dst, void *tmp, unsigned i, unsigned left, unsigned right) const
{
	LineBuffer<const void> src_buf{ src };
	LineBuffer<void> dst_buf{ dst };

	const void *src_p = reinterpret_cast<const char *>(src_buf[i]) + left * pixel_size(m_pixel_in);
	void *dst_p = reinterpret_cast<char *>(dst_buf[i]) + left * pixel_size(m_pixel_out);

	if (!m_func && !m_f16c) {
		if (src_p != dst_p)
			std::copy_n(reinterpret_cast<const char *>(src_p), (right - left) * pixel_size(m_pixel_out), reinterpret_cast<char *>(dst_p));
	} else {
		if (m_func) {
			if (m_f16c)
				dst_p = tmp;

			m_func(src_p, dst_p, m_scale, m_offset, right - left);

			src_p = dst_p;
			dst_p = reinterpret_cast<char *>(dst_buf[i]) + left * pixel_size(m_pixel_out);
		}

		if (m_f16c)
			m_f16c(src_p, dst_p, right - left);
	}
}

} // namespace depth
} // namespace zimg
