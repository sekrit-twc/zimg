#ifdef ZIMG_X86_AVX512

#include <memory>
#include <immintrin.h>
#include "common/make_unique.h"
#include "common/pixel.h"
#include "resize_impl_x86.h"

#define mm512_dpwssd_epi32(src, a, b) _mm512_dpwssd_epi32((src), (a), (b))
#include "resize_impl_avx512_common.h"

namespace zimg {
namespace resize {

std::unique_ptr<graph::ImageFilter> create_resize_impl_h_avx512_vnni(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

#ifndef ZIMG_RESIZE_NO_PERMUTE
	if (type == PixelType::WORD)
		ret = ResizeImplH_Permute_U16_AVX512::create(context, height, depth);
#endif

	if (!ret) {
		if (type == PixelType::WORD)
			ret = ztd::make_unique<ResizeImplH_U16_AVX512>(context, height, depth);
	}

	return ret;
}

std::unique_ptr<graph::ImageFilter> create_resize_impl_v_avx512_vnni(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == PixelType::WORD)
		ret = ztd::make_unique<ResizeImplV_U16_AVX512>(context, width, depth);

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86_AVX512
