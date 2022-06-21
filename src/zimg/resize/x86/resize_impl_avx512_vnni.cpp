#ifdef ZIMG_X86_AVX512

#include <memory>
#include <immintrin.h>
#include "common/pixel.h"
#include "resize_impl_x86.h"

#define mm512_dpwssd_epi32(src, a, b) _mm512_dpwssd_epi32((src), (a), (b))
#include "resize_impl_avx512_common.h"

namespace zimg {
namespace resize {

std::unique_ptr<graphengine::Filter> create_resize_impl_h_ge_avx512_vnni(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

#ifndef ZIMG_RESIZE_NO_PERMUTE
	if (type == PixelType::WORD)
		ret = ResizeImplH_GE_Permute_U16_AVX512::create(context, height, depth);
#endif

	if (!ret) {
		if (type == PixelType::WORD)
			ret = std::make_unique<ResizeImplH_GE_U16_AVX512>(context, height, depth);
	}

	return ret;
}

std::unique_ptr<graphengine::Filter> create_resize_impl_v_ge_avx512_vnni(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graphengine::Filter> ret;

	if (type == PixelType::WORD)
		ret = std::make_unique<ResizeImplV_GE_U16_AVX512>(context, width, depth);

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_X86_AVX512
