#ifdef ZIMG_ARM

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <arm_neon.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "common/checked_int.h"
#include "common/except.h"
#include "common/make_unique.h"
#include "common/pixel.h"
#include "resize/resize_impl.h"
#include "resize_impl_arm.h"

#include "common/arm/neon_util.h"

namespace zimg {
namespace resize {

namespace {

void transpose_line_4x4_f32(float * RESTRICT dst, const float *src_p0, const float *src_p1, const float *src_p2, const float *src_p3, unsigned left, unsigned right)
{
	for (unsigned j = left; j < right; j += 4) {
		float32x4_t x0, x1, x2, x3;

		x0 = vld1q_f32(src_p0 + j);
		x1 = vld1q_f32(src_p1 + j);
		x2 = vld1q_f32(src_p2 + j);
		x3 = vld1q_f32(src_p3 + j);

		neon_transpose4_f32(x0, x1, x2, x3);

		vst1q_f32(dst + 0, x0);
		vst1q_f32(dst + 4, x1);
		vst1q_f32(dst + 8, x2);
		vst1q_f32(dst + 12, x3);

		dst += 16;
	}
}


template <unsigned FWidth, unsigned Tail>
inline FORCE_INLINE float32x4_t resize_line4_h_f32_neon_xiter(unsigned j,
                                                              const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                                                              const float * RESTRICT src, unsigned src_base)
{
	const float *filter_coeffs = filter_data + j * filter_stride;
	const float *src_p = src + (filter_left[j] - src_base) * 4;

	float32x4_t accum0 = vdupq_n_f32(0.0f);
	float32x4_t accum1 = vdupq_n_f32(0.0f);
	float32x4_t x, c, coeffs;

	unsigned k_end = FWidth ? FWidth - Tail : floor_n(filter_width, 4);

	for (unsigned k = 0; k < k_end; k += 4) {
		coeffs = vld1q_f32(filter_coeffs + k);

		c = vdupq_laneq_f32(coeffs, 0);
		x = vld1q_f32(src_p + 0);
		accum0 = vfmaq_f32(accum0, c, x);

		c = vdupq_laneq_f32(coeffs, 1);
		x = vld1q_f32(src_p + 4);
		accum1 = vfmaq_f32(accum1, c, x);

		c = vdupq_laneq_f32(coeffs, 2);
		x = vld1q_f32(src_p + 8);
		accum0 = vfmaq_f32(accum0, c, x);

		c = vdupq_laneq_f32(coeffs, 3);
		x = vld1q_f32(src_p + 12);
		accum1 = vfmaq_f32(accum1, c, x);

		src_p += 16;
	}

	if (Tail >= 1) {
		coeffs = vld1q_f32(filter_coeffs + k_end);

		c = vdupq_laneq_f32(coeffs, 0);
		x = vld1q_f32(src_p + 0);
		accum0 = vfmaq_f32(accum0, c, x);
	}
	if (Tail >= 2) {
		c = vdupq_laneq_f32(coeffs, 1);
		x = vld1q_f32(src_p + 4);
		accum1 = vfmaq_f32(accum1, c, x);
	}
	if (Tail >= 3) {
		c = vdupq_laneq_f32(coeffs, 2);
		x = vld1q_f32(src_p + 8);
		accum0 = vfmaq_f32(accum0, c, x);
	}
	if (Tail >= 4) {
		c = vdupq_laneq_f32(coeffs, 3);
		x = vld1q_f32(src_p + 12);
		accum1 = vfmaq_f32(accum1, c, x);
	}

	if (!FWidth || FWidth >= 2)
		accum0 = vaddq_f32(accum0, accum1);

	return accum0;
}

template <unsigned FWidth, unsigned Tail>
void resize_line4_h_f32_neon(const unsigned * RESTRICT filter_left, const float * RESTRICT filter_data, unsigned filter_stride, unsigned filter_width,
                            const float * RESTRICT src, float * const * RESTRICT dst, unsigned src_base, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	float *dst_p0 = dst[0];
	float *dst_p1 = dst[1];
	float *dst_p2 = dst[2];
	float *dst_p3 = dst[3];

#define XITER resize_line4_h_f32_neon_xiter<FWidth, Tail>
#define XARGS filter_left, filter_data, filter_stride, filter_width, src, src_base
	for (unsigned j = left; j < vec_left; ++j) {
		float32x4_t x = XITER(j, XARGS);
		neon_scatter_f32(dst_p0 + j, dst_p1 + j, dst_p2 + j, dst_p3 + j, x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		float32x4_t x0, x1, x2, x3;

		x0 = XITER(j + 0, XARGS);
		x1 = XITER(j + 1, XARGS);
		x2 = XITER(j + 2, XARGS);
		x3 = XITER(j + 3, XARGS);

		neon_transpose4_f32(x0, x1, x2, x3);

		vst1q_f32(dst_p0 + j, x0);
		vst1q_f32(dst_p1 + j, x1);
		vst1q_f32(dst_p2 + j, x2);
		vst1q_f32(dst_p3 + j, x3);
	}

	for (unsigned j = vec_right; j < right; ++j) {
		float32x4_t x = XITER(j, XARGS);
		neon_scatter_f32(dst_p0 + j, dst_p1 + j, dst_p2 + j, dst_p3 + j, x);
	}
#undef XITER
#undef XARGS
}

const decltype(&resize_line4_h_f32_neon<0, 0>) resize_line4_h_f32_neon_jt_small[] = {
	resize_line4_h_f32_neon<1, 1>,
	resize_line4_h_f32_neon<2, 2>,
	resize_line4_h_f32_neon<3, 3>,
	resize_line4_h_f32_neon<4, 4>,
	resize_line4_h_f32_neon<5, 1>,
	resize_line4_h_f32_neon<6, 2>,
	resize_line4_h_f32_neon<7, 3>,
	resize_line4_h_f32_neon<8, 4>
};

const decltype(&resize_line4_h_f32_neon<0, 0>) resize_line4_h_f32_neon_jt_large[] = {
	resize_line4_h_f32_neon<0, 0>,
	resize_line4_h_f32_neon<0, 1>,
	resize_line4_h_f32_neon<0, 2>,
	resize_line4_h_f32_neon<0, 3>
};


template <unsigned N, bool UpdateAccum>
inline FORCE_INLINE float32x4_t resize_line_v_f32_neon_xiter(unsigned j,
                                                             const float *src_p0, const float *src_p1, const float *src_p2, const float *src_p3,
                                                             const float *src_p4, const float *src_p5, const float *src_p6, const float *src_p7, float * RESTRICT accum_p,
                                                             const float32x4_t &c0, const float32x4_t &c1, const float32x4_t &c2, const float32x4_t &c3,
                                                             const float32x4_t &c4, const float32x4_t &c5, const float32x4_t &c6, const float32x4_t &c7)
{
	float32x4_t accum0 = vdupq_n_f32(0.0f);
	float32x4_t accum1 = vdupq_n_f32(0.0f);
	float32x4_t x;

	if (N >= 0) {
		x = vld1q_f32(src_p0 + j);
		accum0 = UpdateAccum ? vfmaq_f32(vld1q_f32(accum_p + j), c0, x) : vmulq_f32(c0, x);
	}
	if (N >= 1) {
		x = vld1q_f32(src_p1 + j);
		accum1 = vmulq_f32(c1, x);
	}
	if (N >= 2) {
		x = vld1q_f32(src_p2 + j);
		accum0 = vfmaq_f32(accum0, c2, x);
	}
	if (N >= 3) {
		x = vld1q_f32(src_p3 + j);
		accum1 = vfmaq_f32(accum1, c3, x);
	}
	if (N >= 4) {
		x = vld1q_f32(src_p4 + j);
		accum0 = vfmaq_f32(accum0, c4, x);
	}
	if (N >= 5) {
		x = vld1q_f32(src_p5 + j);
		accum1 = vfmaq_f32(accum1, c5, x);
	}
	if (N >= 6) {
		x = vld1q_f32(src_p6 + j);
		accum0 = vfmaq_f32(accum0, c6, x);
	}
	if (N >= 7) {
		x = vld1q_f32(src_p7 + j);
		accum1 = vfmaq_f32(accum1, c7, x);
	}

	accum0 = (N >= 1) ? vaddq_f32(accum0, accum1) : accum0;
	return accum0;
}

template <unsigned N, bool UpdateAccum>
void resize_line_v_f32_neon(const float * RESTRICT filter_data, const float * const * RESTRICT src, float * RESTRICT dst, unsigned left, unsigned right)
{
	const float *src_p0 = src[0];
	const float *src_p1 = src[1];
	const float *src_p2 = src[2];
	const float *src_p3 = src[3];
	const float *src_p4 = src[4];
	const float *src_p5 = src[5];
	const float *src_p6 = src[6];
	const float *src_p7 = src[7];

	unsigned vec_left = ceil_n(left, 4);
	unsigned vec_right = floor_n(right, 4);

	const float32x4_t c0 = vdupq_n_f32(filter_data[0]);
	const float32x4_t c1 = vdupq_n_f32(filter_data[1]);
	const float32x4_t c2 = vdupq_n_f32(filter_data[2]);
	const float32x4_t c3 = vdupq_n_f32(filter_data[3]);
	const float32x4_t c4 = vdupq_n_f32(filter_data[4]);
	const float32x4_t c5 = vdupq_n_f32(filter_data[5]);
	const float32x4_t c6 = vdupq_n_f32(filter_data[6]);
	const float32x4_t c7 = vdupq_n_f32(filter_data[7]);

	float32x4_t accum;

#define XITER resize_line_v_f32_neon_xiter<N, UpdateAccum>
#define XARGS src_p0, src_p1, src_p2, src_p3, src_p4, src_p5, src_p6, src_p7, dst, c0, c1, c2, c3, c4, c5, c6, c7
	if (left != vec_left) {
		accum = XITER(vec_left - 4, XARGS);
		neon_store_idxhi_f32(dst + vec_left - 4, accum, left % 4);
	}

	for (unsigned j = vec_left; j < vec_right; j += 4) {
		accum = XITER(j, XARGS);
		vst1q_f32(dst + j, accum);
	}

	if (right != vec_right) {
		accum = XITER(vec_right, XARGS);
		neon_store_idxlo_f32(dst + vec_right, accum, right % 4);
	}
#undef XITER
#undef XARGS
}

struct resize_line_v_f32_neon_jt {
	typedef decltype(&resize_line_v_f32_neon<0, false>) func_type;

	static const func_type table_a[8];
	static const func_type table_b[8];
};

const resize_line_v_f32_neon_jt::func_type resize_line_v_f32_neon_jt::table_a[8] = {
	resize_line_v_f32_neon<0, false>,
	resize_line_v_f32_neon<1, false>,
	resize_line_v_f32_neon<2, false>,
	resize_line_v_f32_neon<3, false>,
	resize_line_v_f32_neon<4, false>,
	resize_line_v_f32_neon<5, false>,
	resize_line_v_f32_neon<6, false>,
	resize_line_v_f32_neon<7, false>,
};

const resize_line_v_f32_neon_jt::func_type resize_line_v_f32_neon_jt::table_b[8] = {
	resize_line_v_f32_neon<0, true>,
	resize_line_v_f32_neon<1, true>,
	resize_line_v_f32_neon<2, true>,
	resize_line_v_f32_neon<3, true>,
	resize_line_v_f32_neon<4, true>,
	resize_line_v_f32_neon<5, true>,
	resize_line_v_f32_neon<6, true>,
	resize_line_v_f32_neon<7, true>,
};


class ResizeImplH_F32_Neon final : public ResizeImplH {
	decltype(&resize_line4_h_f32_neon<0, 0>) m_func;
public:
	ResizeImplH_F32_Neon(const FilterContext &filter, unsigned height) :
		ResizeImplH(filter, image_attributes{ filter.filter_rows, height, PixelType::FLOAT }),
		m_func{}
	{
		if (filter.filter_width <= 8)
			m_func = resize_line4_h_f32_neon_jt_small[filter.filter_width - 1];
		else
			m_func = resize_line4_h_f32_neon_jt_large[filter.filter_width % 4];
	}

	unsigned get_simultaneous_lines() const override { return 4; }

	size_t get_tmp_size(unsigned left, unsigned right) const override
	{
		auto range = get_required_col_range(left, right);

		try {
			checked_size_t size = (static_cast<checked_size_t>(range.second) - floor_n(range.first, 4) + 4) * sizeof(float) * 4;
			return size.get();
		} catch (const std::overflow_error &) {
			error::throw_<error::OutOfMemory>();
		}
	}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *tmp, unsigned i, unsigned left, unsigned right) const override
	{
		const auto &src_buf = graph::static_buffer_cast<const float>(*src);
		const auto &dst_buf = graph::static_buffer_cast<float>(*dst);
		auto range = get_required_col_range(left, right);

		const float *src_ptr[4] = { 0 };
		float *dst_ptr[4] = { 0 };
		float *transpose_buf = static_cast<float *>(tmp);
		unsigned height = get_image_attributes().height;

		src_ptr[0] = src_buf[std::min(i + 0, height - 1)];
		src_ptr[1] = src_buf[std::min(i + 1, height - 1)];
		src_ptr[2] = src_buf[std::min(i + 2, height - 1)];
		src_ptr[3] = src_buf[std::min(i + 3, height - 1)];

		transpose_line_4x4_f32(transpose_buf, src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], floor_n(range.first, 4), ceil_n(range.second, 4));

		dst_ptr[0] = dst_buf[std::min(i + 0, height - 1)];
		dst_ptr[1] = dst_buf[std::min(i + 1, height - 1)];
		dst_ptr[2] = dst_buf[std::min(i + 2, height - 1)];
		dst_ptr[3] = dst_buf[std::min(i + 3, height - 1)];

		m_func(m_filter.left.data(), m_filter.data.data(), m_filter.stride, m_filter.filter_width,
		       transpose_buf, dst_ptr, floor_n(range.first, 4), left, right);
	}
};


class ResizeImplV_F32_Neon final : public ResizeImplV {
public:
	ResizeImplV_F32_Neon(const FilterContext &filter, unsigned width) :
		ResizeImplV(filter, image_attributes{ width, filter.filter_rows, PixelType::FLOAT })
	{}

	void process(void *, const graph::ImageBuffer<const void> *src, const graph::ImageBuffer<void> *dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		const auto &src_buf = graph::static_buffer_cast<const float>(*src);
		const auto &dst_buf = graph::static_buffer_cast<float>(*dst);

		const float *filter_data = m_filter.data.data() + i * m_filter.stride;
		unsigned filter_width = m_filter.filter_width;
		unsigned src_height = m_filter.input_width;

		const float *src_lines[8] = { 0 };
		float *dst_line = dst_buf[i];

		{
			unsigned taps_remain = std::min(filter_width - 0, 8U);
			unsigned top = m_filter.left[i] + 0;

			src_lines[0] = src_buf[std::min(top + 0, src_height - 1)];
			src_lines[1] = src_buf[std::min(top + 1, src_height - 1)];
			src_lines[2] = src_buf[std::min(top + 2, src_height - 1)];
			src_lines[3] = src_buf[std::min(top + 3, src_height - 1)];
			src_lines[4] = src_buf[std::min(top + 4, src_height - 1)];
			src_lines[5] = src_buf[std::min(top + 5, src_height - 1)];
			src_lines[6] = src_buf[std::min(top + 6, src_height - 1)];
			src_lines[7] = src_buf[std::min(top + 7, src_height - 1)];

			resize_line_v_f32_neon_jt::table_a[taps_remain - 1](filter_data + 0, src_lines, dst_line, left, right);
		}

		for (unsigned k = 8; k < filter_width; k += 8) {
			unsigned taps_remain = std::min(filter_width - k, 8U);
			unsigned top = m_filter.left[i] + k;

			src_lines[0] = src_buf[std::min(top + 0, src_height - 1)];
			src_lines[1] = src_buf[std::min(top + 1, src_height - 1)];
			src_lines[2] = src_buf[std::min(top + 2, src_height - 1)];
			src_lines[3] = src_buf[std::min(top + 3, src_height - 1)];
			src_lines[4] = src_buf[std::min(top + 4, src_height - 1)];
			src_lines[5] = src_buf[std::min(top + 5, src_height - 1)];
			src_lines[6] = src_buf[std::min(top + 6, src_height - 1)];
			src_lines[7] = src_buf[std::min(top + 7, src_height - 1)];

			resize_line_v_f32_neon_jt::table_b[taps_remain - 1](filter_data + k, src_lines, dst_line, left, right);
		}
	}
};

} // namespace


std::unique_ptr<graph::ImageFilter> create_resize_impl_h_neon(const FilterContext &context, unsigned height, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == PixelType::FLOAT)
		ret = ztd::make_unique<ResizeImplH_F32_Neon>(context, height);
	else if (type == PixelType::WORD)
		ret = 0;

	return ret;
}

std::unique_ptr<graph::ImageFilter> create_resize_impl_v_neon(const FilterContext &context, unsigned width, PixelType type, unsigned depth)
{
	std::unique_ptr<graph::ImageFilter> ret;

	if (type == PixelType::FLOAT)
		ret = ztd::make_unique<ResizeImplV_F32_Neon>(context, width);
	else if (type == PixelType::WORD)
		ret = 0;

	return ret;
}

} // namespace resize
} // namespace zimg

#endif // ZIMG_ARM
