#if 0
#pragma once

#ifndef ZIMG_RESIZE_RESIZE_IMPL_H_
#define ZIMG_RESIZE_RESIZE_IMPL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "Common/linebuffer.h"
#include "Common/osdep.h"
#include "filter.h"

namespace zimg {;

enum class CPUClass;
enum class PixelType;

namespace resize {;

struct ScalarPolicy_U16 {
	typedef int32_t num_type;

	FORCE_INLINE int32_t coeff(const FilterContext &filter, ptrdiff_t row, ptrdiff_t k)
	{
		return filter.data_i16[row * filter.stride_i16 + k];
	}

	FORCE_INLINE int32_t load(const uint16_t *src)
	{
		uint16_t x = *src;
		return (int32_t)x + (int32_t)INT16_MIN; // Make signed.
	}

	FORCE_INLINE void store(uint16_t *dst, int32_t x)
	{
		// Convert from 16.14 to 16.0.
		x = ((x + (1 << 13)) >> 14) - (int32_t)INT16_MIN;

		// Clamp out of range values.
		x = std::max(std::min(x, (int32_t)UINT16_MAX), (int32_t)0);

		*dst = (uint16_t)x;
	}
};

struct ScalarPolicy_F32 {
	typedef float num_type;

	FORCE_INLINE float coeff(const FilterContext &filter, ptrdiff_t row, ptrdiff_t k)
	{
		return filter.data[row * filter.stride + k];
	}

	FORCE_INLINE float load(const float *src) { return *src; }

	FORCE_INLINE void store(float *dst, float x) { *dst = x; }
};

template <class T, class Policy>
inline FORCE_INLINE void filter_line_h_scalar(const FilterContext &filter, const LineBuffer<T> &src, LineBuffer<T> &dst,
											  unsigned i_begin, unsigned i_end, unsigned j_begin, unsigned j_end, Policy policy)
{
	typedef typename Policy::num_type num_type;

	for (unsigned i = i_begin; i < i_end; ++i) {
		for (unsigned j = j_begin; j < j_end; ++j) {
			unsigned left = filter.left[j];
			num_type accum = 0;

			for (unsigned k = 0; k < filter.filter_width; ++k) {
				num_type coeff = policy.coeff(filter, j, k);
				num_type x = policy.load(&src[i][left + k]);

				accum += coeff * x;
			}

			policy.store(&dst[i][j], accum);
		}
	}
}

template <class T, class Policy>
inline FORCE_INLINE void filter_line_v_scalar(const FilterContext &filter, const LineBuffer<T> &src, LineBuffer<T> &dst,
											  unsigned i_begin, unsigned i_end, unsigned j_begin, unsigned j_end, Policy policy)
{
	typedef typename Policy::num_type num_type;

	for (unsigned i = i_begin; i < i_end; ++i) {
		unsigned top = filter.left[i];

		for (unsigned j = j_begin; j < j_end; ++j) {
			num_type accum = 0;

			for (unsigned k = 0; k < filter.filter_width; ++k) {
				num_type coeff = policy.coeff(filter, i, k);
				num_type x = policy.load(&src[top + k][j]);

				accum += coeff * x;
			}

			policy.store(&dst[i][j], accum);
		}
	}
}

class ResizeImpl {
	bool m_horizontal;
protected:
	FilterContext m_filter;

	ResizeImpl(const FilterContext &filter, bool horizontal);
public:
	virtual ~ResizeImpl() = 0;

	virtual bool pixel_supported(PixelType type) const;

	virtual size_t tmp_size(PixelType type, unsigned width) const;

	virtual unsigned input_buffering(PixelType type) const;

	virtual unsigned output_buffering(PixelType type) const;

	unsigned dependent_line(unsigned n) const;

	virtual void process_u16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const = 0;

	virtual void process_f16(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, unsigned n, void *tmp) const = 0;

	virtual void process_f32(const LineBuffer<float> &src, LineBuffer<float> &dst, unsigned n, void *tmp) const = 0;
};

/**
 * Create a concrete ResizeImpl.
 *
 * @param f filter
 * @param horizontal whether filter is horizontal
 * @param src_dim input dimension
 * @param dst_dim output dimension
 * @param shift input shift
 * @param subwidth input window size
 */
ResizeImpl *create_resize_impl(const Filter &f, bool horizontal, int src_dim, int dst_dim, double shift, double subwidth, CPUClass cpu);

} // namespace resize
} // namespace zimg

#endif // ZIMG_RESIZE_RESIZE_IMPL_H_
#endif
