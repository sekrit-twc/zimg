#ifdef ZIMG_LOONGARCH

#include <cstdint>
#include <lsxintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "depth/quantize.h"
#include "dither_loongarch.h"

#include "common/loongarch/lsx_util.h"

namespace zimg::depth {

namespace {

// Load 8 pixels as float in two 4-lane vectors.

struct LoadU8 {
	static constexpr unsigned stride = 1;

	static inline FORCE_INLINE void load8(const void *ptr, __m128 &lo,
	                                      __m128 &hi, unsigned n = 8)
	{
		(void)n;
		const uint8_t *p = static_cast<const uint8_t *>(ptr);
		__m128i x = __lsx_vld((void *)p, 0);
		__m128i lo_w = __lsx_vsllwil_hu_bu(x, 0);
		lo = (__m128)__lsx_vffint_s_wu(__lsx_vsllwil_wu_hu(lo_w, 0));
		hi = (__m128)__lsx_vffint_s_wu(__lsx_vexth_wu_hu(lo_w));
	}
};

struct LoadU16 {
	static constexpr unsigned stride = 2;

	static inline FORCE_INLINE void load8(const void *ptr, __m128 &lo,
	                                      __m128 &hi, unsigned n = 8)
	{
		(void)n;
		const uint16_t *p = static_cast<const uint16_t *>(ptr);
		__m128i x = __lsx_vld((void *)p, 0);
		lo = (__m128)__lsx_vffint_s_wu(__lsx_vsllwil_wu_hu(x, 0));
		hi = (__m128)__lsx_vffint_s_wu(__lsx_vexth_wu_hu(x));
	}
};

struct LoadF16 {
	static constexpr unsigned stride = 2;

	static inline FORCE_INLINE void load8(const void *ptr, __m128 &lo,
	                                      __m128 &hi, unsigned n = 8)
	{
		(void)n;
		const uint16_t *p = static_cast<const uint16_t *>(ptr);

		// No native fp16->fp32 SIMD carries the exact bit semantics of the
		// reference (NaN payloads); so convert each half scalar and pack.
		float v[8];
		for (unsigned i = 0; i < 8; ++i)
			v[i] = half_to_float(p[i]);

		lo = (__m128)__lsx_vld((void *)(v + 0), 0);
		hi = (__m128)__lsx_vld((void *)(v + 4), 0);
	}
};

struct LoadF32 {
	static constexpr unsigned stride = 4;

	static inline FORCE_INLINE void load8(const void *ptr, __m128 &lo,
	                                      __m128 &hi, unsigned n = 8)
	{
		const float *p = static_cast<const float *>(ptr);
		lo = (__m128)__lsx_vld((void *)(p + 0), 0);
		hi = n > 4 ? (__m128)__lsx_vld((void *)(p + 4), 0)
		           : (__m128)__lsx_vldi(0);
	}
};


struct StoreU8 {
	static constexpr unsigned stride = 1;

	static inline FORCE_INLINE void store8(uint8_t *ptr, __m128i x)
	{
		// Narrow 8 clamped 16-bit values to bytes; store the low 8.
		__m128i b = __lsx_vpickev_b(x, x);
		__lsx_vstelm_d(b, ptr, 0, 0);
	}

	static inline FORCE_INLINE void store8_idxlo(uint8_t *ptr,
	                                              __m128i x, unsigned idx)
	{
		// Store the low idx bytes; keep the remaining low bytes from dst
		// (utilize a masked read-modify-write and write back the low 8 bytes).
		__m128i b = __lsx_vpickev_b(x, x);
		__m128i orig = __lsx_vld(ptr, 0);
		__m128i mask = __lsx_vld((void *)&lsx_mask_table[idx][0], 0);
		__m128i r = __lsx_vor_v(__lsx_vand_v(mask, b),
		                       __lsx_vandn_v(mask, orig));

		__lsx_vstelm_d(r, ptr, 0, 0);
	}

	static inline FORCE_INLINE void store8_idxhi(uint8_t *ptr,
	                                              __m128i x, unsigned idx)
	{
		// Store the bytes from idx onward; keep the leading bytes from dst
		// (utilize a masked read-modify-write and write back the low 8 bytes).
		__m128i b = __lsx_vpickev_b(x, x);
		__m128i orig = __lsx_vld(ptr, 0);
		__m128i mask = __lsx_vld((void *)&lsx_mask_table[idx][0], 0);
		__m128i r = __lsx_vor_v(__lsx_vandn_v(mask, b),
		                       __lsx_vand_v(mask, orig));

		__lsx_vstelm_d(r, ptr, 0, 0);
	}
};

struct StoreU16 {
	static constexpr unsigned stride = 2;

	static inline FORCE_INLINE void store8(uint8_t *ptr, __m128i x)
	{
		__lsx_vst(x, ptr, 0);
	}

	static inline FORCE_INLINE void store8_idxlo(uint8_t *ptr,
	                                              __m128i x, unsigned idx)
	{
		lsx_store_idxlo_u16(reinterpret_cast<uint16_t *>(ptr), x, idx);
	}

	static inline FORCE_INLINE void store8_idxhi(uint8_t *ptr,
	                                              __m128i x, unsigned idx)
	{
		lsx_store_idxhi_u16(reinterpret_cast<uint16_t *>(ptr), x, idx);
	}
};


inline FORCE_INLINE __m128i ordered_dither_lsx_xiter(
	__m128 lo,
	__m128 hi,
	unsigned j,
	const float *dither,
	unsigned dither_offset,
	unsigned dither_mask,
	const __m128 &scale,
	const __m128 &offset,
	const __m128i &out_max)
{
	__m128 dith;

	// j is always a multiple of 8 (vec_left/vec_right are ceil/floor to 8) and
	// dither_offset is 0, so (dither_offset+j) & dither_mask (with a power-of-two
	// mask) stays in {0, 8} within a 16-float dither row. The four contiguous
	// elements k=0..3, and k=4..7, never wrap the row boundary, so they can be
	// loaded directly as vectors instead of scalar-gathering each value.
	lo = __lsx_vfmadd_s(lo, scale, offset);
	dith = (__m128)__lsx_vld((void *)(dither + ((dither_offset + j) & dither_mask)), 0);
	lo = __lsx_vfadd_s(lo, dith);

	hi = __lsx_vfmadd_s(hi, scale, offset);
	dith = (__m128)__lsx_vld((void *)(dither + ((dither_offset + j + 4) & dither_mask)), 0);
	hi = __lsx_vfadd_s(hi, dith);

	__m128i lo_dw = __lsx_vftintrne_w_s(lo);
	__m128i hi_dw = __lsx_vftintrne_w_s(hi);

	__m128i x = __lsx_vpickev_h(hi_dw, lo_dw);
	x = __lsx_vmin_hu(x, out_max);

	return x;
}

template <class Load, class Store>
void ordered_dither_lsx_impl(const float *dither, unsigned dither_offset,
                             unsigned dither_mask, const void *src, void *dst,
                             float scale, float offset, unsigned bits,
                             unsigned left, unsigned right)
{
	const uint8_t *src_p = static_cast<const uint8_t *>(src);
	uint8_t *dst_p = static_cast<uint8_t *>(dst);

	unsigned vec_left = ceil_n(left, 8);
	unsigned vec_right = floor_n(right, 8);

	float sc = scale;
	float off = offset;
	const __m128 scale_x4 = (__m128)__lsx_vldrepl_w(&sc, 0);
	const __m128 offset_x4 = (__m128)__lsx_vldrepl_w(&off, 0);
	unsigned short max16 = static_cast<unsigned short>((1u << bits) - 1);
	const __m128i out_max = __lsx_vldrepl_h(&max16, 0);

	__m128 lo, hi;

#define XARGS dither, dither_offset, dither_mask, scale_x4, offset_x4, out_max
	if (left != vec_left) {
		Load::load8(src_p + (vec_left - 8) * Load::stride, lo, hi);
		__m128i x = ordered_dither_lsx_xiter(
			lo, hi, vec_left - 8, XARGS);
		Store::store8_idxhi(
			dst_p + (vec_left - 8) * Store::stride, x, left % 8);
	}
	for (unsigned j = vec_left; j < vec_right; j += 8) {
		Load::load8(src_p + j * Load::stride, lo, hi);
		__m128i x = ordered_dither_lsx_xiter(lo, hi, j, XARGS);
		Store::store8(dst_p + j * Store::stride, x);
	}
	if (right != vec_right) {
		Load::load8(src_p + vec_right * Load::stride, lo, hi,
		            right % 8);
		__m128i x = ordered_dither_lsx_xiter(lo, hi, vec_right, XARGS);
		Store::store8_idxlo(
			dst_p + vec_right * Store::stride, x, right % 8);
	}
#undef XARGS
}

} // namespace


void ordered_dither_b2b_lsx(const float *dither, unsigned dither_offset,
                           unsigned dither_mask, const void *src, void *dst,
                           float scale, float offset, unsigned bits,
                           unsigned left, unsigned right)
{
	ordered_dither_lsx_impl<LoadU8, StoreU8>(
		dither, dither_offset, dither_mask, src, dst, scale, offset,
		bits, left, right);
}

void ordered_dither_b2w_lsx(const float *dither, unsigned dither_offset,
                           unsigned dither_mask, const void *src, void *dst,
                           float scale, float offset, unsigned bits,
                           unsigned left, unsigned right)
{
	ordered_dither_lsx_impl<LoadU8, StoreU16>(
		dither, dither_offset, dither_mask, src, dst, scale, offset,
		bits, left, right);
}

void ordered_dither_w2b_lsx(const float *dither, unsigned dither_offset,
                           unsigned dither_mask, const void *src, void *dst,
                           float scale, float offset, unsigned bits,
                           unsigned left, unsigned right)
{
	ordered_dither_lsx_impl<LoadU16, StoreU8>(
		dither, dither_offset, dither_mask, src, dst, scale, offset,
		bits, left, right);
}

void ordered_dither_w2w_lsx(const float *dither, unsigned dither_offset,
                           unsigned dither_mask, const void *src, void *dst,
                           float scale, float offset, unsigned bits,
                           unsigned left, unsigned right)
{
	ordered_dither_lsx_impl<LoadU16, StoreU16>(
		dither, dither_offset, dither_mask, src, dst, scale, offset,
		bits, left, right);
}

void ordered_dither_h2b_lsx(const float *dither, unsigned dither_offset,
                           unsigned dither_mask, const void *src, void *dst,
                           float scale, float offset, unsigned bits,
                           unsigned left, unsigned right)
{
	ordered_dither_lsx_impl<LoadF16, StoreU8>(
		dither, dither_offset, dither_mask, src, dst, scale, offset,
		bits, left, right);
}

void ordered_dither_h2w_lsx(const float *dither, unsigned dither_offset,
                           unsigned dither_mask, const void *src, void *dst,
                           float scale, float offset, unsigned bits,
                           unsigned left, unsigned right)
{
	ordered_dither_lsx_impl<LoadF16, StoreU16>(
		dither, dither_offset, dither_mask, src, dst, scale, offset,
		bits, left, right);
}

void ordered_dither_f2b_lsx(const float *dither, unsigned dither_offset,
                           unsigned dither_mask, const void *src, void *dst,
                           float scale, float offset, unsigned bits,
                           unsigned left, unsigned right)
{
	ordered_dither_lsx_impl<LoadF32, StoreU8>(
		dither, dither_offset, dither_mask, src, dst, scale, offset,
		bits, left, right);
}

void ordered_dither_f2w_lsx(const float *dither, unsigned dither_offset,
                           unsigned dither_mask, const void *src, void *dst,
                           float scale, float offset, unsigned bits,
                           unsigned left, unsigned right)
{
	ordered_dither_lsx_impl<LoadF32, StoreU16>(
		dither, dither_offset, dither_mask, src, dst, scale, offset,
		bits, left, right);
}

} // namespace zimg::depth

#endif // ZIMG_LOONGARCH
