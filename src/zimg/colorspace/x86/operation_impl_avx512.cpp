#ifdef ZIMG_X86_AVX512

#include <cfloat>
#include <immintrin.h>
#include "common/align.h"
#include "common/ccdep.h"
#include "colorspace/gamma.h"
#include "colorspace/operation_impl.h"
#include "gamma_constants_avx512.h"
#include "operation_impl_x86.h"

#include "common/x86/avx512_util.h"

namespace zimg::colorspace {

namespace {

inline FORCE_INLINE void matrix_filter_line_avx512_xiter(unsigned j, const float *src0, const float *src1, const float *src2,
                                                         const __m512 &c00, const __m512 &c01, const __m512 &c02,
                                                         const __m512 &c10, const __m512 &c11, const __m512 &c12,
                                                         const __m512 &c20, const __m512 &c21, const __m512 &c22,
                                                         __m512 &out0, __m512 &out1, __m512 &out2)
{
	__m512 a = _mm512_load_ps(src0 + j);
	__m512 b = _mm512_load_ps(src1 + j);
	__m512 c = _mm512_load_ps(src2 + j);
	__m512 x, y, z;

	x = _mm512_mul_ps(c00, a);
	x = _mm512_fmadd_ps(c01, b, x);
	x = _mm512_fmadd_ps(c02, c, x);
	out0 = x;

	y = _mm512_mul_ps(c10, a);
	y = _mm512_fmadd_ps(c11, b, y);
	y = _mm512_fmadd_ps(c12, c, y);
	out1 = y;

	z = _mm512_mul_ps(c20, a);
	z = _mm512_fmadd_ps(c21, b, z);
	z = _mm512_fmadd_ps(c22, c, z);
	out2 = z;
}

void matrix_filter_line_avx512(const float *matrix, const float * const * RESTRICT src, float * const * RESTRICT dst, unsigned left, unsigned right)
{
	const float *src0 = src[0];
	const float *src1 = src[1];
	const float *src2 = src[2];
	float *dst0 = dst[0];
	float *dst1 = dst[1];
	float *dst2 = dst[2];

	const __m512 c00 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 0));
	const __m512 c01 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 1));
	const __m512 c02 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 2));
	const __m512 c10 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 3));
	const __m512 c11 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 4));
	const __m512 c12 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 5));
	const __m512 c20 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 6));
	const __m512 c21 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 7));
	const __m512 c22 = _mm512_broadcastss_ps(_mm_load_ss(matrix + 8));
	__m512 out0, out1, out2;

	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

#define XITER matrix_filter_line_avx512_xiter
#define XARGS src0, src1, src2, c00, c01, c02, c10, c11, c12, c20, c21, c22, out0, out1, out2
	if (left != vec_left) {
		XITER(vec_left - 16, XARGS);
		__mmask16 mask = mmask16_set_hi(vec_left - left);

		_mm512_mask_store_ps(dst0 + vec_left - 16, mask, out0);
		_mm512_mask_store_ps(dst1 + vec_left - 16, mask, out1);
		_mm512_mask_store_ps(dst2 + vec_left - 16, mask, out2);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		XITER(j, XARGS);

		_mm512_store_ps(dst0 + j, out0);
		_mm512_store_ps(dst1 + j, out1);
		_mm512_store_ps(dst2 + j, out2);
	}

	if (right != vec_right) {
		XITER(vec_right, XARGS);
		__mmask16 mask = mmask16_set_lo(right - vec_right);

		_mm512_mask_store_ps(dst0 + vec_right, mask, out0);
		_mm512_mask_store_ps(dst1 + vec_right, mask, out1);
		_mm512_mask_store_ps(dst2 + vec_right, mask, out2);
	}
#undef XITER
#undef XARGS
}


template <class T, bool Prescale>
struct PowerFunction {
	static inline FORCE_INLINE __m512 func(__m512 x, __m512 scale)
	{
		constexpr bool ExtendedExponent = sizeof(T::table) / sizeof(T::table[0]) == 32;

		const __m512i exponent_min = _mm512_set1_epi32(127 - (ExtendedExponent ? 31 : 15));
		const __m512 two_minus_eps = _mm512_set1_ps(1.99999988f);
		const __m512i sign = _mm512_set1_epi32(0x80000000U);

		__m512 orig, mant, mantpart, exppart;
		__m512i exp;

		if (Prescale)
			x = _mm512_mul_ps(x, scale);

		orig = x;
		x = _mm512_range_ps(x, two_minus_eps, 0x08); // fabs(min(x, 2.0))

		// Decompose into mantissa and exponent.
		mant = _mm512_getmant_ps(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_nan);
		exp = _mm512_srli_epi32(_mm512_castps_si512(x), 23);
		exp = _mm512_max_epi32(exp, exponent_min);

		// Apply polynomial approximation to mantissa.
		mantpart = _mm512_set1_ps(T::horner[0]);
		mantpart = _mm512_fmadd_ps(mantpart, mant, _mm512_set1_ps(T::horner[1]));
		mantpart = _mm512_fmadd_ps(mantpart, mant, _mm512_set1_ps(T::horner[2]));
		mantpart = _mm512_fmadd_ps(mantpart, mant, _mm512_set1_ps(T::horner[3]));
		mantpart = _mm512_fmadd_ps(mantpart, mant, _mm512_set1_ps(T::horner[4]));
		mantpart = _mm512_fmadd_ps(mantpart, mant, _mm512_set1_ps(T::horner[5]));

		// Read f(2^e) from a 16 or 32-element LUT.
		if (ExtendedExponent)
			exppart = _mm512_permutex2var_ps(_mm512_load_ps(T::table), exp, _mm512_load_ps(T::table + 16));
		else
			exppart = _mm512_permutexvar_ps(exp, _mm512_load_ps(T::table));

		// f(m * 2^e) == f(m) * f(2^e)
		x = _mm512_mul_ps(mantpart, exppart);

		if (!Prescale)
			x = _mm512_mul_ps(x, scale);

		// copysign(x, orig)
		x = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(sign, _mm512_castps_si512(orig), _mm512_castps_si512(x), 0xCA));
		return x;
	}
};

template <class T, bool Eotf, bool Prescale>
struct SRGBPowerFunction {
	static inline FORCE_INLINE __m512 func(__m512 x, __m512 scale)
	{
		constexpr bool ExtendedExponent = sizeof(T::table) / sizeof(T::table[0]) == 32;

		const __m512 two_minus_eps = _mm512_set1_ps(1.99999988f);
		const __m512i sign = _mm512_set1_epi32(0x80000000U);

		__m512 orig, mant, mantpart, exppart;
		__m512i exp;
		__mmask16 mask;

		if (Prescale)
			x = _mm512_mul_ps(x, scale);

		orig = x;
		x = _mm512_range_ps(x, two_minus_eps, 0x08); // fabs(min(x, 2.0))

		// Check if the argument belongs to the linear or the power domain.
		mask = _mm512_cmp_ps_mask(x, _mm512_set1_ps(T::knee), _CMP_LE_OQ);

		// f(x) = (x * a + b) ^ p
		if (Eotf)
			x = _mm512_fmadd_ps(x, _mm512_set1_ps(T::power_scale), _mm512_set1_ps(T::power_offset));

		// Decompose into mantissa and exponent.
		mant = _mm512_getmant_ps(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_nan);
		exp = _mm512_srli_epi32(_mm512_castps_si512(x), 23); // Exponent range limit not needed because of mask.

		// Apply polynomial approximation to mantissa.
		mantpart = _mm512_set1_ps(T::horner[0]);
		mantpart = _mm512_fmadd_ps(mantpart, mant, _mm512_set1_ps(T::horner[1]));
		mantpart = _mm512_fmadd_ps(mantpart, mant, _mm512_set1_ps(T::horner[2]));
		mantpart = _mm512_fmadd_ps(mantpart, mant, _mm512_set1_ps(T::horner[3]));
		mantpart = _mm512_fmadd_ps(mantpart, mant, _mm512_set1_ps(T::horner[4]));
		mantpart = _mm512_fmadd_ps(mantpart, mant, _mm512_set1_ps(T::horner[5]));

		// Read f(2^e) from a 16-element LUT.
		exppart = _mm512_permutexvar_ps(exp, _mm512_load_ps(ExtendedExponent ? T::table + 16 : T::table));

		// f(m * 2^e) == f(m) * f(2^e)
		x = _mm512_mul_ps(mantpart, exppart);

		// f(x) = (x ^ p) * a + b
		if (!Eotf)
			x = _mm512_fmadd_ps(x, _mm512_set1_ps(T::power_scale), _mm512_set1_ps(T::power_offset));

		// Merge with the linear segment.
		x = _mm512_mask_mul_ps(x, mask, orig, _mm512_set1_ps(T::linear_scale));

		if (!Prescale)
			x = _mm512_mul_ps(x, scale);

		// copysign(x, orig)
		x = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(sign, _mm512_castps_si512(orig), _mm512_castps_si512(x), 0xCA));
		return x;
	}
};

template <class T, bool Log, bool Prescale>
struct SegmentedPolynomial {
	static inline FORCE_INLINE __m512 func(__m512 x, __m512 scale)
	{
		__m512 result;
		__m512i idx;

		if (Prescale)
			x = _mm512_mul_ps(x, scale);

		x = _mm512_max_ps(x, _mm512_set1_ps(FLT_MIN));

		if (Log) {
			// Classify the argument into one of 32 segments by its exponent.
			const __m512i exponent_min = _mm512_set1_epi32(127 - 32);
			const __m512i exponent_max = _mm512_set1_epi32(127 - 1);
			idx = _mm512_srli_epi32(_mm512_castps_si512(x), 23);
			idx = _mm512_max_epi32(idx, exponent_min);
			idx = _mm512_min_epi32(idx, exponent_max);
		} else {
			// Classify the argument into one of 32 uniform segments on [0, 1].
			const __m512 one_minus_eps = _mm512_set1_ps(0.999999940f);
			__m512 tmp = x;
			tmp = _mm512_max_ps(tmp, _mm512_setzero_ps());
			tmp = _mm512_min_ps(tmp, one_minus_eps);
			tmp = _mm512_mul_ps(tmp, _mm512_set1_ps(32.0f));
			idx = _mm512_cvttps_epi32(tmp);
		}

		// Apply the polynomial approximation for the segment.
		result = _mm512_permutex2var_ps(_mm512_load_ps(T::horner0), idx, _mm512_load_ps(T::horner0 + 16));
		result = _mm512_fmadd_ps(result, x, _mm512_permutex2var_ps(_mm512_load_ps(T::horner1), idx, _mm512_load_ps(T::horner1 + 16)));
		result = _mm512_fmadd_ps(result, x, _mm512_permutex2var_ps(_mm512_load_ps(T::horner2), idx, _mm512_load_ps(T::horner2 + 16)));
		result = _mm512_fmadd_ps(result, x, _mm512_permutex2var_ps(_mm512_load_ps(T::horner3), idx, _mm512_load_ps(T::horner3 + 16)));
		result = _mm512_fmadd_ps(result, x, _mm512_permutex2var_ps(_mm512_load_ps(T::horner4), idx, _mm512_load_ps(T::horner4 + 16)));

		if (!Log)
			result = _mm512_max_ps(result, _mm512_setzero_ps());

		if (!Prescale)
			result = _mm512_mul_ps(result, scale);

		return result;
	}
};

typedef PowerFunction<avx512constants::Rec1886EOTF, false> FuncRec1886EOTF;
typedef PowerFunction<avx512constants::Rec1886InverseEOTF, true> FuncRec1886InverseEOTF;
typedef SRGBPowerFunction<avx512constants::SRGBEOTF, true, false> FuncSRGBEOTF;
typedef SRGBPowerFunction<avx512constants::SRGBInverseEOTF, false, true> FuncSRGBInverseEOTF;
typedef SegmentedPolynomial<avx512constants::ST2084EOTF, false, false> FuncST2084EOTF;
typedef SegmentedPolynomial<avx512constants::ST2084InverseEOTF, true, true> FuncST2084InverseEOTF;

template <class Op>
void gamma_filter_line_avx512(const float *src, float *dst, float scale, unsigned left, unsigned right)
{
	unsigned vec_left = ceil_n(left, 16);
	unsigned vec_right = floor_n(right, 16);

	if (left != vec_left) {
		__m512 x = Op::func(_mm512_load_ps(src + vec_left - 16), _mm512_set1_ps(scale));
		__mmask16 mask = mmask16_set_hi(vec_left - left);
		_mm512_mask_store_ps(dst + vec_left - 16, mask, x);
	}

	for (unsigned j = vec_left; j < vec_right; j += 16) {
		__m512 x = Op::func(_mm512_load_ps(src + j), _mm512_set1_ps(scale));
		_mm512_store_ps(dst + j, x);
	}

	if (right != vec_right) {
		__m512 x = Op::func(_mm512_load_ps(src + vec_right), _mm512_set1_ps(scale));
		__mmask16 mask = mmask16_set_lo(right - vec_right);
		_mm512_mask_store_ps(dst + vec_right, mask, x);
	}
}


class MatrixOperationAVX512 final : public MatrixOperationImpl {
public:
	explicit MatrixOperationAVX512(const Matrix3x3 &m) :
		MatrixOperationImpl(m)
	{
	}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		matrix_filter_line_avx512(static_cast<const float *>(&m_matrix[0][0]), src, dst, left, right);
	}
};

template <class Op>
class GammaOperationAVX512 final : public Operation {
	float m_scale;
public:
	explicit GammaOperationAVX512(float scale) : m_scale{ scale } {}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		gamma_filter_line_avx512<Op>(src[0], dst[0], m_scale, left, right);
		gamma_filter_line_avx512<Op>(src[1], dst[1], m_scale, left, right);
		gamma_filter_line_avx512<Op>(src[2], dst[2], m_scale, left, right);
	}
};

} // namespace


std::unique_ptr<Operation> create_matrix_operation_avx512(const Matrix3x3 &m)
{
	return std::make_unique<MatrixOperationAVX512>(m);
}

std::unique_ptr<Operation> create_gamma_operation_avx512(const TransferFunction &transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	if (transfer.to_gamma == rec_1886_inverse_eotf)
		return std::make_unique<GammaOperationAVX512<FuncRec1886InverseEOTF>>(transfer.to_gamma_scale);
	else if (transfer.to_gamma == srgb_inverse_eotf)
		return std::make_unique<GammaOperationAVX512<FuncSRGBInverseEOTF>>(transfer.to_gamma_scale);
	else if (transfer.to_gamma == st_2084_inverse_eotf)
		return std::make_unique<GammaOperationAVX512<FuncST2084InverseEOTF>>(transfer.to_gamma_scale);

	return nullptr;
}

std::unique_ptr<Operation> create_inverse_gamma_operation_avx512(const TransferFunction &transfer, const OperationParams &params)
{
	if (!params.approximate_gamma)
		return nullptr;

	if (transfer.to_linear == rec_1886_eotf)
		return std::make_unique<GammaOperationAVX512<FuncRec1886EOTF>>(transfer.to_linear_scale);
	else if (transfer.to_linear == srgb_eotf)
		return std::make_unique<GammaOperationAVX512<FuncSRGBEOTF>>(transfer.to_linear_scale);
	else if (transfer.to_linear == st_2084_eotf)
		return std::make_unique<GammaOperationAVX512<FuncST2084EOTF>>(transfer.to_linear_scale);

	return nullptr;
}

} // namespace zimg::colorspace

#endif // ZIMG_X86_AVX512
