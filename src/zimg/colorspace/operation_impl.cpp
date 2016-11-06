#include <algorithm>
#include <cfloat>
#include <cmath>

#include "common/make_unique.h"
#include "common/zassert.h"
#include "colorspace_param.h"
#include "matrix3.h"
#include "operation.h"
#include "operation_impl.h"
#include "operation_impl_x86.h"

// MSVC 32-bit compiler generates x87 instructions when operating on floats
// returned from external functions. Force single precision to avoid errors.
#if defined(_MSC_VER) && defined(_M_IX86)
  #define fpu_save() _control87(0, 0)
  #define fpu_set_single() _control87(_PC_24, _MCW_PC)
  #define fpu_restore(x) _control87((x), _MCW_PC)
#else
  #define fpu_save() 0
  #define fpu_set_single() (void)0
  #define fpu_restore(x) (void)x
#endif /* _MSC_VER */

namespace zimg {
namespace colorspace {

namespace {

class MatrixOperationC : public MatrixOperationImpl {
public:
	explicit MatrixOperationC(const Matrix3x3 &m) :
		MatrixOperationImpl(m)
	{
	}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		for (unsigned i = left; i < right; ++i) {
			float a, b, c;
			float x, y, z;

			a = src[0][i];
			b = src[1][i];
			c = src[2][i];

			x = m_matrix[0][0] * a + m_matrix[0][1] * b + m_matrix[0][2] * c;
			y = m_matrix[1][0] * a + m_matrix[1][1] * b + m_matrix[1][2] * c;
			z = m_matrix[2][0] * a + m_matrix[2][1] * b + m_matrix[2][2] * c;

			dst[0][i] = x;
			dst[1][i] = y;
			dst[2][i] = z;
		}
	}
};

class Rec709GammaOperationC : public Operation {
public:
	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		for (unsigned p = 0; p < 3; ++p) {
			for (unsigned i = left; i < right; ++i) {
				float x = src[p][i];

				dst[p][i] = rec_709_gamma(x);
			}
		}
	}
};

class Rec709InverseGammaOperationC : public Operation {
public:
	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		for (unsigned p = 0; p < 3; ++p) {
			for (unsigned i = left; i < right; ++i) {
				float x = src[p][i];

				dst[p][i] = rec_709_inverse_gamma(x);
			}
		}
	}
};

class SRGBGammaOperationC : public Operation {
public:
	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		for (unsigned p = 0; p < 3; ++p) {
			for (unsigned i = left; i < right; ++i) {
				float x = src[p][i];

				dst[p][i] = srgb_gamma(x);
			}
		}
	}
};

class SRGBInverseGammaOperationC : public Operation {
public:
	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		for (unsigned p = 0; p < 3; ++p) {
			for (unsigned i = left; i < right; ++i) {
				float x = src[p][i];

				dst[p][i] = srgb_inverse_gamma(x);
			}
		}
	}
};

class St2084GammaOperationC : public Operation {
	float m_scale;
public:
	explicit St2084GammaOperationC(double peak_luminance) :
		m_scale{ static_cast<float>(peak_luminance / ST2084_PEAK_LUMINANCE) }
	{
	}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		for (unsigned p = 0; p < 3; ++p) {
			for (unsigned i = left; i < right; ++i) {
					float x = src[p][i];

					dst[p][i] = st_2084_gamma(m_scale * x);
			}
		}
	}
};

class St2084InverseGammaOperationC : public Operation {
	float m_scale;
public:
	explicit St2084InverseGammaOperationC(double peak_luminance) :
		m_scale{ static_cast<float>(ST2084_PEAK_LUMINANCE / peak_luminance) }
	{
	}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		for (unsigned p = 0; p < 3; ++p) {
			for (unsigned i = left; i < right; ++i) {
				float x = src[p][i];

				dst[p][i] = m_scale * st_2084_inverse_gamma(x);
			}
		}
	}
};

class B67GammaOperationC : public Operation {
public:
	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		for (unsigned p = 0; p < 3; ++p) {
			for (unsigned i = left; i < right; ++i) {
				float x = src[p][i];

				dst[p][i] = arib_b67_gamma(x * (1.0f / 12.0f));
			}
		}
	}
};

class B67InverseGammaOperationC : public Operation {
public:
	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		for (unsigned p = 0; p < 3; ++p) {
			for (unsigned i = left; i < right; ++i) {
				float x = src[p][i];

				dst[p][i] = 12.0f * arib_b67_inverse_gamma(x);
			}
		}
	}
};

class Rec2020CLToRGBOperationC : public Operation {
public:
	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		const float kr = static_cast<float>(REC_2020_KR);
		const float kb = static_cast<float>(REC_2020_KB);
		const float kg = 1.0f - kr - kb;

		const float pb = 0.7909854f;
		const float nb = -0.9701716f;
		const float pr = 0.4969147f;
		const float nr = -0.8591209f;

		unsigned w = fpu_save();
		fpu_set_single();

		for (unsigned i = left; i < right; ++i) {
			float y = src[0][i];
			float u = src[1][i];
			float v = src[2][i];

			float r, g, b;
			float b_minus_y, r_minus_y;

			if (u < 0)
				b_minus_y = u * 2.0f * -nb;
			else
				b_minus_y = u * 2.0f * pb;

			if (v < 0)
				r_minus_y = v * 2.0f * -nr;
			else
				r_minus_y = v * 2.0f * pr;

			b = rec_709_inverse_gamma(b_minus_y + y);
			r = rec_709_inverse_gamma(r_minus_y + y);

			y = rec_709_inverse_gamma(y);
			g = (y - kr * r - kb * b) / kg;

			dst[0][i] = r;
			dst[1][i] = g;
			dst[2][i] = b;
		}

		fpu_restore(w);
	}
};

class Rec2020CLToYUVOperationC : public Operation {
public:
	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		const float kr = static_cast<float>(REC_2020_KR);
		const float kb = static_cast<float>(REC_2020_KB);
		const float kg = 1.0f - kr - kb;

		const float pb = 0.7909854f;
		const float nb = -0.9701716f;
		const float pr = 0.4969147f;
		const float nr = -0.8591209f;

		unsigned w = fpu_save();
		fpu_set_single();

		for (unsigned i = left; i < right; ++i) {
			float r = src[0][i];
			float g = src[1][i];
			float b = src[2][i];

			float y = rec_709_gamma(kr * r + kg * g + kb * b);
			float u, v;

			b = rec_709_gamma(b);
			r = rec_709_gamma(r);

			if (b - y < 0.0f)
				u = (b - y) / (2.0f * -nb);
			else
				u = (b - y) / (2.0f * pb);

			if (r - y < 0.0f)
				v = (r - y) / (2.0f * -nr);
			else
				v = (r - y) / (2.0f * pr);

			dst[0][i] = y;
			dst[1][i] = u;
			dst[2][i] = v;
		}

		fpu_restore(w);
	}
};

} // namespace


float rec_709_gamma(float x)
{
	unsigned w = fpu_save();
	fpu_set_single();

	if (x < REC709_BETA)
		x = x * 4.5f;
	else
		x = REC709_ALPHA * zimg_x_powf(x, 0.45f) - (REC709_ALPHA - 1.0f);

	fpu_restore(w);
	return x;
}

float rec_709_inverse_gamma(float x)
{
	unsigned w = fpu_save();
	fpu_set_single();

	if (x < 4.5f * REC709_BETA)
		x = x / 4.5f;
	else
		x = zimg_x_powf((x + (REC709_ALPHA - 1.0f)) / REC709_ALPHA, 1.0f / 0.45f);

	fpu_restore(w);
	return x;
}

float srgb_gamma(float x)
{
	unsigned w = fpu_save();
	fpu_set_single();

	if (x < SRGB_BETA)
		x = x * 12.92f;
	else
		x = SRGB_ALPHA * zimg_x_powf(x, 1.0f / 2.4f) - (SRGB_ALPHA - 1.0f);

	fpu_restore(w);
	return x;
}

float srgb_inverse_gamma(float x)
{
	unsigned w = fpu_save();
	fpu_set_single();

	if (x < 12.92f * SRGB_BETA)
		x = x / 12.92f;
	else
		x = zimg_x_powf((x + (SRGB_ALPHA - 1.0f)) / SRGB_ALPHA, 2.4f);

	fpu_restore(w);
	return x;
}


float st_2084_gamma(float x)
{
	unsigned w = fpu_save();
	fpu_set_single();

	// Filter negative values to avoid NAN, and also special-case 0 so that (f(g(0)) == 0).
	if (x > 0.0f) {
		float xpow = zimg_x_powf(x, ST2084_M1);
#if 0
		// Original formulation from SMPTE ST 2084:2014 publication.
		float num = ST2084_C1 + ST2084_C2 * xpow;
		float den = 1.0f + ST2084_C3 * xpow;
		x = zimg_x_powf(num / den, ST2084_M2);
#else
		// More stable arrangement that avoids some cancellation error.
		float num = (ST2084_C1 - 1.0f) + (ST2084_C2 - ST2084_C3) * xpow;
		float den = 1.0f + ST2084_C3 * xpow;
		x = zimg_x_powf(1.0f + num / den, ST2084_M2);
#endif
	} else {
		x = 0.0f;
	}

	fpu_restore(w);
	return x;
}

float st_2084_inverse_gamma(float x)
{
	unsigned w = fpu_save();
	fpu_set_single();

	// Filter negative values to avoid NAN.
	if (x > 0.0f) {
		float xpow = zimg_x_powf(x, 1.0f / ST2084_M2);
		float num = std::max(xpow - ST2084_C1, 0.0f);
		float den = std::max(ST2084_C2 - ST2084_C3 * xpow, FLT_MIN);
		x = zimg_x_powf(num / den, 1.0f / ST2084_M1);
	} else {
		x = 0.0f;
	}

	fpu_restore(w);
	return x;
}

float arib_b67_gamma(float x)
{
	unsigned w = fpu_save();
	fpu_set_single();

	// Prevent negative pixels from yielding NAN.
	x = std::max(x, 0.0f);

	if (x <= (1.0f / 12.0f))
		x = zimg_x_sqrtf(3.0f * x);
	else
		x = ARIB_B67_A * zimg_x_logf(12.0f * x - ARIB_B67_B) + ARIB_B67_C;

	fpu_restore(w);
	return x;
}

float arib_b67_inverse_gamma(float x)
{
	unsigned w = fpu_save();
	fpu_set_single();

	// Prevent negative pixels expanding into positive values.
	x = std::max(x, 0.0f);

	if (x <= 0.5f)
		x = (x * x) * (1.0f / 3.0f);
	else
		x = (zimg_x_expf((x - ARIB_B67_C) / ARIB_B67_A) + ARIB_B67_B) / 12.0f;

	fpu_restore(w);
	return x;
}


MatrixOperationImpl::MatrixOperationImpl(const Matrix3x3 &m)
{
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			m_matrix[i][j] = static_cast<float>(m[i][j]);
		}
	}
}


std::unique_ptr<Operation> create_matrix_operation(const Matrix3x3 &m, CPUClass cpu)
{
	std::unique_ptr<Operation> ret;

#ifdef ZIMG_X86
	ret = create_matrix_operation_x86(m, cpu);
#endif
	if (!ret)
		ret = ztd::make_unique<MatrixOperationC>(m);

	return ret;
}

std::unique_ptr<Operation> create_rec709_gamma_operation(CPUClass cpu)
{
	return ztd::make_unique<Rec709GammaOperationC>();
}

std::unique_ptr<Operation> create_rec709_inverse_gamma_operation(CPUClass cpu)
{
	return ztd::make_unique<Rec709InverseGammaOperationC>();
}

std::unique_ptr<Operation> create_srgb_gamma_operation(CPUClass cpu)
{
	return ztd::make_unique<SRGBGammaOperationC>();
}

std::unique_ptr<Operation> create_srgb_inverse_gamma_operation(CPUClass cpu)
{
	return ztd::make_unique<SRGBInverseGammaOperationC>();
}

std::unique_ptr<Operation> create_st2084_gamma_operation(double peak_luminance, CPUClass cpu)
{
	zassert_d(!std::isnan(peak_luminance), "nan detected");
	return ztd::make_unique<St2084GammaOperationC>(peak_luminance);
}

std::unique_ptr<Operation> create_st2084_inverse_gamma_operation(double peak_luminance, CPUClass cpu)
{
	zassert_d(!std::isnan(peak_luminance), "nan detected");
	return ztd::make_unique<St2084InverseGammaOperationC>(peak_luminance);
}

std::unique_ptr<Operation> create_b67_gamma_operation(CPUClass cpu)
{
	return ztd::make_unique<B67GammaOperationC>();
}

std::unique_ptr<Operation> create_b67_inverse_gamma_operation(CPUClass cpu)
{
	return ztd::make_unique<B67InverseGammaOperationC>();
}

std::unique_ptr<Operation> create_2020_cl_yuv_to_rgb_operation(const OperationParams &params, CPUClass cpu)
{
	return ztd::make_unique<Rec2020CLToRGBOperationC>();
}

std::unique_ptr<Operation> create_2020_cl_rgb_to_yuv_operation(const OperationParams &params, CPUClass cpu)
{
	return ztd::make_unique<Rec2020CLToYUVOperationC>();
}

} // namespace colorspace
} // namespace zimg
