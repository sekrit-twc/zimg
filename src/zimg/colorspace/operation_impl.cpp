#include <algorithm>
#if defined(_MSC_VER) && defined(_M_IX86)
  #include <cfloat>
#endif

#include "common/make_unique.h"
#include "colorspace_param.h"
#include "matrix3.h"
#include "operation.h"
#include "operation_impl.h"
#include "operation_impl_x86.h"

namespace zimg {
namespace colorspace {

namespace {

class MatrixOperationC : public MatrixOperationImpl {
public:
	explicit MatrixOperationC(const Matrix3x3 &m) :
		MatrixOperationImpl(m)
	{
	}

	void process(const float * const *src, float * const * dst, unsigned left, unsigned right) const override
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
	void process(const float * const *src, float * const * dst, unsigned left, unsigned right) const override
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
	void process(const float * const *src, float * const * dst, unsigned left, unsigned right) const override
	{
		for (unsigned p = 0; p < 3; ++p) {
			for (unsigned i = left; i < right; ++i) {
				float x = src[p][i];

				dst[p][i] = rec_709_inverse_gamma(x);
			}
		}
	}
};

class Smpte2084GammaOperationC : public Operation {
public:
	void process(const float * const *src, float * const * dst, unsigned left, unsigned right) const override
	{
		for (unsigned p = 0; p < 3; ++p) {
			for (unsigned i = left; i < right; ++i) {
				float x = src[p][i];

				dst[p][i] = smpte_2084_transfer(x);
			}
		}
	}
};

class Smpte2084InverseGammaOperationC : public Operation {
public:
	void process(const float * const *src, float * const * dst, unsigned left, unsigned right) const override
	{
		for (unsigned p = 0; p < 3; ++p) {
			for (unsigned i = left; i < right; ++i) {
				float x = src[p][i];

				dst[p][i] = smpte_2084_inverse_transfer(x);
			}
		}
	}
};

class Rec2020CLToRGBOperationC : public Operation {
public:
	void process(const float * const *src, float * const * dst, unsigned left, unsigned right) const override
	{
		float kr = (float)REC_2020_KR;
		float kb = (float)REC_2020_KB;
		float kg = 1.0f - kr - kb;

		float pb = 0.7909854f;
		float nb = -0.9701716f;
		float pr = 0.4969147f;
		float nr = -0.8591209f;

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
	}
};

class Rec2020CLToYUVOperationC : public Operation {
public:
	void process(const float * const *src, float * const * dst, unsigned left, unsigned right) const override
	{
		float kr = (float)REC_2020_KR;
		float kb = (float)REC_2020_KB;
		float kg = 1.0f - kr - kb;

		float pb = 0.7909854f;
		float nb = -0.9701716f;
		float pr = 0.4969147f;
		float nr = -0.8591209f;

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
	}
};

} // namespace


float rec_709_gamma(float x)
{
	// MSVC 32-bit compiler generates x87 instructions when operating on floats
	// returned from external functions. Force single precision to avoid errors.
#if defined(_MSC_VER) && defined(_M_IX86)
	unsigned w = _control87(0, 0);
	_control87(_PC_24, _MCW_PC);
#endif
	if (x < TRANSFER_BETA)
		x = x * 4.5f;
	else
		x = TRANSFER_ALPHA * _zimg_powf(x, 0.45f) - (TRANSFER_ALPHA - 1.0f);
#if defined(_MSC_VER) && defined(_M_IX86)
	_control87(w, _MCW_PC);
#endif
	return x;
}

float rec_709_inverse_gamma(float x)
{
	// MSVC 32-bit compiler generates x87 instructions when operating on floats
	// returned from external functions. Force single precision to avoid errors.
#if defined(_MSC_VER) && defined(_M_IX86)
	unsigned w = _control87(0, 0);
	_control87(_PC_24, _MCW_PC);
#endif
	if (x < 4.5f * TRANSFER_BETA)
		x = x / 4.5f;
	else
		x = _zimg_powf((x + (TRANSFER_ALPHA - 1.0f)) / TRANSFER_ALPHA, 1.0f / 0.45f);
#if defined(_MSC_VER) && defined(_M_IX86)
	_control87(w, _MCW_PC);
#endif
	return x;
}

float smpte_2084_transfer(float x)
{
#if defined(_MSC_VER) && defined(_M_IX86)
	unsigned w = _control87(0, 0);
	_control87(_PC_24, _MCW_PC);
#endif
	x = _zimg_powf(((SMPTE_2084_C1 + SMPTE_2084_C2 * _zimg_powf(x, SMPTE_2084_N)) / (1.0f + SMPTE_2084_C3 * _zimg_powf(x, SMPTE_2084_N))), SMPTE_2084_M);
#if defined(_MSC_VER) && defined(_M_IX86)
	_control87(w, _MCW_PC);
#endif
	return x;
}

float smpte_2084_inverse_transfer(float x)
{
#if defined(_MSC_VER) && defined(_M_IX86)
	unsigned w = _control87(0, 0);
	_control87(_PC_24, _MCW_PC);
#endif
	x = _zimg_powf((std::max((_zimg_powf(x, 1.0f / SMPTE_2084_M) - SMPTE_2084_C1 ), 0.0f))/(SMPTE_2084_C2 - SMPTE_2084_C3 * _zimg_powf(x, 1.0f / SMPTE_2084_M)), 1.0f / SMPTE_2084_N);
#if defined(_MSC_VER) && defined(_M_IX86)
	_control87(w, _MCW_PC);
#endif
	return x;
}


MatrixOperationImpl::MatrixOperationImpl(const Matrix3x3 &m)
{
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			m_matrix[i][j] = (float)m[i][j];
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

std::unique_ptr<Operation> create_smpte2084_gamma_operation(CPUClass cpu)
{
	return ztd::make_unique<Smpte2084GammaOperationC>();
}

std::unique_ptr<Operation> create_smpte2084_inverse_gamma_operation(CPUClass cpu)
{
	return ztd::make_unique<Smpte2084InverseGammaOperationC>();
}

std::unique_ptr<Operation> create_2020_cl_yuv_to_rgb_operation(CPUClass cpu)
{
	return ztd::make_unique<Rec2020CLToRGBOperationC>();
}

std::unique_ptr<Operation> create_2020_cl_rgb_to_yuv_operation(CPUClass cpu)
{
	return ztd::make_unique<Rec2020CLToYUVOperationC>();
}

} // namespace colorspace
} // namespace zimg
