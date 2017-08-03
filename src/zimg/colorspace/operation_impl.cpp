#include <algorithm>
#include <cfloat>
#include "common/make_unique.h"
#include "common/zassert.h"
#include "colorspace.h"
#include "colorspace_param.h"
#include "gamma.h"
#include "matrix3.h"
#include "operation.h"
#include "operation_impl.h"

#ifdef ZIMG_X86
  #include "x86/operation_impl_x86.h"
#endif

namespace zimg {
namespace colorspace {

namespace {

class MatrixOperationC final : public MatrixOperationImpl {
public:
	explicit MatrixOperationC(const Matrix3x3 &m) : MatrixOperationImpl(m) {}

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

class GammaOperationC final : public Operation {
	gamma_func m_func;
	float m_prescale;
	float m_postscale;
public:
	GammaOperationC(gamma_func func, float prescale, float postscale) :
		m_func{ func },
		m_prescale{ prescale },
		m_postscale{ postscale }
	{}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		EnsureSinglePrecision x87;

		for (unsigned p = 0; p < 3; ++p) {
			const float *src_p = src[p];
			float *dst_p = dst[p];

			for (unsigned i = left; i < right; ++i) {
				dst_p[i] = m_postscale * m_func(src_p[i] * m_prescale);
			}
		}
	}
};

class AribB67OperationC final : public Operation {
	float m_kr;
	float m_kg;
	float m_kb;
	float m_scale;
public:
	AribB67OperationC(double kr, double kg, double kb, float scale) :
		m_kr{ static_cast<float>(kr) },
		m_kg{ static_cast<float>(kg) },
		m_kb{ static_cast<float>(kb) },
		m_scale{ scale }
	{}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		const float gamma = 1.2f;

		for (unsigned i = left; i < right; ++i) {
			float r = src[0][i] * m_scale;
			float g = src[1][i] * m_scale;
			float b = src[2][i] * m_scale;

			float yd = std::max(m_kr * r + m_kg * g + m_kb * b, FLT_MIN);
			float ys_inv = zimg_x_powf(yd, (1.0f - gamma) / gamma);

			r = arib_b67_oetf(r * ys_inv);
			g = arib_b67_oetf(g * ys_inv);
			b = arib_b67_oetf(b * ys_inv);

			dst[0][i] = r;
			dst[1][i] = g;
			dst[2][i] = b;
		}
	}
};

class AribB67InverseOperationC final : public Operation {
	float m_kr;
	float m_kg;
	float m_kb;
	float m_scale;
public:
	AribB67InverseOperationC(double kr, double kg, double kb, float scale) :
		m_kr{ static_cast<float>(kr) },
		m_kg{ static_cast<float>(kg) },
		m_kb{ static_cast<float>(kb) },
		m_scale{ scale }
	{}

	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		const float gamma = 1.2f;

		for (unsigned i = left; i < right; ++i) {
			float r = src[0][i];
			float g = src[1][i];
			float b = src[2][i];

			float ys = std::max(m_kr * r + m_kg * g + m_kb * b, FLT_MIN);
			ys = zimg_x_powf(ys, gamma - 1.0f);

			r = arib_b67_inverse_oetf(r * ys);
			g = arib_b67_inverse_oetf(g * ys);
			b = arib_b67_inverse_oetf(b * ys);

			dst[0][i] = r * m_scale;
			dst[1][i] = g * m_scale;
			dst[2][i] = b * m_scale;
		}
	}
};

class Rec2020CLToRGBOperationC final : public Operation {
public:
	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		constexpr float kr = static_cast<float>(REC_2020_KR);
		constexpr float kb = static_cast<float>(REC_2020_KB);
		constexpr float kg = 1.0f - kr - kb;

		constexpr float pb = 0.7909854f;
		constexpr float nb = -0.9701716f;
		constexpr float pr = 0.4969147f;
		constexpr float nr = -0.8591209f;

		EnsureSinglePrecision x87;

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

			b = rec_709_inverse_oetf(b_minus_y + y);
			r = rec_709_inverse_oetf(r_minus_y + y);

			y = rec_709_inverse_oetf(y);
			g = (y - kr * r - kb * b) / kg;

			dst[0][i] = r;
			dst[1][i] = g;
			dst[2][i] = b;
		}
	}
};

class Rec2020CLToYUVOperationC final : public Operation {
public:
	void process(const float * const *src, float * const *dst, unsigned left, unsigned right) const override
	{
		constexpr float kr = static_cast<float>(REC_2020_KR);
		constexpr float kb = static_cast<float>(REC_2020_KB);
		constexpr float kg = 1.0f - kr - kb;

		constexpr float pb = 0.7909854f;
		constexpr float nb = -0.9701716f;
		constexpr float pr = 0.4969147f;
		constexpr float nr = -0.8591209f;

		EnsureSinglePrecision x87;

		for (unsigned i = left; i < right; ++i) {
			float r = src[0][i];
			float g = src[1][i];
			float b = src[2][i];

			float y = rec_709_oetf(kr * r + kg * g + kb * b);
			float u, v;

			b = rec_709_oetf(b);
			r = rec_709_oetf(r);

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

std::unique_ptr<Operation> create_gamma_operation(TransferCharacteristics transfer, const OperationParams &params)
{
	TransferFunction func = select_transfer_function(transfer, params.peak_luminance, params.scene_referred);
	return ztd::make_unique<GammaOperationC>(func.to_gamma, func.to_gamma_scale, 1.0f);
}

std::unique_ptr<Operation> create_inverse_gamma_operation(TransferCharacteristics transfer, const OperationParams &params)
{
	TransferFunction func = select_transfer_function(transfer, params.peak_luminance, params.scene_referred);
	return ztd::make_unique<GammaOperationC>(func.to_linear, 1.0f, func.to_linear_scale);
}

std::unique_ptr<Operation> create_arib_b67_operation(const Matrix3x3 &m, const OperationParams &params)
{
	zassert_d(!params.scene_referred, "must be display-referred");

	TransferFunction func = select_transfer_function(TransferCharacteristics::ARIB_B67, params.peak_luminance, false);
	return ztd::make_unique<AribB67OperationC>(m[0][0], m[0][1], m[0][2], func.to_gamma_scale);
}

std::unique_ptr<Operation> create_inverse_arib_b67_operation(const Matrix3x3 &m, const OperationParams &params)
{
	zassert_d(!params.scene_referred, "must be display-referred");

	TransferFunction func = select_transfer_function(TransferCharacteristics::ARIB_B67, params.peak_luminance, false);
	return ztd::make_unique<AribB67InverseOperationC>(m[0][0], m[0][1], m[0][2], func.to_linear_scale);
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
