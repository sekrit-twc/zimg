#include <algorithm>
#include <cmath>
#include <cstdint>
#include "Common/cpuinfo.h"
#include "Common/except.h"
#include "colorspace_param.h"
#include "matrix3.h"
#include "operation.h"
#include "operation_impl.h"

namespace zimg {;
namespace colorspace {;

namespace {;

const float TRANSFER_ALPHA = 1.09929682680944f;
const float TRANSFER_BETA = 0.018053968510807f;

float rec_709_gamma(float x)
{
	if (x < TRANSFER_BETA)
		x = x * 4.5f;
	else
		x = TRANSFER_ALPHA * std::pow(x, 0.45f) - (TRANSFER_ALPHA - 1.0f);

	return x;
}

float rec_709_inverse_gamma(float x)
{
	if (x < 4.5f * TRANSFER_BETA)
		x = x / 4.5f;
	else
		x = std::pow((x + (TRANSFER_ALPHA - 1.0f)) / TRANSFER_ALPHA, 1.0f / 0.45f);

	return x;
}

class PixelAdapterC : public PixelAdapter {
	void f16_to_f32(const uint16_t *src, float *dst, int width) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in C impl" };
	}

	void f16_from_f32(const float *src, uint16_t *dst, int width) const override
	{
		throw ZimgUnsupportedError{ "f16 not supported in C impl" };
	}
};

class MatrixOperationC : public Operation {
	float m_matrix[3][3];
public:
	explicit MatrixOperationC(const Matrix3x3 &matrix)
	{
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				m_matrix[i][j] = (float)matrix[i][j];
			}
		}
	}

	void process(float * const *ptr, int width) const override
	{
		for (int i = 0; i < width; ++i) {
			float a, b, c;
			float x, y, z;

			a = ptr[0][i];
			b = ptr[1][i];
			c = ptr[2][i];

			x = m_matrix[0][0] * a + m_matrix[0][1] * b + m_matrix[0][2] * c;
			y = m_matrix[1][0] * a + m_matrix[1][1] * b + m_matrix[1][2] * c;
			z = m_matrix[2][0] * a + m_matrix[2][1] * b + m_matrix[2][2] * c;

			ptr[0][i] = x;
			ptr[1][i] = y;
			ptr[2][i] = z;
		}
	}
};

class Rec709GammaOperationC : public Operation {
public:
	void process(float * const *ptr, int width) const override
	{
		for (int p = 0; p < 3; ++p) {
			for (int i = 0; i < width; ++i) {
				float x = ptr[p][i];

				ptr[p][i] = rec_709_gamma(x);
			}
		}
	}
};

class Rec709InverseGammaOperationC : public Operation {
	void process(float * const *ptr, int width) const override
	{
		for (int p = 0; p < 3; ++p) {
			for (int i = 0; i < width; ++i) {
				float x = ptr[p][i];

				ptr[p][i] = rec_709_inverse_gamma(x);
			}
		}
	}
};

class Rec2020CLToRGBOperationC : public Operation {
	void process(float * const *ptr, int width) const override
	{
		float kr = (float)REC_2020_KR;
		float kb = (float)REC_2020_KB;
		float kg = 1.0f - kr - kb;

		float pb = 0.7909854f;
		float nb = -0.9701716f;
		float pr = 0.4969147f;
		float nr = -0.8591209f;

		for (int i = 0; i < width; ++i) {
			float y = ptr[0][i];
			float u = ptr[1][i];
			float v = ptr[2][i];

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

			ptr[0][i] = r;
			ptr[1][i] = g;
			ptr[2][i] = b;
		}
	}
};

class Rec2020CLToYUVOperationC : public Operation {
	void process(float * const *ptr, int width) const override
	{
		float kr = (float)REC_2020_KR;
		float kb = (float)REC_2020_KB;
		float kg = 1.0f - kr - kb;

		float pb = 0.7909854f;
		float nb = -0.9701716f;
		float pr = 0.4969147f;
		float nr = -0.8591209f;

		for (int i = 0; i < width; ++i) {
			float r = ptr[0][i];
			float g = ptr[1][i];
			float b = ptr[2][i];

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

			ptr[0][i] = y;
			ptr[1][i] = u;
			ptr[2][i] = v;
		}
	}
};

} // namespace


PixelAdapter *create_pixel_adapter(CPUClass cpu)
{
	return new PixelAdapterC{};
}

Operation *create_matrix_operation(const Matrix3x3 &m, CPUClass cpu)
{
	return new MatrixOperationC{ m };
}

Operation *create_rec709_gamma_operation(CPUClass cpu)
{
	return new Rec709GammaOperationC{};
}

Operation *create_rec709_inverse_gamma_operation(CPUClass cpu)
{
	return new Rec709InverseGammaOperationC{};
}

Operation *create_2020_cl_yuv_to_rgb_operation(CPUClass cpu)
{
	return new Rec2020CLToRGBOperationC{};
}

Operation *create_2020_cl_rgb_to_yuv_operation(CPUClass cpu)
{
	return new Rec2020CLToYUVOperationC{};
}

} // namespace colorspace
} // namespace zimg
