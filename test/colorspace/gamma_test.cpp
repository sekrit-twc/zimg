#include <cmath>

#include "colorspace/gamma.h"
#include "gtest/gtest.h"

namespace {

void test_monotonic(float (*func)(float), float min, float max, unsigned long steps)
{
	zimg::colorspace::EnsureSinglePrecision x87;

	float cur = -INFINITY;

	for (unsigned long i = 0; i <= steps; ++i) {
		float x = min + i * ((max - min) / steps);
		float y = func(x);
		ASSERT_FALSE(std::isnan(y)) << " x=" << x << " i=" << i;
		ASSERT_GE(y, cur) << " x=" << x << " i=" << i;
		cur = y;
	}
}

void test_accuracy(float (*f)(float), float (*g)(float), float min, float max, float errthr, float biasthr)
{
	zimg::colorspace::EnsureSinglePrecision x87;

	const unsigned long STEPS = 1UL << 16;
	float err = 0.0f;
	float bias = 0.0f;

	for (unsigned long i = 0; i <= STEPS; ++i) {
		float x = min + i * ((max - min) / STEPS);
		float y = f(x);
		float x_roundtrip = g(y);

		err += std::fabs((x_roundtrip - x) / (x == 0.0f ? FLT_EPSILON : x));
		bias += x_roundtrip - x;
	}

	err /= (STEPS + 1);
	bias /= (STEPS + 1);

	EXPECT_LT(err, errthr);
	EXPECT_LT(std::fabs(bias), biasthr);
}

} // namespace


TEST(GammaTest, test_rec709)
{
	EXPECT_EQ(0.0f, zimg::colorspace::rec_709_oetf(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::rec_709_oetf(1.0f));
	EXPECT_EQ(0.0f, zimg::colorspace::rec_709_inverse_oetf(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::rec_709_inverse_oetf(1.0f));

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::rec_709_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::rec_709_inverse_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::rec_709_oetf, zimg::colorspace::rec_709_inverse_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::rec_709_inverse_oetf, zimg::colorspace::rec_709_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::rec_709_oetf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::rec_709_inverse_oetf, 1.0f, 2.0f, 1UL << 16);

	SCOPED_TRACE("btb");
	test_monotonic(zimg::colorspace::rec_709_oetf, -1.0f, 0.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::rec_709_inverse_oetf, -1.0f, 0.0f, 1UL << 16);
}

TEST(GammaTest, test_log100)
{
	EXPECT_EQ(0.0f, zimg::colorspace::log100_oetf(0.01f));
	EXPECT_GE(zimg::colorspace::log100_oetf(std::nextafter(0.01f, INFINITY)), 0.0f);
	EXPECT_EQ(1.0f, zimg::colorspace::log100_oetf(1.0f));
	EXPECT_EQ(0.01f, zimg::colorspace::log100_inverse_oetf(0.0f));
	EXPECT_GE(zimg::colorspace::log100_inverse_oetf(std::nextafter(0.0f, INFINITY)), 0.01f);
	EXPECT_EQ(1.0f, zimg::colorspace::log100_inverse_oetf(1.0f));

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::log100_inverse_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::log100_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::log100_oetf, zimg::colorspace::log100_inverse_oetf, 0.01f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::log100_inverse_oetf, zimg::colorspace::log100_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::log100_inverse_oetf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::log100_oetf, 1.0f, 2.0f, 1UL << 16);
}

TEST(GammaTest, test_log316)
{
	EXPECT_EQ(0.0f, zimg::colorspace::log316_oetf(0.00316227766f));
	EXPECT_GE(zimg::colorspace::log316_oetf(std::nextafter(0.00316227766f, INFINITY)), 0.0f);
	EXPECT_EQ(1.0f, zimg::colorspace::log316_oetf(1.0f));
	EXPECT_EQ(0.00316227766f, zimg::colorspace::log316_inverse_oetf(0.0f));
	EXPECT_GE(zimg::colorspace::log316_inverse_oetf(std::nextafterf(0.0f, INFINITY)), 0.00316227766f);
	EXPECT_EQ(1.0f, zimg::colorspace::log316_inverse_oetf(1.0f));

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::log316_inverse_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::log316_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::log316_oetf, zimg::colorspace::log316_inverse_oetf, 0.0316227766f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::log316_inverse_oetf, zimg::colorspace::log316_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::log316_inverse_oetf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::log316_oetf, 1.0f, 2.0f, 1UL << 16);
}

TEST(GammaTest, test_rec_470m_470bg_st428)
{
	SCOPED_TRACE("470m");
	test_accuracy(zimg::colorspace::rec_470m_oetf, zimg::colorspace::rec_470m_inverse_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	test_accuracy(zimg::colorspace::rec_470m_inverse_oetf, zimg::colorspace::rec_470m_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("470bg");
	test_accuracy(zimg::colorspace::rec_470bg_oetf, zimg::colorspace::rec_470bg_inverse_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	test_accuracy(zimg::colorspace::rec_470bg_inverse_oetf, zimg::colorspace::rec_470bg_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("st428");
	test_accuracy(zimg::colorspace::st_428_eotf, zimg::colorspace::st_428_inverse_eotf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	test_accuracy(zimg::colorspace::st_428_inverse_eotf, zimg::colorspace::st_428_eotf, 0.0f, 1.0f, 1e-6f, 1e-6f);
}

TEST(GammaTest, test_smpte_240m)
{
	EXPECT_EQ(0.0f, zimg::colorspace::smpte_240m_oetf(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::smpte_240m_oetf(1.0f));
	EXPECT_EQ(0.0f, zimg::colorspace::smpte_240m_inverse_oetf(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::smpte_240m_inverse_oetf(1.0f));

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::smpte_240m_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::smpte_240m_inverse_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::smpte_240m_oetf, zimg::colorspace::smpte_240m_inverse_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::smpte_240m_inverse_oetf, zimg::colorspace::smpte_240m_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::smpte_240m_oetf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::smpte_240m_inverse_oetf, 1.0f, 2.0f, 1UL << 16);

	SCOPED_TRACE("btb");
	test_monotonic(zimg::colorspace::smpte_240m_oetf, -1.0f, 0.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::smpte_240m_inverse_oetf, -1.0f, 0.0f, 1UL << 16);
}

TEST(GammaTest, test_rec1886)
{
	EXPECT_EQ(0.0f, zimg::colorspace::rec_1886_inverse_eotf(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::rec_1886_inverse_eotf(1.0f));
	EXPECT_EQ(0.0f, zimg::colorspace::rec_1886_eotf(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::rec_1886_eotf(1.0f));

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::rec_1886_inverse_eotf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::rec_1886_eotf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::rec_1886_inverse_eotf, zimg::colorspace::rec_1886_eotf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::rec_1886_eotf, zimg::colorspace::rec_1886_inverse_eotf, 0.0f, 1.0f, 1e-6f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::rec_1886_inverse_eotf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::rec_1886_eotf, 1.0f, 2.0f, 1UL << 16);

	SCOPED_TRACE("btb");
	test_monotonic(zimg::colorspace::rec_1886_inverse_eotf, -1.0f, 0.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::rec_1886_eotf, -1.0f, 0.0f, 1UL << 16);
}

TEST(GammaTest, test_srgb)
{
	EXPECT_EQ(0.0f, zimg::colorspace::srgb_inverse_eotf(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::srgb_inverse_eotf(1.0f));
	EXPECT_EQ(0.0f, zimg::colorspace::srgb_eotf(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::srgb_eotf(1.0f));

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::srgb_inverse_eotf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::srgb_eotf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::srgb_inverse_eotf, zimg::colorspace::srgb_eotf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::srgb_eotf, zimg::colorspace::srgb_inverse_eotf, 0.0f, 1.0f, 1e-6f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::srgb_inverse_eotf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::srgb_eotf, 1.0f, 2.0f, 1UL << 16);

	SCOPED_TRACE("btb");
	test_monotonic(zimg::colorspace::srgb_inverse_eotf, -1.0f, 0.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::srgb_eotf, -1.0f, 0.0f, 1UL << 16);
}

TEST(GammaTest, test_xvycc)
{
	SCOPED_TRACE("oetf forward");
	test_accuracy(zimg::colorspace::xvycc_oetf, zimg::colorspace::rec_709_inverse_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("oetf reverse");
	test_accuracy(zimg::colorspace::xvycc_inverse_oetf, zimg::colorspace::rec_709_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("eotf forward");
	test_accuracy(zimg::colorspace::xvycc_oetf, zimg::colorspace::rec_709_inverse_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("eotf reverse");
	test_accuracy(zimg::colorspace::xvycc_inverse_oetf, zimg::colorspace::rec_709_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);

	SCOPED_TRACE("oetf wtw");
	test_monotonic(zimg::colorspace::xvycc_oetf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::xvycc_inverse_oetf, 1.0f, 2.0f, 1UL << 16);
	SCOPED_TRACE("eotf wtw");
	test_monotonic(zimg::colorspace::xvycc_inverse_eotf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::xvycc_eotf, 1.0f, 2.0f, 1UL << 16);
	SCOPED_TRACE("oetf btb");
	test_monotonic(zimg::colorspace::xvycc_oetf, -1.0f, 0.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::xvycc_inverse_oetf, -1.0f, 0.0f, 1UL << 16);
	SCOPED_TRACE("eotf btb");
	test_monotonic(zimg::colorspace::xvycc_inverse_eotf, -1.0f, 0.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::xvycc_eotf, -1.0f, 1.0f, 1UL << 16);
}

TEST(GammaTest, test_st_2084)
{
	EXPECT_EQ(0.0f, zimg::colorspace::st_2084_inverse_eotf(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::st_2084_inverse_eotf(1.0f));
	EXPECT_EQ(0.0f, zimg::colorspace::st_2084_eotf(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::st_2084_eotf(1.0f));

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::st_2084_inverse_eotf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::st_2084_eotf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::st_2084_inverse_eotf, zimg::colorspace::st_2084_eotf, 0.0f, 1.0f, 1e-4f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::st_2084_eotf, zimg::colorspace::st_2084_inverse_eotf, 0.0f, 1.0f, 1e-4f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::st_2084_inverse_eotf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::st_2084_eotf, 1.0f, 2.0f, 1UL << 16);
}

TEST(GammaTest, test_st_2084_oetf)
{
	EXPECT_EQ(0.0f, zimg::colorspace::st_2084_oetf(0.0f));
	EXPECT_NEAR(1.0f, zimg::colorspace::st_2084_oetf(1.0f), 1e-6f);
	EXPECT_EQ(0.0f, zimg::colorspace::st_2084_inverse_oetf(0.0f));
	EXPECT_NEAR(1.0f, zimg::colorspace::st_2084_inverse_oetf(1.0f), 1e-6f);

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::st_2084_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::st_2084_inverse_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::st_2084_oetf, zimg::colorspace::st_2084_inverse_oetf, 0.0f, 1.0f, 1e-4f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::st_2084_inverse_oetf, zimg::colorspace::st_2084_oetf, 0.0f, 1.0f, 1e-4f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::st_2084_oetf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::st_2084_inverse_oetf, 1.0f, 2.0f, 1UL << 16);
}

TEST(GammaTest, test_arib_b67)
{
	EXPECT_EQ(0.0f, zimg::colorspace::arib_b67_oetf(0.0f));
	EXPECT_EQ(0.5f, zimg::colorspace::arib_b67_oetf(1.0f / 12.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::arib_b67_oetf(1.0f));
	EXPECT_EQ(0.0f, zimg::colorspace::arib_b67_inverse_oetf(0.0f));
	EXPECT_EQ(1.0f / 12.0f, zimg::colorspace::arib_b67_inverse_oetf(0.5f));
	EXPECT_NEAR(1.0f, zimg::colorspace::arib_b67_inverse_oetf(1.0f), 1e-6f);

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::arib_b67_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::arib_b67_inverse_oetf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::arib_b67_oetf, zimg::colorspace::arib_b67_inverse_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::arib_b67_inverse_oetf, zimg::colorspace::arib_b67_oetf, 0.0f, 1.0f, 1e-6f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::arib_b67_oetf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::arib_b67_inverse_oetf, 1.0f, 2.0f, 1UL << 16);
}

TEST(GammaTest, test_arib_b67_eotf)
{
	EXPECT_EQ(0.0f, zimg::colorspace::arib_b67_inverse_eotf(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::arib_b67_inverse_eotf(1.0f));
	EXPECT_EQ(0.0f, zimg::colorspace::arib_b67_eotf(0.0f));
	EXPECT_NEAR(1.0f, zimg::colorspace::arib_b67_eotf(1.0f), 1e-6f);

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::arib_b67_inverse_eotf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::arib_b67_eotf, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::arib_b67_inverse_eotf, zimg::colorspace::arib_b67_eotf, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::arib_b67_eotf, zimg::colorspace::arib_b67_inverse_eotf, 0.0f, 1.0f, 1e-6f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::arib_b67_inverse_eotf, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::arib_b67_eotf, 1.0f, 2.0f, 1UL << 16);
}
