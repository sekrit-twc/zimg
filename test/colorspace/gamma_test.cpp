#include <cmath>

#include "colorspace/operation_impl.h"
#include "gtest/gtest.h"

namespace {

void test_monotonic(float (*func)(float), float min, float max, unsigned long steps)
{
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
	EXPECT_EQ(0.0f, zimg::colorspace::rec_709_gamma(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::rec_709_gamma(1.0f));
	EXPECT_EQ(0.0f, zimg::colorspace::rec_709_inverse_gamma(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::rec_709_inverse_gamma(1.0f));

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::rec_709_gamma, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::rec_709_inverse_gamma, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::rec_709_gamma, zimg::colorspace::rec_709_inverse_gamma, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::rec_709_inverse_gamma, zimg::colorspace::rec_709_gamma, 0.0f, 1.0f, 1e-6f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::rec_709_gamma, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::rec_709_inverse_gamma, 1.0f, 2.0f, 1UL << 16);

	SCOPED_TRACE("btb");
	test_monotonic(zimg::colorspace::rec_709_gamma, -1.0f, 0.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::rec_709_inverse_gamma, -1.0f, 0.0f, 1UL << 16);
}

TEST(GammaTest, test_st_2084)
{
	EXPECT_EQ(0.0f, zimg::colorspace::st_2084_gamma(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::st_2084_gamma(1.0f));
	EXPECT_EQ(0.0f, zimg::colorspace::st_2084_inverse_gamma(0.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::st_2084_inverse_gamma(1.0f));

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::st_2084_gamma, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::st_2084_inverse_gamma, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::st_2084_gamma, zimg::colorspace::st_2084_inverse_gamma, 0.0f, 1.0f, 1e-4f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::st_2084_inverse_gamma, zimg::colorspace::st_2084_gamma, 0.0f, 1.0f, 1e-4f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::st_2084_gamma, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::st_2084_inverse_gamma, 1.0f, 2.0f, 1UL << 16);
}

TEST(GammaTest, test_arib_b67)
{
	EXPECT_EQ(0.0f, zimg::colorspace::arib_b67_gamma(0.0f));
	EXPECT_EQ(0.5f, zimg::colorspace::arib_b67_gamma(1.0f / 12.0f));
	EXPECT_EQ(1.0f, zimg::colorspace::arib_b67_gamma(1.0f));
	EXPECT_EQ(0.0f, zimg::colorspace::arib_b67_inverse_gamma(0.0f));
	EXPECT_EQ(1.0f / 12.0f, zimg::colorspace::arib_b67_inverse_gamma(0.5f));
	EXPECT_NEAR(1.0f, zimg::colorspace::arib_b67_inverse_gamma(1.0f), 1e-6f);

	SCOPED_TRACE("forward");
	test_monotonic(zimg::colorspace::arib_b67_gamma, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("reverse");
	test_monotonic(zimg::colorspace::arib_b67_inverse_gamma, 0.0f, 1.0f, 1UL << 16);
	SCOPED_TRACE("forward->reverse");
	test_accuracy(zimg::colorspace::arib_b67_gamma, zimg::colorspace::arib_b67_inverse_gamma, 0.0f, 1.0f, 1e-6f, 1e-6f);
	SCOPED_TRACE("reverse->forward");
	test_accuracy(zimg::colorspace::arib_b67_inverse_gamma, zimg::colorspace::arib_b67_gamma, 0.0f, 1.0f, 1e-6f, 1e-6f);

	SCOPED_TRACE("wtw");
	test_monotonic(zimg::colorspace::arib_b67_gamma, 1.0f, 2.0f, 1UL << 16);
	test_monotonic(zimg::colorspace::arib_b67_inverse_gamma, 1.0f, 2.0f, 1UL << 16);
}
