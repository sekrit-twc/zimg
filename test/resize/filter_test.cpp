#include <cmath>
#include "resize/filter.h"

#include "gtest/gtest.h"

namespace {

void check_interpolating(const zimg::resize::Filter& f)
{
	unsigned support = f.support();

	EXPECT_EQ(1.0, f(0));
	for (unsigned i = 1; i <= support; ++i) {
		SCOPED_TRACE(i);
		EXPECT_NEAR(0.0, f(-static_cast<double>(i)), 1e-15);
		EXPECT_NEAR(0.0, f(static_cast<double>(i)), 1e-15);
	}
	EXPECT_EQ(0.0, f(std::nextafter(-static_cast<double>(support), -INFINITY)));
	EXPECT_EQ(0.0, f(std::nextafter(static_cast<double>(support), INFINITY)));
}

} // namespace


TEST(FilterTest, test_bilinear)
{
	zimg::resize::BilinearFilter f;
	EXPECT_EQ(1U, f.support());
	check_interpolating(f);
	EXPECT_EQ(0.5, f(0.5));
	EXPECT_EQ(0.5, f(-0.5));
}

TEST(FilterTest, test_bicubic_interpolating)
{
	zimg::resize::BicubicFilter catmull{ 0.0, 0.5 };
	EXPECT_EQ(2U, catmull.support());
	check_interpolating(catmull);
	EXPECT_EQ(0.5625, catmull(0.5));
	EXPECT_EQ(0.5625, catmull(-0.5));
	EXPECT_EQ(-0.0625, catmull(1.5));
	EXPECT_EQ(-0.0625, catmull(-1.5));
}

TEST(FilterTest, test_bicubic_noninterpolating)
{
	zimg::resize::BicubicFilter mitchell{ 1.0 / 3.0, 1.0 / 3.0 };
	EXPECT_DOUBLE_EQ(8.0 / 9.0, mitchell(0.0));
	EXPECT_NEAR(1.0 / 18.0, mitchell(-1.0), 1e-15);
	EXPECT_NEAR(1.0 / 18.0, mitchell(1.0), 1e-15);
	EXPECT_EQ(0.0, mitchell(-2.0));
	EXPECT_EQ(0.0, mitchell(2.0));
	EXPECT_DOUBLE_EQ(77.0 / 144.0, mitchell(0.5));
	EXPECT_DOUBLE_EQ(77.0 / 144.0, mitchell(-0.5));
	EXPECT_NEAR(-5.0 / 144.0 , mitchell(1.5), 1e-15);
	EXPECT_NEAR(-5.0 / 144.0, mitchell(-1.5), 1e-15);
}

TEST(FilterTest, test_spline16)
{
	zimg::resize::Spline16Filter f;
	EXPECT_EQ(2U, f.support());
	check_interpolating(f);
	EXPECT_EQ(0.575, f(0.5));
	EXPECT_EQ(0.575, f(-0.5));
	EXPECT_DOUBLE_EQ(-0.075, f(1.5));
	EXPECT_DOUBLE_EQ(-0.075, f(-1.5));
}

TEST(FilterTest, test_spline36)
{
	zimg::resize::Spline36Filter f;
	EXPECT_EQ(3U, f.support());
	check_interpolating(f);
	EXPECT_NEAR(0.598684, f(0.5), 1e-6);
	EXPECT_NEAR(0.598684, f(-0.5), 1e-6);
	EXPECT_NEAR(-0.118421, f(1.5), 1e-6);
	EXPECT_NEAR(-0.118421, f(-1.5), 1e-6);
	EXPECT_NEAR(0.0197368, f(2.5), 1e-6);
	EXPECT_NEAR(0.0197368, f(-2.5), 1e-6);
}

TEST(FilterTest, test_lanczos)
{
	for (unsigned i = 1; i < 4; ++i) {
		SCOPED_TRACE(i);
		zimg::resize::LanczosFilter f{ i };
		EXPECT_EQ(i, f.support());
		check_interpolating(f);
	}
}
