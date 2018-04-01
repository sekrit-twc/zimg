#ifdef ZIMG_X86_AVX512

#include <cmath>

#include "colorspace/x86/gamma_constants_avx512.h"
#include "colorspace/gamma.h"
#include "common/x86/cpuinfo_x86.h"
#include "gtest/gtest.h"

namespace {

void test_gamma_to_linear(float (*f)(float), float (*g)(float), float min, float max, float errthr, float biasthr)
{
	if (!zimg::query_x86_capabilities().avx512f) {
		SUCCEED() << "avx512 not available, skipping";
		return;
	}

	zimg::colorspace::EnsureSinglePrecision x87;

	const unsigned long STEPS = 1UL << 16;
	float err = 0.0f;
	float bias = 0.0f;

	for (unsigned long i = 0; i <= STEPS; ++i) {
		float x = min + i * ((max - min) / STEPS);
		float ref = f(x);
		float test = g(x);
		float err_local = (test - ref) / (ref == 0.0f ? FLT_EPSILON : ref);

		err += std::fabs(err_local);
		bias += test - ref;
	}

	err /= (STEPS + 1);
	bias /= (STEPS + 1);

	EXPECT_LT(err, errthr);
	EXPECT_LT(std::fabs(bias), biasthr);
}

void test_linear_to_gamma(float (*f)(float), float (*g)(float), float min, float max, float errthr, float biasthr)
{
	if (!zimg::query_x86_capabilities().avx512f) {
		SUCCEED() << "avx512 not available, skipping";
		return;
	}

	zimg::colorspace::EnsureSinglePrecision x87;

	const unsigned long STEPS = 1UL << 16;
	float err = 0.0f;
	float bias = 0.0f;

	for (unsigned long i = 0; i <= STEPS; ++i) {
		float x = std::exp2(min + i * ((max - min) / STEPS));
		float ref = f(x);
		float test = g(x);
		float err_local = test - ref;

		err += std::fabs(err_local);
		bias += test - ref;
	}

	err /= (STEPS + 1);
	bias /= (STEPS + 1);

	EXPECT_LT(err, errthr);
	EXPECT_LT(std::fabs(bias), biasthr);
}

} // namespace

#endif // ZIMG_X86_AVX512
