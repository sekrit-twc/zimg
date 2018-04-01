#ifdef ZIMG_X86_AVX512

#include <algorithm>
#include <cmath>
#include "gamma_constants_avx512.h"

namespace zimg {
namespace colorspace {
namespace avx512constants {

const float Rec1886EOTF::horner[6] = {
	3.9435861748560828e-3f,
	-4.7562005414496558e-2f,
	3.7223934327652687e-1f,
	8.0976688115998239e-1f,
	-1.6576549352714846e-1f,
	2.7378258372778144e-2f,
};

const float Rec1886EOTF::table alignas(64)[16] = {
	0.00000000e+00f, // [-inf, -14)
	7.68054690e-11f, // [-14, -13)
	4.05381696e-10f, // [-13, -12)
	2.13961742e-09f, // [-12, -11)
	1.12929684e-08f, // [-11, -10)
	5.96046448e-08f, // [-10, -9)
	3.14595201e-07f, // [-9, -8)
	1.66044343e-06f, // [-8, -7)
	8.76387295e-06f, // [-7, -6)
	4.62559987e-05f, // [-6, -5)
	2.44140625e-04f, // [-5, -4)
	1.28858194e-03f, // [-4, -3)
	6.80117628e-03f, // [-3, -2)
	3.58968236e-02f, // [-2, -1)
	1.89464571e-01f, // [-1, 0)
	1.00000000e+00f, // [0, 1)
};

const float Rec1886InverseEOTF::horner[6] = {
	5.3331316297790816e-3f,
	-5.0653335401261418e-2f,
	2.0631810268332693e-1f,
	-4.8846483066245743e-1f,
	9.5048057786988787e-1f,
	3.7698771958831039e-1f,
};

const float Rec1886InverseEOTF::table alignas(64)[32] = {
	0.00000000e+00f, // [-inf, -30)
	1.72633492e-04f, // [-30, -29)
	2.30438065e-04f, // [-29, -28)
	3.07597913e-04f, // [-28, -27)
	4.10593953e-04f, // [-27, -26)
	5.48077172e-04f, // [-26, -25)
	7.31595252e-04f, // [-25, -24)
	9.76562500e-04f, // [-24, -23)
	1.30355455e-03f, // [-23, -22)
	1.74003656e-03f, // [-22, -21)
	2.32267015e-03f, // [-21, -20)
	3.10039268e-03f, // [-20, -19)
	4.13852771e-03f, // [-19, -18)
	5.52427173e-03f, // [-18, -17)
	7.37401807e-03f, // [-17, -16)
	9.84313320e-03f, // [-16, -15)
	1.31390065e-02f, // [-15, -14)
	1.75384695e-02f, // [-14, -13)
	2.34110481e-02f, // [-13, -12)
	3.12500000e-02f, // [-12, -11)
	4.17137454e-02f, // [-11, -10)
	5.56811699e-02f, // [-10, -9)
	7.43254447e-02f, // [-9, -8)
	9.92125657e-02f, // [-8, -7)
	1.32432887e-01f, // [-7, -6)
	1.76776695e-01f, // [-6, -5)
	2.35968578e-01f, // [-5, -4)
	3.14980262e-01f, // [-4, -3)
	4.20448208e-01f, // [-3, -2)
	5.61231024e-01f, // [-2, -1)
	7.49153538e-01f, // [-1, 0)
	1.00000000e+00f, // [0, 1)
};


// Debug implementations.
namespace {

template <class T>
float frexp_1_2(T x, int *exp)
{
	if (x == static_cast<T>(0.0)) {
		*exp = 0;
		return x;
	}

	x = std::frexp(x, exp);
	x *= static_cast<T>(2.0);
	*exp -= 1;
	return x;
}

template <class T>
float power_function(float x)
{
	const float input_max = std::nextafterf(2.0f, -INFINITY);
	constexpr int exponent_min = -static_cast<int>(sizeof(T::table) / sizeof(T::table[0]) - 1);

	float orig, mant, mantpart, exppart;
	int exp;

	orig = x;
	x = std::fabs(x);
	x = std::min(x, input_max);

	mant = frexp_1_2(x, &exp);
	exp = x == 0.0f ? -127 : exp;
	exp = std::max(exp, exponent_min) + 127;

	mantpart = T::horner[0];
	for (unsigned i = 1; i < sizeof(T::horner) / sizeof(T::horner[0]); ++i) {
		mantpart = std::fma(mantpart, mant, T::horner[i]);
	}

	exppart = T::table[exp & (sizeof(T::table) / sizeof(T::table[0]) - 1)];
	return std::copysign(mantpart * exppart, orig);
}

} // namespace


float rec_1886_eotf(float x)
{
	return power_function<Rec1886EOTF>(x);
}

float rec_1886_inverse_eotf(float x)
{
	return power_function<Rec1886InverseEOTF>(x);
}

} // namespace avx512constants
} // namespace colorspace
} // namespace zimg

#endif // ZIMG_X86_AVX512
