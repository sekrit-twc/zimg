#include "libm.h"

float my__math_invalidf(float x)
{
	return (x - x) / (x - x);
}
