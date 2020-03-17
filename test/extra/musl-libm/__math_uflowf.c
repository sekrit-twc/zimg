#include "libm.h"

float my__math_uflowf(uint32_t sign)
{
	return my__math_xflowf(sign, 0x1p-95f);
}
