#ifdef ZIMG_X86

#include "x86util.h"

namespace zimg {;

#define REPEAT_1(x) x
#define REPEAT_2(x) REPEAT_1(x), REPEAT_1(x)
#define REPEAT_3(x) REPEAT_2(x), REPEAT_1(x)
#define REPEAT_4(x) REPEAT_2(x), REPEAT_2(x)
#define REPEAT_5(x) REPEAT_4(x), REPEAT_1(x)
#define REPEAT_6(x) REPEAT_4(x), REPEAT_2(x)
#define REPEAT_7(x) REPEAT_4(x), REPEAT_3(x)
#define REPEAT_8(x) REPEAT_4(x), REPEAT_4(x)
#define REPEAT_9(x) REPEAT_8(x), REPEAT_1(x)
#define REPEAT_10(x) REPEAT_8(x), REPEAT_2(x)
#define REPEAT_11(x) REPEAT_8(x), REPEAT_3(x)
#define REPEAT_12(x) REPEAT_8(x), REPEAT_4(x)
#define REPEAT_13(x) REPEAT_8(x), REPEAT_5(x)
#define REPEAT_14(x) REPEAT_8(x), REPEAT_6(x)
#define REPEAT_15(x) REPEAT_8(x), REPEAT_7(x)
#define REPEAT_16(x) REPEAT_8(x), REPEAT_8(x)

const uint8_t xmm_mask_table_l alignas(16)[17][16] = {
	{ REPEAT_16(0x00) },
	{ REPEAT_15(0x00), REPEAT_1(0xFF) },
	{ REPEAT_14(0x00), REPEAT_2(0xFF) },
	{ REPEAT_13(0x00), REPEAT_3(0xFF) },
	{ REPEAT_12(0x00), REPEAT_4(0xFF) },
	{ REPEAT_11(0x00), REPEAT_5(0xFF) },
	{ REPEAT_10(0x00), REPEAT_6(0xFF) },
	{ REPEAT_9(0x00),  REPEAT_7(0xFF) },
	{ REPEAT_8(0x00),  REPEAT_8(0xFF) },
	{ REPEAT_7(0x00),  REPEAT_9(0xFF) },
	{ REPEAT_6(0x00),  REPEAT_10(0xFF) },
	{ REPEAT_5(0x00),  REPEAT_11(0xFF) },
	{ REPEAT_4(0x00),  REPEAT_12(0xFF) },
	{ REPEAT_3(0x00),  REPEAT_13(0xFF) },
	{ REPEAT_2(0x00),  REPEAT_14(0xFF) },
	{ REPEAT_1(0x00),  REPEAT_15(0xFF) },
	{ REPEAT_16(0xFF) }
};

const uint8_t xmm_mask_table_r alignas(16)[17][16] = {
	{ REPEAT_16(0x00) },
	{ REPEAT_1(0xFF),  REPEAT_15(0x00) },
	{ REPEAT_2(0xFF),  REPEAT_14(0x00) },
	{ REPEAT_3(0xFF),  REPEAT_13(0x00) },
	{ REPEAT_4(0xFF),  REPEAT_12(0x00) },
	{ REPEAT_5(0xFF),  REPEAT_11(0x00) },
	{ REPEAT_6(0xFF),  REPEAT_10(0x00) },
	{ REPEAT_7(0xFF),  REPEAT_9(0x00) },
	{ REPEAT_8(0xFF),  REPEAT_8(0x00) },
	{ REPEAT_9(0xFF),  REPEAT_7(0x00) },
	{ REPEAT_10(0xFF), REPEAT_6(0x00) },
	{ REPEAT_11(0xFF), REPEAT_5(0x00) },
	{ REPEAT_12(0xFF), REPEAT_4(0x00) },
	{ REPEAT_13(0xFF), REPEAT_3(0x00) },
	{ REPEAT_14(0xFF), REPEAT_2(0x00) },
	{ REPEAT_15(0xFF), REPEAT_1(0x00) },
	{ REPEAT_16(0xFF) }
};

} // namespace zimg

#endif // ZIMG_X86
