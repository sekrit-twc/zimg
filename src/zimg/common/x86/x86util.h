#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_X86_X86UTIL_H_
#define ZIMG_X86_X86UTIL_H_

#include <cstdint>
#include "common/ccdep.h"

namespace zimg {

// The n-th mask vector has the lower n bytes set to all-ones.
extern const uint8_t xmm_mask_table alignas(16)[17][16];
extern const uint8_t ymm_mask_table alignas(32)[33][32];

} // namespace zimg

#endif // ZIMG_X86_X86UTIL_H_

#endif // ZIMG_X86
