#pragma once

#ifdef ZIMG_LOONGARCH

#ifndef ZIMG_LOONGARCH_CPUINFO_LOONGARCH_H_
#define ZIMG_LOONGARCH_CPUINFO_LOONGARCH_H_

namespace zimg {

/**
 * Bitfield of selected LoongArch feature flags.
 */
struct LoongArchCapabilities {
};

LoongArchCapabilities query_loongarch_capabilities() noexcept;

} // namespace zimg

#endif // ZIMG_LOONGARCH_CPUINFO_LOONGARCH_H_
#endif // ZIMG_LOONGARCH
