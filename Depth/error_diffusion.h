#if 0
#pragma once

#ifndef ZIMG_DEPTH_ERROR_DIFFUSION_H_
#define ZIMG_DEPTH_ERROR_DIFFISION_H_

namespace zimg {;

enum class CPUClass;

namespace depth {;

class DitherConvert;

/**
 * Create a concrete DitherConvert based on error-diffusion.
 *
 * @param cpu create implementation optimized for given cpu
 */
DitherConvert *create_error_diffusion(CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_ERROR_DIFFUSION_H_
#endif
