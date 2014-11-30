#ifndef ZIMG_H_
#define ZIMG_H_

#ifdef __cplusplus
extern "C" {;
#endif

#include <stddef.h>

#define ZIMG_API_VERSION 1

#define ZIMG_ERROR_UNKNOWN           -1
#define ZIMG_ERROR_LOGIC            100
#define ZIMG_ERROR_OUT_OF_MEMORY    200
#define ZIMG_ERROR_ILLEGAL_ARGUMENT 300
#define ZIMG_ERROR_UNSUPPORTED      400

/**
 * Return the last error code. Error information is thread-local.
 * A descriptive error message is placed in the [n]-byte buffer located at [err_msg].
 */
int zimg_get_last_error(char *err_msg, size_t n);

/* Set the last error to 0 and clear the stored error message. */
void zimg_clear_last_error(void);


#define ZIMG_CPU_NONE 0
#define ZIMG_CPU_AUTO 1

#if defined(__i386) || defined(_M_IX86) || defined(_M_X64) || defined(__x86_64__)
  #define ZIMG_CPU_X86_MMX   1000
  #define ZIMG_CPU_X86_SSE   1001
  #define ZIMG_CPU_X86_SSE2  1002
  #define ZIMG_CPU_X86_SSE3  1003
  #define ZIMG_CPU_X86_SSSE3 1004
  #define ZIMG_CPU_X86_SSE41 1005
  #define ZIMG_CPU_X86_SSE42 1006
  #define ZIMG_CPU_X86_AVX   1007
  #define ZIMG_CPU_X86_F16C  1008
  #define ZIMG_CPU_X86_AVX2  1009
#endif

/**
 * Set the desired CPU type to [cpu]. The result is set globally.
 * This function is thread-safe.
 */
void zimg_set_cpu(int cpu);


#define ZIMG_PIXEL_BYTE  0
#define ZIMG_PIXEL_WORD  1
#define ZIMG_PIXEL_HALF  2
#define ZIMG_PIXEL_FLOAT 3


/* Chosen to match ITU-T H.264 and H.265 */
#define ZIMG_MATRIX_RGB        0
#define ZIMG_MATRIX_709        1
#define ZIMG_MATRIX_470BG      5
#define ZIMG_MATRIX_170M       6 /* Equivalent to 5. */
#define ZIMG_MATRIX_2020_NCL   9
#define ZIMG_MATRIX_2020_CL   10

#define ZIMG_TRANSFER_709      1
#define ZIMG_TRANSFER_601      6 /* Equivalent to 1. */
#define ZIMG_TRANSFER_LINEAR   8
#define ZIMG_TRANSFER_2020_10 14 /* The Rec.709 curve is used for both 2020 10-bit and 12-bit. */
#define ZIMG_TRANSFER_2020_12 15

#define ZIMG_PRIMARIES_709     1
#define ZIMG_PRIMARIES_170M    6
#define ZIMG_PRIMARIES_240M    7 /* Equivalent to 6. */
#define ZIMG_PRIMARIES_2020    9

typedef struct zimg_colorspace_context zimg_colorspace_context;

/**
 * Create a context to convert between the described colorspaces.
 * On error, a NULL pointer is returned.
 */
zimg_colorspace_context *zimg_colorspace_create(int matrix_in, int transfer_in, int primaries_in,
                                                int matrix_out, int transfer_out, int primaries_out);

/* Get the temporary buffer size in bytes required to process a frame with [width] using [ctx]. */
size_t zimg_colorspace_tmp_size(zimg_colorspace_context *ctx, int width);

/**
 * Process a frame. The input and output must contain 3 planes.
 * On success, 0 is returned, else a corresponding error code.
 */
int zimg_colorspace_process(zimg_colorspace_context *ctx, const void * const src[3], void * const dst[3], void *tmp,
                            int width, int height, const int *src_stride, const int *dst_stride, int pixel_type);

/* Delete the context. */
void zimg_colorspace_delete(zimg_colorspace_context *ctx);


#define ZIMG_DITHER_NONE            0
#define ZIMG_DITHER_ORDERED         1
#define ZIMG_DITHER_RANDOM          2
#define ZIMG_DITHER_ERROR_DIFFUSION 3

typedef struct zimg_depth_context zimg_depth_context;

/* Create a context to convert between pixel formats using the given [dither_type]. */
zimg_depth_context *zimg_depth_create(int dither_type);

/* Get the temporary buffer size in bytes required to process a plane with [width] using [ctx]. */
size_t zimg_depth_tmp_size(zimg_depth_context *ctx, int width);

/* Process a plane. On success, 0 is returned, else a corresponding error code. */
int zimg_depth_process(zimg_depth_context *ctx, const void *src, void *dst, void *tmp,
                       int width, int height, int src_stride, int dst_stride,
                       int pixel_in, int pixel_out, int depth_in, int depth_out, int fullrange_in, int fullrange_out, int chroma);

/* Delete the context. */
void zimg_depth_delete(zimg_depth_context *ctx);


#define ZIMG_RESIZE_POINT    0
#define ZIMG_RESIZE_BILINEAR 1
#define ZIMG_RESIZE_BICUBIC  2
#define ZIMG_RESIZE_SPLINE16 3
#define ZIMG_RESIZE_SPLINE36 4
#define ZIMG_RESIZE_LANCZOS  5

typedef struct zimg_resize_context zimg_resize_context;

/**
 * Create a context to apply the given resampling ratio.
 * The meaning of [filter_param_a] and [filter_param_b] depend on the selected filter.
 * Passing NAN for either filter parameter results in a default value being used.
 * For lanczos, "a" is the number of taps, and for bicubic, they are the "b" and "c" parameters.
 */
zimg_resize_context *zimg_resize_create(int filter_type, int src_width, int src_height, int dst_width, int dst_height,
                                        double shift_w, double shift_h, double subwidth, double subheight,
                                        double filter_param_a, double filter_param_b);

/* Get the temporary buffer size in bytes required to process a plane with [pixel_type]. */
size_t zimg_resize_tmp_size(zimg_resize_context *ctx, int pixel_type);

/* Process a plane. On success, 0 is returned, else a corresponding error code. */
int zimg_resize_process(zimg_resize_context *ctx, const void *src, void *dst, void *tmp,
                        int src_width, int src_height, int dst_width, int dst_height,
                        int src_stride, int dst_stride, int pixel_type);

/* Delete the context. */
void zimg_resize_delete(zimg_resize_context *ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ZIMG_H_
