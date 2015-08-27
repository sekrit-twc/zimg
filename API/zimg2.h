#ifndef ZIMG2_H_
#define ZIMG2_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {;
#endif

/** @file */

/**
 * Greatest version of API described by this header.
 *
 * Generally, later versions of the API are backwards-compatible
 * with prior versions. In order to maintain compatibility with the maximum
 * number of library versions, the user should pass the lowest required
 * API version wherever possible to relevant API functions.
 *
 * A number of structure definitions described in this header begin with
 * a member indicating the API version used by the caller. Whenver such
 * a structure is a parameter to a function, the version field should be set
 * to the API version corresponding to its layout to ensure that the library
 * does not access memory beyond the end of the structure.
 */
#define ZIMG_API_VERSION 2

/**
 * Get the version number of the library.
 *
 * This function should not be used to query for API details.
 * Instead, use {@link zimg2_get_api_version} to obtain the API version.
 *
 * @see zimg2_get_api_version
 *
 * @pre major != 0 && minor != 0 && micro != 0
 * @param[out] major set to the major version
 * @param[out] minor set to the minor verison
 * @param[out] micro set to the micro (patch) version
 */
void zimg2_get_version_info(unsigned *major, unsigned *minor, unsigned *micro);

/**
 * Get the API version supported by the library.
 * The API version is separate from the library version.
 *
 * @see zimg2_get_version_info
 *
 * @return API version number
 */
unsigned zimg2_get_api_version(void);


/**
 * Library error codes.
 *
 * API functions may return error codes not listed in this header.
 * The user should not rely solely on specific error codes being returned,
 * but also gracefully handle error categories (multiples of 100 and 1000).
 *
 * Functions returning error codes return 0 on success.
 */
#define ZIMG_ERROR_UNKNOWN          (-1)
#define ZIMG_ERROR_LOGIC             100 /**< Internal logic error. */
#define ZIMG_ERROR_OUT_OF_MEMORY     200 /**< Error allocating internal structures. */
#define ZIMG_ERROR_ILLEGAL_ARGUMENT  300 /**< Illegal value provided for argument. */
#define ZIMG_ERROR_UNSUPPORTED       400 /**< Operation not supported. */

/**
 * Get information regarding the last error to occur.
 *
 * The error code is stored per-thread. It is not reset when a function
 * completes successfully, but only upon calling {@link zimg_clear_last_error}.
 *
 * @see zimg_clear_last_error
 *
 * @param[out] err_msg buffer to receive the error message, may be NULL
 * @param n length of {@p err_msg} buffer in bytes
 * @return error code
 */
int zimg_get_last_error(char *err_msg, size_t n);

/**
 * Clear the stored error code.
 *
 * @see zimg_get_last_error
 *
 * @post zimg_get_last_error() == 0
 */
void zimg_clear_last_error(void);


/**
 * CPU feature set constants.
 *
 * Available values are defined on a per-architecture basis.
 * Constants are not implied to be in any particular order.
 */
#define ZIMG_CPU_NONE 0 /**< Portable C-based implementation. */
#define ZIMG_CPU_AUTO 1 /**< Runtime CPU detection. */

#if defined(__i386) || defined(_M_IX86) || defined(_M_X64) || defined(__x86_64__)
  #define ZIMG_CPU_X86_MMX   1000
  #define ZIMG_CPU_X86_SSE   1001
  #define ZIMG_CPU_X86_SSE2  1002
  #define ZIMG_CPU_X86_SSE3  1003
  #define ZIMG_CPU_X86_SSSE3 1004
  #define ZIMG_CPU_X86_SSE41 1005
  #define ZIMG_CPU_X86_SSE42 1006
  #define ZIMG_CPU_X86_AVX   1007
  #define ZIMG_CPU_X86_F16C  1008 /**< AVX with F16C extension (e.g. Ivy Bridge) */
  #define ZIMG_CPU_X86_AVX2  1009
#endif

/**
 * Set the desired CPU architecture to target.
 *
 * The default value on library startup is {@link ZIMG_CPU_AUTO} since v2.0.
 * In prior versions, the default value was {@link ZIMG_CPU_NONE}.
 *
 * The target CPU affects the implementation of filters created by subsequent
 * API calls. If the CPU architecture selected is not supported by the host
 * processor, the most likely result is a program fault.
 *
 * The target CPU is shared across all threads. This function may be called
 * from multiple threads, but the target CPU will be updated globally and
 * may affect other users of the library.
 *
 * If an unsupported value is passed to this function, it will be treated as
 * though it were ZIMG_CPU_NONE.
 *
 * @param cpu one of the ZIMG_CPU family of constants
 */
void zimg_set_cpu(int cpu);


/**
 * Pixel format constants.
 *
 * Since v2.0, all filter instances are capable of processing all formats.
 * In prior versions, some filters have restrictions on available formats.
 *
 * The use of the {@link ZIMG_PIXEL_HALF} format is likely to be slow
 * on CPU architectures that do not support hardware binary16 operations.
 */
#define ZIMG_PIXEL_BYTE  0 /**< Unsigned integer, one byte per sample. */
#define ZIMG_PIXEL_WORD  1 /**< Unsigned integer, two bytes per sample. */
#define ZIMG_PIXEL_HALF  2 /**< IEEE-754 half precision (binary16). */
#define ZIMG_PIXEL_FLOAT 3 /**< IEEE-754 single precision (binary32). */

/**
 * Pixel range constants for integer formats.
 *
 * Additional range types may be defined besides ZIMG_RANGE_LIMITED and
 * ZIMG_RANGE_FULL. Users should not treat range as a boolean quantity.
 */
#define ZIMG_RANGE_LIMITED 0 /**< Studio (TV) legal range, 16-235 in 8 bits. */
#define ZIMG_RANGE_FULL    1 /**< Full (PC) dynamic range, 0-255 in 8 bits. */


 /**
  * Read-only buffer structure.
  *
  * Image data is read and written from a circular array described by
  * this structure. This structure is used for input parameters, and the
  * {@link zimg_image_buffer} structure for output parameters.
  *
  * The circular array holds a power-of-2 number of image scanlines,
  * where the beginning of the i-th row of the p-th plane is stored at
  * (data[p] + (ptrdiff_t)(i & mask[p]) * stride[p]).
  *
  * The row index mask can be set to the special value of UINT_MAX (-1)
  * to indicate a fully allocated image plane. Filter instances will
  * not read or write beyond image bounds, and no padding is necessary.
  *
  * Generally, the image stride must be a multiple of the alignment
  * imposed by the target CPU architecture, which is 64 bytes on x86/AMD64.
  */
typedef struct zimg_image_buffer_const {
	unsigned version;    /**< @see ZIMG_API_VERSION */
	const void *data[3]; /**< per-plane data buffers */
	ptrdiff_t stride[3]; /**< per-plane stride in bytes */
	unsigned mask[3];    /**< per-plane row index mask */
} zimg_image_buffer_const;

/**
 * Writable buffer structure.
 *
 * This union overlays a read-only {@link zimg_image_buffer_const} and a
 * corresponding structure with writable data pointers. This allows a buffer
 * used as an output parameter to one API call to be reused as an input
 * parameter to a subsequent call.
 *
 * From a strict standards point-of-view, the use of the union to alias the
 * const and mutable buffers may be undefined behaviour in C++ (but not C).
 * This has not been observed to impact GCC and MSVC compilers.
 *
 * @see zimg_image_buffer_const
 */
typedef union zimg_image_buffer {
	struct {
		unsigned version;
		void *data[3];
		ptrdiff_t stride[3];
		unsigned mask[3];
	} m;

	zimg_image_buffer_const c;
} zimg_image_buffer;

/**
 * Filter constraint flags structure.
 *
 * This structure holds flags that document capabilities and constraints for
 * {@link zimg_filter} instances, which affect the valid sequence of API
 * calls that may be made.
 */
typedef struct zimg_filter_flags {
	unsigned version; /**< @see ZIMG_API_VERSION */

	/**
	 * Defines if a filter is stateful or stateless.
	 *
	 * If true, the user must request output scanlines in a strictly consecutive
	 * manner, beginning with the 0-th line and incrementing in units of the
	 * filter step given by {@link zimg2_filter_get_simultaneous_lines}.
	 *
	 * The column index must also not change until the current image tile has
	 * been fully processed.
	 *
	 * If false, output scanlines can be requested in any order.
	 *
	 * A stateful filter implies a non-zero frame context size, but the converse
	 * is not necessarily the case.
	 */
	unsigned char has_state;

	/**
	 * Defines if a filter has a simple mapping of input to output lines.
	 *
	 * If true, the filter produces a group of scanlines using only the
	 * corresponding rows of the input image. If two filters both have this flag
	 * set and additionally have the same step given by
	 * {@link zimg2_filter_get_simultaneous_lines}, then no caches are needed
	 * to link the filters toegther.
	 *
	 * If false, the relationship of input to output lines must be queried by
	 * calling {@link zimg2_filter_get_required_row_range}.
	 */
	unsigned char same_row;

	/**
	 * Defines if a filter can be applied in-place.
	 *
	 * If true, the input and output buffers to {@link zimg2_filter_process} may
	 * point to the same underlying memory object. This implies that the filter
	 * will not re-read previously read pixels.
	 *
	 * If false, the input and output buffers must not overlap.
	 */
	unsigned char in_place;

	/**
	 * Defines if a filter must process entire scanlines.
	 *
	 * If true, calls to {@link zimg2_filter_process} must request a column range
	 * corresponding to the entire row.
	 *
	 * If false, the filter can produce vertical tiles of an image independently,
	 * except as restricted by other flags.
	 */
	unsigned char entire_row;

	/**
	 * Defines if a filter must process an entire plane.
	 *
	 * If true, the filter produces the entire output plane in a single call.
	 *
	 * If false, the filter produces a fixed number of scanlines, given by
	 * {@link zimg2_filter_get_simultaneous_lines}.
	 */
	unsigned char entire_plane;

	/**
	 * Defines if a filter processes one or three image channels.
	 *
	 * If true, the filter reads all three image channels from the input image
	 * and produces three corresponding image channels in the output image.
	 *
	 * If false, the filter reads and writes only the first channel in the
	 * input and output buffers.
	 */
	unsigned char color;
} zimg_filter_flags;

/**
 * Handle to a filter instance.
 *
 * A filter represents an operation that incrementally transforms an image
 * (either greyscale or color) to another image. The output image is generated
 * in increments of a fixed number of lines from a variable number of input
 * lines, which may overlap the input for previous output lines.
 *
 * The incremental nature of the filter processing model is intended to allow
 * multiple filters to be composed in a cache and memory-efficient manner.
 *
 * The filter instance itself is stateless and can be used to process multiple
 * images in multiple threads simultaneously.
 */
typedef struct zimg_filter zimg_filter;

/**
 * Delete the filter.
 *
 * @param ptr filter handle, may be NULL
 */
void zimg2_filter_free(zimg_filter *ptr);

/**
 * Query the filter characteristics.
 *
 * @see zimg_filter_flags
 *
 * @pre flags != 0
 * @param ptr filter handle
 * @param[out] flags pointer to receive filter flags
 * @param version version of flags structure to be written
 * @return error code
 */
int zimg2_filter_get_flags(const zimg_filter *ptr, zimg_filter_flags *flags, unsigned version);

/**
 * Query the input row range required for a given output row range.
 *
 * The filter model imposes the constraint of non-decreasing input range.
 * If this function is called with an {@p i} greater than a previous call,
 * the {@p first} argument is set to a value no less than the previous value.
 * This constraint ensures that input data from previous executions can be
 * safely overwritten with new data.
 *
 * Note that different output rows may require varying numbers of input rows.
 * The maximum input buffer size required is given by
 * {@link zimg2_filter_get_max_buffering}.
 *
 * @pre first != 0 && second != 0
 * @param ptr filter handle
 * @param i index of first line to produce
 * @param[out] first set to the index of first required line
 * @param[out] second set to the index of last required line plus one
 * @return error code
 */
int zimg2_filter_get_required_row_range(const zimg_filter *ptr, unsigned i, unsigned *first, unsigned *second);

/**
 * Query the input column range required for a given vertical tile.
 *
 * Unless the {@link zimg_filter_flags::entire_row} flag is set, filters can
 * produce vertical tiles (stripes) of the output image independently and in
 * any order. However, the required input tile may overlap the tile for
 * previously generated tiles, typically by a fixed overlap. Care must be
 * taken when using composing filters to avoid excessive recomputation.
 *
 * In some cases, the output bounds may exceed the actual dimensions of the
 * input image. In this case, the result should be clamped by the user.
 *
 * @pre first != 0 && second != 0 && right >= left
 * @param ptr filter handle
 * @param left index of left column in desired tile
 * @param right index of right column in desired tile plus one
 * @param[out] first set to the index of the first required column
 * @param[out] second set to the index of the last required column plus one
 * @return error code
 */
int zimg2_filter_get_required_col_range(const zimg_filter *ptr, unsigned left, unsigned right, unsigned *first, unsigned *second);

/**
 * Query the number of scanlines (step) produced in each filter execution.
 *
 * If the filter generates an entire plane at once, the special value of
 * UINT_MAX may be returned. As a result, an overflow may occur if a non-zero
 * row index is incremented by the step in this case. To check for this
 * condition, read the flag {@link zimg_filter_flags::entire_plane}.
 *
 * @pre out != 0
 * @param ptr filter handle
 * @param[out] out set to the filter step
 * @return error code
 */
int zimg2_filter_get_simultaneous_lines(const zimg_filter *ptr, unsigned *out);

/**
 * Query the maximum number of input lines required for any output line.
 *
 * For filters that require an entire plane as input, the special value of
 * UINT_MAX will be returned. Note that this does not imply that the filter
 * will produce the entire output plane simultaneously. To check for this
 * condition, read the flag {@link zimg_filter_flags::entire_plane}.
 *
 * @pre out != 0
 * @param ptr filter handle
 * @param[out] out set to the buffering requirement
 * @return error code
 */
int zimg2_filter_get_max_buffering(const zimg_filter *ptr, unsigned *out);

/**
 * Query the size of the filter frame context.
 *
 * Filters generally do not allocate memory during execution. Instead, any
 * additional memory buffers are provided by the user. These buffers are
 * separated into those with frame and execution scope. The frame context
 * holds state that persists for the entire processing of an image frame
 * and can not be shared between filters or simultaneous frames.
 *
 * All temporary buffers must be aligned appropriately for the target CPU
 * architecture, similar to the alignment of image buffers.
 *
 * @pre out != 0
 * @param ptr filter handle
 * @param[out] out set to the size of the context in bytes
 * @return error code
 */
int zimg2_filter_get_context_size(const zimg_filter *ptr, size_t *out);

/**
 * Query the size of the temporary buffer for a given vertical tile.
 *
 * The temporary buffer holds state that persists for a single call to
 * {@link zimg2_filter_process}. This buffer can be reused with different
 * filters or a different frame from the same filter
 *
 * @see zimg2_filter_get_context_size
 *
 * @pre out != 0 && right >= left
 * @param ptr filter handle
 * @param left index of left column in desired tile
 * @param right index of right column in desired tile
 * @param[out] out set to the size of the buffer in bytes
 * @return error code
 */
int zimg2_filter_get_tmp_size(const zimg_filter *ptr, unsigned left, unsigned right, size_t *out);

/**
 * Initialize the filter state for a frame.
 *
 * This function must be called before beginning processing if the filter has
 * non-zero frame context size.
 *
 * @see zimg2_filter_get_context_size
 *
 * @param ptr filter handle
 * @param ctx[out] ctx initialized with the filter state, may be NULL
 *                     if the filter does not require a frame context.
 * @return error code
 */
int zimg2_filter_init_context(const zimg_filter *ptr, void *ctx);

/**
 * Execute the filter and produce a given range of scanlines.
 *
 * This function never reads nor writes outside of image bounds.
 *
 * @see zimg_filter_get_simultaneous_lines
 * @see zimg_filter_get_context_size
 * @see zimg_filter_flags
 *
 * @param ptr filter handle
 * @param[in,out] ctx frame context
 * @param[in] src input image buffer
 * @param[out] dst output image buffer
 * @param tmp temporary buffer
 * @param i index of first row to produce
 * @param left index of left column in desired tile
 * @param right index of right column in desired tile plus one
 */
int zimg2_filter_process(const zimg_filter *ptr, void *ctx, const zimg_image_buffer_const *src, const zimg_image_buffer *dst, void *tmp,
                         unsigned i, unsigned left, unsigned right);


/**
 * Query the size of the temporary buffer required to process an entire plane.
 *
 * This function is implemented in terms of the zimg2_filter_ functions.
 *
 * @param ptr filter handle
 * @param width image width
 * @param height image height
 * @param[out] out set to the size of the buffer in bytes
 * @return error code
 */
int zimg2_plane_filter_get_tmp_size(const zimg_filter *ptr, int width, int height, size_t *out);

/**
 * Process an entire plane with a filter instance.
 *
 * This function is implemented in terms of the zimg2_filter_ functions.
 *
 * @see zimg2_plane_filter_get_tmp_size
 *
 * @param ptr filter handle
 * @param tmp_pool temporary buffer
 * @param[in] src array of pointers to the input planes
 * @param[out] dst array of pointers to the output planes
 * @param[in] src_stride array of input strides
 * @param[in] dst_stride array of output strides
 * @param width image width
 * @param height image height
 */
int zimg2_plane_filter_process(const zimg_filter *ptr, void *tmp_pool, const void * const src[3], void * const dst[3],
                               const ptrdiff_t src_stride[3], const ptrdiff_t dst_stride[3],
                               unsigned width, unsigned height);


/**
 * Colorspace definition constants.
 *
 * These constants mirror those defined in ITU-T H.264 and H.265.
 */
#define ZIMG_MATRIX_RGB        0
#define ZIMG_MATRIX_709        1
#define ZIMG_MATRIX_470BG      5
#define ZIMG_MATRIX_170M       6 /* Equivalent to 5. */
#define ZIMG_MATRIX_2020_NCL   9
#define ZIMG_MATRIX_2020_CL   10

#define ZIMG_TRANSFER_709      1
#define ZIMG_TRANSFER_601      6 /* Equivalent to 1. */
#define ZIMG_TRANSFER_LINEAR   8
#define ZIMG_TRANSFER_2020_10 14 /* Equivalent to 1. */
#define ZIMG_TRANSFER_2020_12 15 /* Equivalent to 1. */

#define ZIMG_PRIMARIES_709     1
#define ZIMG_PRIMARIES_170M    6
#define ZIMG_PRIMARIES_240M    7 /* Equivalent to 6. */
#define ZIMG_PRIMARIES_2020    9

/**
 * Parameters for {@link zimg2_colorspace_create}.
 *
 * Default values can be obtained by {@link zimg2_colorspace_params_default}.
 */
typedef struct zimg_colorspace_params {
	unsigned version;   /**< @see ZIMG_API_VERSION */

	unsigned width;     /**< Image width (required). */
	unsigned height;    /**< Image height (required). */

	int matrix_in;      /**< Input YUV transform matrix (required). */
	int transfer_in;    /**< Input transfer characteristics (required). */
	int primaries_in;   /**< Input color primaries (required). */

	int matrix_out;     /**< Output YUV transform matrix (required). */
	int transfer_out;   /**< Output transfer characteristics (required). */
	int primaries_out;  /**< Output color primaries (required). */

	int pixel_type;     /**< Input/output pixel type (required). */
	unsigned depth;     /**< Input/output bit depth. Required for integer formats. */
	unsigned range_in;  /**< Input pixel range. Required for integer formats. */
	unsigned range_out; /**< Output pixel range. Required for integer formats. */
} zimg_colorspace_params;

/**
 * Initialize parameters structure with default values.
 *
 * @param[out] ptr structure to be initialized
 * @param version API version used by caller
 */
void zimg2_colorspace_params_default(zimg_colorspace_params *ptr, unsigned version);

/**
 * Create a colorspace conversion filter.
 *
 * Upon failure, a NULL pointer is returned. The function
 * {@link zimg_get_last_error} may be called to obtain the failure reason.
 *
 * @param[in] params structure containing filter parameters
 * @return filter handle, or NULL on failure
 */
zimg_filter *zimg2_colorspace_create(const zimg_colorspace_params *params);

#ifdef ZIMG_API_V1

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
                            int width, int height, const int src_stride[3], const int dst_stride[3], int pixel_type);

/* Delete the context. */
void zimg_colorspace_delete(zimg_colorspace_context *ctx);

#endif /* ZIMG_API_V1 */


/**
 * Dither method constants.
 */
#define ZIMG_DITHER_NONE            0 /**< Round to nearest. */
#define ZIMG_DITHER_ORDERED         1 /**< Bayer patterend dither. */
#define ZIMG_DITHER_RANDOM          2 /**< Pseudo-random noise of magnitude 0.5. */
#define ZIMG_DITHER_ERROR_DIFFUSION 3 /**< Floyd-Steinberg error diffusion. */

/**
 * Parameters for {@link zimg2_depth_create}.
 *
 * Default values can be obtained by {@link zimg2_depth_params_default}.
 */
typedef struct zimg_depth_params {
	unsigned version;   /**< @see ZIMG_API_VERSION */

	unsigned width;     /**< Image width (required). */
	unsigned height;    /**< Image height (required). */

	int dither_type;    /**< Dithering method (default ZIMG_DITHER_NONE). */
	int chroma;         /**< Whether image is a chroma plane (default false). */

	int pixel_in;       /**< Input pixel format (required). */
	unsigned depth_in;  /**< Input bit depth. Required for integer formats. */
	unsigned range_in;  /**< Input pixel range. Rquired for integer formats. */

	int pixel_out;      /**< Output pixel format (required). */
	unsigned depth_out; /**< Output bit depth. Required for integer formats. */
	unsigned range_out; /**< Output pixel range. Required for integer formats. */
} zimg_depth_params;

/**
 * Initialize parameters structure with default values.
 *
 * @param[out] ptr structure to be initialized
 * @param version API version used by caller
 */
void zimg2_depth_params_default(zimg_depth_params *ptr, unsigned version);

/**
 * Create a pixel format conversion filter.
 *
 * Upon failure, a NULL pointer is returned. The function
 * {@link zimg_get_last_error} may be called to obtain the failure reason.
 *
 * @param[in] params structure containing filter parameters
 * @return filter handle, or NULL on failure
 */
zimg_filter *zimg2_depth_create(const zimg_depth_params *params);

#ifdef ZIMG_API_V1

typedef struct zimg_depth_context zimg_depth_context;

/**
 * Create a context to convert between pixel formats using the given [dither_type].
 * On error, a NULL pointer is returned.
 */
zimg_depth_context *zimg_depth_create(int dither_type);

/* Get the temporary buffer size in bytes required to process a plane with [width] using [ctx]. */
size_t zimg_depth_tmp_size(zimg_depth_context *ctx, int width);

/* Process a plane. On success, 0 is returned, else a corresponding error code. */
int zimg_depth_process(zimg_depth_context *ctx, const void *src, void *dst, void *tmp,
                       int width, int height, int src_stride, int dst_stride,
                       int pixel_in, int pixel_out, int depth_in, int depth_out, int fullrange_in, int fullrange_out, int chroma);

/* Delete the context. */
void zimg_depth_delete(zimg_depth_context *ctx);

#endif /* ZIMG_API_V1 */


/**
 * Resampling method constants.
 */
#define ZIMG_RESIZE_POINT    0 /**< Nearest-neighbor filter, never anti-aliased. */
#define ZIMG_RESIZE_BILINEAR 1 /**< Bilinear interpolation. */
#define ZIMG_RESIZE_BICUBIC  2 /**< Bicubic convolution (separable) filter. */
#define ZIMG_RESIZE_SPLINE16 3 /**< "Spline16" filter from AviSynth. */
#define ZIMG_RESIZE_SPLINE36 4 /**< "Spline36" filter from AviSynth. */
#define ZIMG_RESIZE_LANCZOS  5 /**< Lanczos resampling filter with variable number of taps. */

/**
 * Parameters for {@link zimg2_resize_create}.
 *
 * Default values can be obtained by {@link zimg2_resize_params_default}.
 */
typedef struct zimg_resize_params {
	unsigned version;      /**< @see ZIMG_API_VERSION */

	unsigned src_width;    /**< Input image width (required). */
	unsigned src_height;   /**< Input image height (required). */
	unsigned dst_width;    /**< Output image width (required). */
	unsigned dst_height;   /**< Output image height (required). */

	int pixel_type;        /**< Input/output pixel format (required). */
	unsigned depth;        /**< Input/output bit depth. Required for integer formats. */

	/**
	 * Sub-pixel adjustment parameters.
	 *
	 * By default, the resampling filter is applied to the entire input image and
	 * produces an output image that shares the same center position. However, a
	 * user-specified phase shift can be applied to the input, essentially
	 * shifting the image center by some fractional quantity. This can be used
	 * to preserve some other spatial relationship, such as top-left alignment.
	 *
	 * If the phase shift is less than -1.0 or if the combined shift and
	 * image subwidth/subheight is greater than one plus the input dimensions,
	 * an error or other unexpected behavior may occur.
	 *
	 * The default value of {@p shift_w} and {@p shift_h} is 0, and the default
	 * {@p subwidth} and {@p subheight} is NAN, which processes the entire image.
	 *
	 * All quantities are specified in units of input pixels.
	 */
	double shift_w;
	double shift_h;        /**< @see shift_w */
	double subwidth;       /**< @see shift_w */
	double subheight;      /**< @see shift_w */

	int filter_type;       /**< Resampling filter (default ZIMG_RESIZE_POINT). */

	/**
	 * Parameters for resampling filter.
	 *
	 * The meaning of this value depends on the filter selected.
	 *
	 * For ZIMG_RESIZE_BICUBIC, {@p filter_param_a} and {@p filter_param_b} are
	 * the "b" and "c" parameters. If one parameter is specified, the other must
	 * also be specified to avoid unexpected behavior.
	 *
	 * For ZIMG_RESIZE_LANCZOS, {@p filter_param_a} is the number of filter taps.
	 *
	 * The default value is NAN, which results in a library default being used.
	 */
	double filter_param_a;
	double filter_param_b; /**< @see filter_param_a */
} zimg_resize_params;

/**
 * Initialize parameters structure with default values.
 *
 * @param[out] ptr structure to be initialized
 * @param version API version used by caller
 */
void zimg2_resize_params_default(zimg_resize_params *ptr, unsigned version);

/**
 * Create a resampling filter.
 *
 * Upon failure, a NULL pointer is returned. The function
 * {@link zimg_get_last_error} may be called to obtain the failure reason.
 *
 * @param[in] params structure containing filter parameters
 * @return filter handle, or NULL on failure
 */
zimg_filter *zimg2_resize_create(const zimg_resize_params *params);

#ifdef ZIMG_API_V1

typedef struct zimg_resize_context zimg_resize_context;

/**
 * Create a context to apply the given resampling ratio.
 *
 * The meaning of [filter_param_a] and [filter_param_b] depend on the selected filter.
 * Passing NAN for either filter parameter results in a default value being used.
 * For lanczos, "a" is the number of taps, and for bicubic, they are the "b" and "c" parameters.
 *
 * On error, a NULL pointer is returned.
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

#endif /* ZIMG_API_V1 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ZIMG_H_ */
