#ifndef ZIMG3_H_
#define ZIMG3_H_

#include <limits.h>
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
#if 0
#define ZIMG_ERROR_LOGIC             100 /**< Internal logic error. */
#endif
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
 * Pixel format constants.
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
 * Color family constants.
 */
#define ZIMG_COLOR_GREY 0 /**< Single image plane. */
#define ZIMG_COLOR_RGB  1 /**< Generic RGB color image. */
#define ZIMG_COLOR_YUV  2 /**< Generic YUV color image. */

/**
 * Field parity constants.
 *
 * It is possible to process interlaced images with the library by separating
 * them into their individual fields. Each field can then be resized by
 * specifying the appropriate field parity to maintain correct alignment.
 */
#define ZIMG_FIELD_PROGRESSIVE 0 /**< Progressive scan image. */
#define ZIMG_FIELD_TOP         1 /**< Top field of interlaced image. */
#define ZIMG_FIELD_BOTTOM      2 /**< Bottom field of interlaced image. */

/**
 * Chroma location constants.
 *
 * These constants mirror those defined in ITU-T H.264 and H.265.
 *
 * The ITU-T standards define these constants only for 4:2:0 subsampled
 * images, but as an extension, the library also interprets them for other
 * subsampled formats. When used with interlaced images, the field parity must
 * also be provided.
 *
 * Chroma location is always treated as centered on unsubsampled axes.
 */
#define ZIMG_CHROMA_LEFT        0 /**< MPEG-2 */
#define ZIMG_CHROMA_CENTER      1 /**< MPEG-1/JPEG */
#define ZIMG_CHROMA_TOP_LEFT    2 /**< DV */
#define ZIMG_CHROMA_TOP         3
#define ZIMG_CHROMA_BOTTOM_LEFT 4
#define ZIMG_CHROMA_BOTTOM      5

/**
 * Colorspace definition constants.
 *
 * These constants mirror those defined in ITU-T H.264 and H.265.
 *
 * The UNSPECIFIED value is intended to allow colorspace conversions which do
 * not require a fully specified colorspace. The primaries must be defined if
 * the transfer function is defined, and the matrix coefficients must likewise
 * be defined if the transfer function is defined.
 */
#define ZIMG_MATRIX_RGB            0
#define ZIMG_MATRIX_709            1
#define ZIMG_MATRIX_UNSPECIFIED    2
#define ZIMG_MATRIX_470BG          5
#define ZIMG_MATRIX_170M           6 /* Equivalent to 5. */
#define ZIMG_MATRIX_2020_NCL       9
#define ZIMG_MATRIX_2020_CL       10

#define ZIMG_TRANSFER_709          1
#define ZIMG_TRANSFER_UNSPECIFIED  2
#define ZIMG_TRANSFER_601          6 /* Equivalent to 1. */
#define ZIMG_TRANSFER_LINEAR       8
#define ZIMG_TRANSFER_2020_10     14 /* Equivalent to 1. */
#define ZIMG_TRANSFER_2020_12     15 /* Equivalent to 1. */

#define ZIMG_PRIMARIES_709         1
#define ZIMG_PRIMARIES_UNSPECIFIED 2
#define ZIMG_PRIMARIES_170M        6
#define ZIMG_PRIMARIES_240M        7 /* Equivalent to 6. */
#define ZIMG_PRIMARIES_2020        9

/**
 * Dither method constants.
 */
#define ZIMG_DITHER_NONE            0 /**< Round to nearest. */
#define ZIMG_DITHER_ORDERED         1 /**< Bayer patterend dither. */
#define ZIMG_DITHER_RANDOM          2 /**< Pseudo-random noise of magnitude 0.5. */
#define ZIMG_DITHER_ERROR_DIFFUSION 3 /**< Floyd-Steinberg error diffusion. */

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
  * imposed by the target CPU architecture, which is up to 64 bytes on
  * x86 and AMD64. The stride may be negative.
  */
typedef struct zimg_image_buffer_const {
	unsigned version;    /**< @see ZIMG_API_VERSION */
	const void *data[3]; /**< per-plane data buffers, order is R-G-B or Y-U-V */
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
 * Convert a number of lines to a {@link zimg_image_buffer} mask.
 *
 * @param count number of lines, can be UINT_MAX
 * @return buffer mask, can be UINT_MAX
 */
unsigned zimg2_select_buffer_mask(unsigned count);


/**
 * Handle to an image processing contxt.
 *
 * The filter graph constitutes a series of indivdual image manipulations
 * that are executed in sequence to convert an image between formats.
 *
 * A format is defined as the set of attributes that uniquely define the
 * memory representation and interpretation of an image, including its
 * resolution, colorspace, chroma format, and storage.
 */
typedef struct zimg_filter_graph zimg_filter_graph;

/**
 * User callback for custom input/output.
 *
 * The filter graph can be used either to process an in-memory planar image,
 * as defined by a {@link zimg_image_buffer} with a mask of UINT_MAX, or to
 * process images stored in arbitrary packed formats or other address spaces.
 *
 * If provided to {@link zimg2_filter_graph_process}, the callback will be
 * called before image data is read from the input and output buffers.
 * The callback may be invoked on the same pixels multiple times, in which
 * case those pixels must be re-read unless the buffer mask is UINT_MAX.
 *
 * If the image is subsampled, a number of scanlines in units of the chroma
 * subsampling (e.g. 2 lines for 4:2:0) must be processed.
 *
 * If the callback fails, processing will be aborted and non-zero value will
 * be returned to the caller of {@link zimg2_filter_graph_process}, but the
 * return code of the callback will not be propagated.
 *
 * @param user user-defined private data
 * @param i index of first line to read/write
 * @param left index of left column in line
 * @param right index of right column in line plus one
 * @return zero on success or non-zero on failure
 */
typedef int (*zimg_filter_graph_callback)(void *user, unsigned i, unsigned left, unsigned right);

/**
 * Delete the filter graph.
 *
 * @param ptr graph handle, may be NULL
 */
void zimg2_filter_graph_free(zimg_filter_graph *ptr);

/**
 * Query the size of the temporary buffer required to execute the graph.
 *
 * The filter graph does not allocate memory during processing and generally
 * will not fail unless a user-provided callback fails. To facilitate this,
 * memory allocate is delgated to the caller.
 *
 * @pre out != 0
 * @param ptr graph handle
 * @param[out] out set to the size of the buffer in bytes
 * @return error code
 */
int zimg2_filter_graph_get_tmp_size(const zimg_filter_graph *ptr, size_t *out);

/**
 * Query the maximum number of lines required in the input buffer.
 *
 * When reading an image through a user-defined callback function, the loaded
 * image data is stored in a buffer of sufficient size for the granularity of
 * the image filters used.
 *
 * @pre out != 0
 * @param ptr graph handle
 * @param[out] out set to the number of scanlines
 * @return error code
 */
int zimg2_filter_graph_get_input_buffering(const zimg_filter_graph *ptr, unsigned *out);

/**
 * Query the maximum number of lines required in the output buffer.
 *
 * @pre out != 0
 * @param ptr graph handle
 * @param[out] out set to the number of scanlines
 * @return error code
 * @see zimg2_filter_grpah_get_input_buffering
 */
int zimg2_filter_graph_get_output_buffering(const zimg_filter_graph *ptr, unsigned *out);

/**
 * Process an image with the filter graph.
 *
 * @param ptr graph handle
 * @param[in] src input image buffer
 * @param[out] dst output image buffer
 * @param tmp temporary buffer
 * @param unpack_cb user-defined input callback, may be NULL
 * @param unpack_user private data for callback
 * @param pack_cb user-defined output callback, may be NULL
 * @param pack_user private data for callback
 * @return error code
 */
int zimg2_filter_graph_process(const zimg_filter_graph *ptr, const zimg_image_buffer_const *src, const zimg_image_buffer *dst, void *tmp,
                               zimg_filter_graph_callback unpack_cb, void *unpack_user,
                               zimg_filter_graph_callback pack_cb, void *pack_user);


/**
 * Image format descriptor.
 */
typedef struct zimg_image_format {
	unsigned version;             /**< @see ZIMG_API_VERSION */

	unsigned width;               /**< Image width (required). */
	unsigned height;              /**< Image height (required). */
	int pixel_type;               /**< Pixel type (required). */

	unsigned subsample_w;         /**< Horizontal subsampling factor log2 (default 0). */
	unsigned subsample_h;         /**< Vertical subsampling factor log2 (default 0). */

	int color_family;             /**< Color family (default UNSPECIFIED). */
	int matrix_coefficients;      /**< YUV transform matrix (default UNSPECIFIED). */
	int transfer_characteristics; /**< Transfer characteristics (default UNSPECIFIED). */
	int color_primaries;          /**< Color primaries (default UNSPECIFIED). */

	unsigned depth;               /**< Bit depth (default 8 bits per byte). */
	int pixel_range;              /**< Pixel range. Required for integer formats. */

	int field_parity;             /**< Field parity (default ZIMG_FIELD_PROGRESSIVE). */
	int chroma_location;          /**< Chroma location (default ZIMG_CHROMA_LEFT). */
} zimg_image_format;

/**
 * Graph filter parameters.
 */
typedef struct zimg_filter_graph_params {
	unsigned version;         /**< @see ZIMG_API_VERSION */

	int resample_filter;      /**< Luma resampling filter (default ZIMG_RESIZE_BICUBIC). */

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
	double filter_param_b;    /**< @see filter_param_a */

	int resample_filter_uv;   /**< Chroma resampling filter (default ZIMG_RESIZE_BILINEAR) */
	double filter_param_a_uv; /**< @see filter_param_a */
	double filter_param_b_uv; /**< @see filter_param_a */

	int dither_type;          /**< Dithering method (default ZIMG_DITHER_NONE). */

	int cpu_type;             /**< Target CPU architecture (default (ZIMG_CPU_AUTO). */
} zimg_filter_graph_params;

/**
 * Initialize image format structure with default values.
 *
 * @param[out] ptr structure to be initialized
 * @param version API version used by caller
 */
void zimg2_image_format_default(zimg_image_format *ptr, unsigned version);

/**
 * Initialize parameters structure with default values.
 *
 * @param[out] ptr structure to be initialized
 * @param version API version used by caller
 */
void zimg2_filter_graph_params_default(zimg_filter_graph_params *ptr, unsigned version);

/**
 * Create a graph converting the specified formats.
 *
 * Upon failure, a NULL pointer is returned. The function
 * {@link zimg_get_last_error} may be called to obtain the failure reason.
 *
 * @param[in] src_format input image format
 * @param[in] dst_format output image format
 * @param[in] params filter parameters, may be NULL
 * @return graph handle, or NULL on failure
 */
zimg_filter_graph *zimg2_filter_graph_build(const zimg_image_format *src_format, const zimg_image_format *dst_format, const zimg_filter_graph_params *params);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ZIMG_H_ */
