#pragma once

#ifndef ZIMG_DEPTH_DITHER_H_
#define ZIMG_DEPTH_DITHER_H_

#include <cstdint>

namespace zimg {;

enum class CPUClass;

template <class T>
class LineBuffer;

struct PixelFormat;

namespace depth {;

enum class DitherType;

/**
 * Base class for dithering conversions.
 */
class DitherConvert {
public:
	/**
	 * Destroy implementation.
	 */
	virtual ~DitherConvert() = 0;

	/**
	 * Convert from byte to byte.
	 *
	 * @param src input buffer
	 * @param dst output buffer
	 * @param src_fmt input pixel format
	 * @param dst_fmt output pixel format
	 * @param n line number
	 * @param tmp temporary buffer (implementation defined size)
	 */
	virtual void byte_to_byte(const LineBuffer<uint8_t> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const = 0;

	/**
	 * Convert from byte to word.
	 *
	 * @see DitherConvert::byte_to_byte
	 */
	virtual void byte_to_word(const LineBuffer<uint8_t> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const = 0;

	/**
	 * Convert from word to word.
	 *
	 * @see DitherConvert::byte_to_byte
	 */
	virtual void word_to_byte(const LineBuffer<uint16_t> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const = 0;

	/**
	 * Convert from word to word.
	 *
	 * @see DitherConvert::byte_to_byte
	 */
	virtual void word_to_word(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const = 0;

	/**
	 * Convert from half precision to byte.
	 *
	 * @see DitherConvert::byte_to_byte
	 */
	virtual void half_to_byte(const LineBuffer<uint16_t> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const = 0;

	/**
	 * Convert from half precision to word.
	 *
	 * @see DitherConvert::byte_to_byte
	 */
	virtual void half_to_word(const LineBuffer<uint16_t> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const = 0;

	/**
	 * Convert from single precision to byte.
	 *
	 * @see DitherConvert::byte_to_byte
	 */
	virtual void float_to_byte(const LineBuffer<float> &src, LineBuffer<uint8_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const = 0;

	/**
	 * Convert from single precision to word.
	 *
	 * @see DitherConvert::byte_to_byte
	 */
	virtual void float_to_word(const LineBuffer<float> &src, LineBuffer<uint16_t> &dst, const PixelFormat &src_fmt, const PixelFormat &dst_fmt, unsigned n, void *tmp) const = 0;
};

/**
 * Create a concrete DitherConvert.
 *
 * @param type dither type
 * @param cpu create implementation optimized for given cpu
 */
DitherConvert *create_dither_convert(DitherType type, CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DITHER_H_
