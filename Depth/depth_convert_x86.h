#pragma once

#ifdef ZIMG_X86

#ifndef ZIMG_DEPTH_DEPTH_CONVERT_X86_H_
#define ZIMG_DEPTH_DEPTH_CONVERT_X86_H_

#include "depth_convert.h"

namespace zimg {;

enum class CPUClass;

namespace depth {;

class DepthConvert;

/**
 * Shared DepthConvert helper for all x86 implementations.
 */
class DepthConvertX86 : public DepthConvert {
	template <int N, int M>
	struct Max {
		static const int value = N > M ? N : M;
	};

	template <int N, int M>
	struct Div {
		static const int value = N / M;
	};
protected:
	template <class T, class U, class Unpack, class Pack, class VectorOp, class ScalarOp>
	void process(const T *src, U *dst, int width, Unpack unpack, Pack pack, VectorOp op, ScalarOp scalar_op) const
	{
		typedef typename Unpack::type src_vector_type;
		typedef typename Pack::type dst_vector_type;

		typedef Max<Unpack::loop_step, Pack::loop_step> loop_step;
		typedef Div<loop_step::value, Unpack::loop_step> loop_unroll_unpack;
		typedef Div<loop_step::value, Pack::loop_step> loop_unroll_pack;

		src_vector_type src_unpacked[loop_unroll_unpack::value * Unpack::unpacked_count];
		dst_vector_type dst_unpacked[loop_unroll_pack::value * Pack::unpacked_count];

		for (int i = 0; i < mod(width, loop_step::value); i += loop_step::value) {
			for (int k = 0; k < loop_unroll_unpack::value; ++k) {
				unpack.unpack(&src_unpacked[k * Unpack::unpacked_count], &src[i + k * Unpack::loop_step]);
			}

			for (int k = 0; k < loop_unroll_pack::value * Pack::unpacked_count; ++k) {
				dst_unpacked[k] = op(src_unpacked[k]);
			}

			for (int k = 0; k < loop_unroll_pack::value; ++k) {
				pack.pack(&dst[i + k * Pack::loop_step], &dst_unpacked[k * Pack::unpacked_count]);
			}
		}
		for (int i = mod(width, loop_step::value); i < width; ++i) {
			dst[i] = scalar_op(src[i]);
		}
	}
};

DepthConvert *create_depth_convert_sse2();
DepthConvert *create_depth_convert_avx2();

DepthConvert *create_depth_convert_x86(CPUClass cpu);

} // namespace depth
} // namespace zimg

#endif // ZIMG_DEPTH_DEPTH_CONVERT_X86_H_

#endif // ZIMG_X86
