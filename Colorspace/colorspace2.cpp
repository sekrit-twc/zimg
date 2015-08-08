#include "Common/except.h"
#include "Common/linebuffer.h"
#include "colorspace.h"
#include "colorspace_param.h"
#include "colorspace2.h"
#include "graph.h"

namespace zimg {;
namespace colorspace {;

namespace {;

bool is_valid_csp(const ColorspaceDefinition &csp)
{
	ColorspaceConversion2 c;
	return !(csp.matrix == MatrixCoefficients::MATRIX_2020_CL && csp.transfer == TransferCharacteristics::TRANSFER_LINEAR);
}

} // namespace


ColorspaceConversion2::ColorspaceConversion2(const ColorspaceDefinition &in, const ColorspaceDefinition &out, CPUClass cpu)
try
{
	if (!is_valid_csp(in) || !is_valid_csp(out))
		throw ZimgIllegalArgument{ "invalid colorspace definition" };

	for (const auto &func : get_operation_path(in, out)) {
		m_operations.emplace_back(func(cpu));
	}
} catch (const std::bad_alloc &) {
	throw ZimgOutOfMemory{};
}

zimg_filter_flags ColorspaceConversion2::get_flags() const
{
	zimg_filter_flags flags{};

	flags.same_row = true;
	flags.in_place = true;
	flags.color = true;

	return flags;
}

void ColorspaceConversion2::process(void *, const zimg_image_buffer *src, const zimg_image_buffer *dst, void *, unsigned i, unsigned left, unsigned right) const
{
	float *buf[3];
	unsigned count = right - left;

	for (int p = 0; p < 3; ++p) {
		LineBuffer<float> src_buf{ reinterpret_cast<float *>(src->data[p]), right, (unsigned)src->stride[p], src->mask[p] };
		LineBuffer<float> dst_buf{ reinterpret_cast<float *>(dst->data[p]), right, (unsigned)dst->stride[p], dst->mask[p] };

		const float *src_p = src_buf[i] + left;
		float *dst_p = dst_buf[i] + left;

		if (src[p].data != dst[p].data)
			std::copy_n(src_p, count, dst_p);

		buf[p] = dst_p;
	}

	for (auto &o : m_operations) {
		o->process(buf, count);
	}
}

} // namespace colorspace
} // namespace zimg
