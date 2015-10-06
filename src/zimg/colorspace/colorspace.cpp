#include <algorithm>
#include <memory>
#include <vector>
#include "common/except.h"
#include "common/linebuffer.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "graph/copy_filter.h"
#include "graph/image_filter.h"
#include "graph/mux_filter.h"
#include "colorspace.h"
#include "colorspace_param.h"
#include "graph.h"
#include "operation.h"

namespace zimg {;
namespace colorspace {;

namespace {;

class ColorspaceConversion final : public graph::ImageFilterBase {
	std::vector<std::unique_ptr<Operation>> m_operations;
	unsigned m_width;
	unsigned m_height;
public:
	ColorspaceConversion(unsigned width, unsigned height, const ColorspaceDefinition &in, const ColorspaceDefinition &out, CPUClass cpu) :
		m_width{ width },
		m_height{ height }
	{
		auto path = get_operation_path(in, out);
		_zassert(!path.empty(), "empty path");

		for (const auto &func : path) {
			m_operations.emplace_back(func(cpu));
		}
	}

	filter_flags get_flags() const override
	{
		filter_flags flags{};

		flags.same_row = true;
		flags.in_place = true;
		flags.color = true;

		return flags;
	}

	image_attributes get_image_attributes() const override
	{
		return{ m_width, m_height, PixelType::FLOAT };
	}

	void process(void *, const graph::ImageBufferConst &src, const graph::ImageBuffer &dst, void *, unsigned i, unsigned left, unsigned right) const override
	{
		const float *src_ptr[3];
		float *dst_ptr[3];

		for (unsigned p = 0; p < 3; ++p) {
			src_ptr[p] = LineBuffer<const float>{ src, p }[i];
			dst_ptr[p] = LineBuffer<float>{ dst, p }[i];
		}

		m_operations[0]->process(src_ptr, dst_ptr, left, right);

		for (size_t i = 1; i < m_operations.size(); ++i) {
			m_operations[i]->process(dst_ptr, dst_ptr, left, right);
		}
	}
};

} // namespace


ColorspaceDefinition ColorspaceDefinition::to(MatrixCoefficients matrix_) const
{
	return{ matrix_, transfer, primaries };
}

ColorspaceDefinition ColorspaceDefinition::to(TransferCharacteristics transfer_) const
{
	return{ matrix, transfer_, primaries };
}

ColorspaceDefinition ColorspaceDefinition::to(ColorPrimaries primaries_) const
{
	return{ matrix, transfer, primaries_ };
}

ColorspaceDefinition ColorspaceDefinition::toRGB() const
{
	return to(MatrixCoefficients::MATRIX_RGB);
}

ColorspaceDefinition ColorspaceDefinition::toLinear() const
{
	return to(TransferCharacteristics::TRANSFER_LINEAR);
}

bool operator==(const ColorspaceDefinition &a, const ColorspaceDefinition &b)
{
	return a.matrix == b.matrix && a.primaries == b.primaries && a.transfer == b.transfer;
}

bool operator!=(const ColorspaceDefinition &a, const ColorspaceDefinition &b)
{
	return !operator==(a, b);
}


graph::ImageFilter *create_colorspace(unsigned width, unsigned height, const ColorspaceDefinition &in, const ColorspaceDefinition &out, CPUClass cpu)
{
	if (in == out) {
		std::unique_ptr<graph::ImageFilter> filter{ new graph::CopyFilter{ width, height, PixelType::FLOAT } };
		std::unique_ptr<graph::ImageFilter> mux{ new graph::MuxFilter{ filter.get(), nullptr } };
		filter.release();
		return mux.release();
	} else {
		return new ColorspaceConversion{ width, height, in, out, cpu };
	}
}

} // namespace colorspace
} // namespace zimg
