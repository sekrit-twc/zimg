#include <array>
#include <memory>
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "graph/basic_filter.h"
#include "graph/image_filter.h"
#include "graphengine/filter.h"
#include "colorspace.h"
#include "graph.h"
#include "operation.h"

namespace zimg {
namespace colorspace {

namespace {

class ColorspaceConversionImpl_GE : public graphengine::Filter {
	graphengine::FilterDescriptor m_desc;
	std::array<std::unique_ptr<Operation>, 6> m_operations;

	void build_graph(const ColorspaceDefinition &in, const ColorspaceDefinition &out, const OperationParams &params, CPUClass cpu)
	{
		ColorspaceDefinition csp_in = in;
		ColorspaceDefinition csp_out = out;

		if (!params.scene_referred) {
			if (csp_in.transfer == TransferCharacteristics::SMPTE_240M)
				csp_in.transfer = TransferCharacteristics::REC_709;
			if (csp_out.transfer == TransferCharacteristics::SMPTE_240M)
				csp_out.transfer = TransferCharacteristics::REC_709;
		}

		auto path = get_operation_path(csp_in, csp_out);
		zassert(!path.empty(), "empty path");
		zassert(path.size() <= 6, "too many operations");

		for (size_t i = 0; i < path.size(); ++i) {
			m_operations[i] = path[i](params, cpu);
		}
	}
public:
	ColorspaceConversionImpl_GE(unsigned width, unsigned height, const ColorspaceDefinition &in, const ColorspaceDefinition &out,
	                         const OperationParams &params, CPUClass cpu) :
		m_desc{}
	{
		zassert_d(width <= pixel_max_width(PixelType::FLOAT), "overflow");

		m_desc.format = { width, height, sizeof(float) };
		m_desc.num_deps = 3;
		m_desc.num_planes = 3;
		m_desc.step = 1;
		m_desc.flags.in_place = 1;

		build_graph(in, out, params, cpu);
	}

	const graphengine::FilterDescriptor &descriptor() const noexcept override { return m_desc; }

	std::pair<unsigned, unsigned> get_row_deps(unsigned i) const noexcept override { return{ i, i + 1 }; }

	std::pair<unsigned, unsigned> get_col_deps(unsigned left, unsigned right) const noexcept override { return{ left, right }; }

	void init_context(void *) const noexcept override {}

	void process(const graphengine::BufferDescriptor in[3], const graphengine::BufferDescriptor out[3],
	             unsigned i, unsigned left, unsigned right, void *, void *) const noexcept override
	{
		const float *src_ptr[3];
		float *dst_ptr[3];

		for (unsigned p = 0; p < 3; ++p) {
			src_ptr[p] = in[p].get_line<float>(i);
			dst_ptr[p] = out[p].get_line<float>(i);
		}

		m_operations[0]->process(src_ptr, dst_ptr, left, right);

		if (!m_operations[1])
			return;
		m_operations[1]->process(dst_ptr, dst_ptr, left, right);

		if (!m_operations[2])
			return;
		m_operations[2]->process(dst_ptr, dst_ptr, left, right);

		if (!m_operations[3])
			return;
		m_operations[3]->process(dst_ptr, dst_ptr, left, right);

		if (!m_operations[4])
			return;
		m_operations[4]->process(dst_ptr, dst_ptr, left, right);

		if (!m_operations[5])
			return;
		m_operations[5]->process(dst_ptr, dst_ptr, left, right);
	}
};


class ColorspaceConversionImpl final : public graph::ImageFilterBase {
	std::array<std::unique_ptr<Operation>, 6> m_operations;
	unsigned m_width;
	unsigned m_height;
public:
	ColorspaceConversionImpl(unsigned width, unsigned height, const ColorspaceDefinition &in, const ColorspaceDefinition &out,
	                         const OperationParams &params, CPUClass cpu) :
		m_width{ width },
		m_height{ height }
	{
		zassert_d(width <= pixel_max_width(PixelType::FLOAT), "overflow");

		ColorspaceDefinition csp_in = in;
		ColorspaceDefinition csp_out = out;

		if (!params.scene_referred) {
			if (csp_in.transfer == TransferCharacteristics::SMPTE_240M)
				csp_in.transfer = TransferCharacteristics::REC_709;
			if (csp_out.transfer == TransferCharacteristics::SMPTE_240M)
				csp_out.transfer = TransferCharacteristics::REC_709;
		}

		auto path = get_operation_path(csp_in, csp_out);
		zassert(!path.empty(), "empty path");
		zassert(path.size() <= 6, "too many operations");

		for (size_t i = 0; i < path.size(); ++i) {
			m_operations[i] = path[i](params, cpu);
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

	void process(void *, const graph::ImageBuffer<const void> src[], const graph::ImageBuffer<void> dst[], void *, unsigned i, unsigned left, unsigned right) const override
	{
		const float *src_ptr[3];
		float *dst_ptr[3];

		for (unsigned p = 0; p < 3; ++p) {
			src_ptr[p] = static_cast<const float *>(src[p][i]);
			dst_ptr[p] = static_cast<float *>(dst[p][i]);
		}

		m_operations[0]->process(src_ptr, dst_ptr, left, right);

		if (!m_operations[1])
			return;
		m_operations[1]->process(dst_ptr, dst_ptr, left, right);

		if (!m_operations[2])
			return;
		m_operations[2]->process(dst_ptr, dst_ptr, left, right);

		if (!m_operations[3])
			return;
		m_operations[3]->process(dst_ptr, dst_ptr, left, right);

		if (!m_operations[4])
			return;
		m_operations[4]->process(dst_ptr, dst_ptr, left, right);

		if (!m_operations[5])
			return;
		m_operations[5]->process(dst_ptr, dst_ptr, left, right);
	}
};

} // namespace


ColorspaceConversion::ColorspaceConversion(unsigned width, unsigned height) :
	width{ width },
	height{ height },
	csp_in{},
	csp_out{},
	peak_luminance{ 100.0 },
	approximate_gamma{},
	scene_referred{},
	cpu{ CPUClass::NONE }
{}

std::unique_ptr<graph::ImageFilter> ColorspaceConversion::create() const try
{
	if (width > pixel_max_width(PixelType::FLOAT))
		error::throw_<error::OutOfMemory>();

	OperationParams params;
	params.set_peak_luminance(peak_luminance)
	      .set_approximate_gamma(approximate_gamma)
	      .set_scene_referred(scene_referred);

	if (csp_in == csp_out)
		return std::make_unique<graph::CopyFilter>(width, height, PixelType::FLOAT, true);
	else
		return std::make_unique<ColorspaceConversionImpl>(width, height, csp_in, csp_out, params, cpu);
} catch (const std::bad_alloc &) {
	error::throw_<error::OutOfMemory>();
}

std::unique_ptr<graphengine::Filter> ColorspaceConversion::create_ge() const try
{
	if (width > pixel_max_width(PixelType::FLOAT))
		error::throw_<error::OutOfMemory>();

	if (csp_in == csp_out)
		return nullptr;

	OperationParams params;
	params.set_peak_luminance(peak_luminance)
	      .set_approximate_gamma(approximate_gamma)
	      .set_scene_referred(scene_referred);

	return std::make_unique<ColorspaceConversionImpl_GE>(width, height, csp_in, csp_out, params, cpu);
} catch (const std::bad_alloc &) {
	error::throw_<error::OutOfMemory>();
}

} // namespace colorspace
} // namespace zimg
