#include <array>
#include <memory>
#include "common/cpuinfo.h"
#include "common/except.h"
#include "common/pixel.h"
#include "common/zassert.h"
#include "graph/filter_base.h"
#include "colorspace.h"
#include "graph.h"
#include "operation.h"

namespace zimg::colorspace {

namespace {

class ColorspaceConversionImpl : public graph::PointFilter {
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
	ColorspaceConversionImpl(unsigned width, unsigned height,
	                         const ColorspaceDefinition &in, const ColorspaceDefinition &out,
	                         const OperationParams &params, CPUClass cpu) :
		PointFilter(width, height, PixelType::FLOAT)
	{
		zassert_d(width <= pixel_max_width(PixelType::FLOAT), "overflow");

		m_desc.num_deps = 3;
		m_desc.num_planes = 3;
		m_desc.flags.in_place = 1;

		build_graph(in, out, params, cpu);
	}

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

std::unique_ptr<graphengine::Filter> ColorspaceConversion::create() const try
{
	if (width > pixel_max_width(PixelType::FLOAT))
		error::throw_<error::OutOfMemory>();

	if (csp_in == csp_out)
		return nullptr;

	OperationParams params;
	params.set_peak_luminance(peak_luminance)
	      .set_approximate_gamma(approximate_gamma)
	      .set_scene_referred(scene_referred);

	return std::make_unique<ColorspaceConversionImpl>(width, height, csp_in, csp_out, params, cpu);
} catch (const std::bad_alloc &) {
	error::throw_<error::OutOfMemory>();
}

} // namespace zimg::colorspace
