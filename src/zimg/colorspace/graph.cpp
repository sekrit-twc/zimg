#include <algorithm>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <type_traits>
#include "common/except.h"
#include "common/zassert.h"
#include "colorspace.h"
#include "graph.h"
#include "operation.h"

namespace zimg {
namespace colorspace {

namespace {

template <class T>
class EnumRange {
	static_assert(std::is_enum<T>::value, "not an enum");
	typedef typename std::underlying_type<T>::type integer_type;

	class iterator {
		T x;

		iterator(T x) : x{ x } {}
	public:
		iterator &operator++()
		{
			x = static_cast<T>(static_cast<integer_type>(x) + 1);
			return *this;
		}

		bool operator!=(const iterator &other) const { return x != other.x; }

		T operator*() const { return x; }

		friend class EnumRange;
	};

	T m_first;
	T m_last;
public:
	EnumRange(T first, T last) :
		m_first{ first },
		m_last{ static_cast<T>(static_cast<integer_type>(last) + 1) }
	{
	}

	iterator begin() const { return m_first; }
	iterator end() const { return m_last; }
};

EnumRange<MatrixCoefficients> all_matrix()
{
	return{ MatrixCoefficients::UNSPECIFIED, MatrixCoefficients::REC_2100_ICTCP };
}

EnumRange<TransferCharacteristics> all_transfer()
{
	return{ TransferCharacteristics::UNSPECIFIED, TransferCharacteristics::ARIB_B67 };
}

EnumRange<ColorPrimaries> all_primaries()
{
	return{ ColorPrimaries::UNSPECIFIED, ColorPrimaries::DCI_P3_D65 };
}

struct ColorspaceHash {
	bool operator()(const ColorspaceDefinition &csp) const
	{
		return std::hash<unsigned>{}(
			(static_cast<unsigned>(csp.matrix) << 16) |
			(static_cast<unsigned>(csp.transfer) << 8) |
			(static_cast<unsigned>(csp.primaries)));
	}
};

typedef std::pair<ColorspaceDefinition, OperationFactory> ColorspaceNode;

std::vector<ColorspaceNode> get_neighboring_colorspaces(const ColorspaceDefinition &csp)
{
	zassert_d(is_valid_csp(csp), "invalid colorspace");

	std::vector<ColorspaceNode> edges;

	auto add_edge = [&](const ColorspaceDefinition &out_csp, decltype(&create_ncl_rgb_to_yuv_operation) func)
	{
		edges.emplace_back(out_csp, std::bind(func, csp, out_csp, std::placeholders::_1, std::placeholders::_2));
	};

	if (csp.matrix == MatrixCoefficients::RGB) {
		// RGB can be converted to YUV with exception of REC_2020_CL, REC_2100_ICTCP, REC_2100_LMS, and UNSPECIFIED
		for (auto coeffs : all_matrix()) {
			if (coeffs != MatrixCoefficients::RGB && coeffs != MatrixCoefficients::REC_2020_CL && coeffs != MatrixCoefficients::CHROMATICITY_DERIVED_CL &&
			    coeffs != MatrixCoefficients::REC_2100_ICTCP && coeffs != MatrixCoefficients::REC_2100_LMS && coeffs != MatrixCoefficients::UNSPECIFIED) {
				add_edge(csp.to(coeffs), create_ncl_rgb_to_yuv_operation);
			}
		}

		// Linear RGB can be converted to other transfer functions and primaries; also to REC_2020_CL and REC_2100_LMS.
		if (csp.transfer == TransferCharacteristics::LINEAR) {
			for (auto transfer : all_transfer()) {
				if (transfer != csp.transfer && transfer != TransferCharacteristics::UNSPECIFIED) {
					add_edge(csp.to(transfer), create_linear_to_gamma_operation);
					add_edge(csp.to(transfer).to(MatrixCoefficients::CHROMATICITY_DERIVED_CL), create_cl_rgb_to_yuv_operation);
				}
			}
			for (auto primaries : all_primaries()) {
				if (primaries != csp.primaries && primaries != ColorPrimaries::UNSPECIFIED)
					add_edge(csp.to(primaries), create_gamut_operation);
			}

			add_edge(csp.to(MatrixCoefficients::REC_2020_CL).to(TransferCharacteristics::REC_709), create_cl_rgb_to_yuv_operation);

			if (csp.primaries == ColorPrimaries::REC_2020)
				add_edge(csp.to(MatrixCoefficients::REC_2100_LMS), create_ncl_rgb_to_yuv_operation);
		} else if (csp.transfer != TransferCharacteristics::UNSPECIFIED) {
			// Gamma RGB can be converted to linear RGB.
			add_edge(csp.to_linear(), create_gamma_to_linear_operation);
		}
	} else if (csp.matrix == MatrixCoefficients::REC_2020_CL || csp.matrix == MatrixCoefficients::CHROMATICITY_DERIVED_CL) {
		add_edge(csp.to_rgb().to_linear(), create_cl_yuv_to_rgb_operation);
	} else if (csp.matrix == MatrixCoefficients::REC_2100_LMS) {
		// LMS with ST_2084 or ARIB_B67 transfer functions can be converted to ICtCp and also to linear transfer function
		if (csp.transfer == TransferCharacteristics::ST_2084 || csp.transfer == TransferCharacteristics::ARIB_B67) {
			add_edge(csp.to(MatrixCoefficients::REC_2100_ICTCP), create_lms_to_ictcp_operation);
			add_edge(csp.to(TransferCharacteristics::LINEAR), create_gamma_to_linear_operation);
		}
		// LMS with linear transfer function can be converted to RGB matrix and to ARIB_B67 and ST_2084 transfer functions
		if (csp.transfer == TransferCharacteristics::LINEAR) {
			add_edge(csp.to_rgb(), create_ncl_yuv_to_rgb_operation);
			add_edge(csp.to(TransferCharacteristics::ST_2084), create_linear_to_gamma_operation);
			add_edge(csp.to(TransferCharacteristics::ARIB_B67), create_linear_to_gamma_operation);
		}
	} else if (csp.matrix == MatrixCoefficients::REC_2100_ICTCP) {
		// ICtCp with ST_2084 or ARIB_B67 transfer functions can be converted to LMS
		if (csp.transfer == TransferCharacteristics::ST_2084 || csp.transfer == TransferCharacteristics::ARIB_B67)
			add_edge(csp.to(MatrixCoefficients::REC_2100_LMS), create_ictcp_to_lms_operation);
	} else if (csp.matrix != MatrixCoefficients::UNSPECIFIED) {
		// YUV can be converted to RGB.
		add_edge(csp.to_rgb(), create_ncl_yuv_to_rgb_operation);
	}

	return edges;
}

} // namespace


std::vector<OperationFactory> get_operation_path(const ColorspaceDefinition &in, const ColorspaceDefinition &out)
{
	if (!is_valid_csp(in) || !is_valid_csp(out))
		error::throw_<error::NoColorspaceConversion>("invalid colorspace definition");

	std::vector<OperationFactory> path;
	std::deque<ColorspaceDefinition> queue;
	std::unordered_set<ColorspaceDefinition, ColorspaceHash> visited;
	std::unordered_map<ColorspaceDefinition, ColorspaceNode, ColorspaceHash> parents;

	ColorspaceDefinition vertex;

	visited.insert(in);
	queue.push_back(in);

	while (!queue.empty()) {
		vertex = queue.front();
		queue.pop_front();

		if (vertex == out)
			break;

		for (auto &&edge : get_neighboring_colorspaces(vertex)) {
			if (visited.find(edge.first) != visited.end())
				continue;

			visited.insert(edge.first);
			queue.push_back(edge.first);
			parents[edge.first] = std::make_pair(vertex, std::move(edge.second));
		}
	}
	if (vertex != out)
		error::throw_<error::NoColorspaceConversion>("no path between colorspaces");

	while (vertex != in) {
		auto it = parents.find(vertex);
		zassert_d(it != parents.end(), "missing link in traversal path");

		ColorspaceNode node = std::move(it->second);
		path.push_back(std::move(node.second));
		vertex = node.first;
	}
	std::reverse(path.begin(), path.end());

	return path;
}

} // namespace colorspace
} // namespace zimg
