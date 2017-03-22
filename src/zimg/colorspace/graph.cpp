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

constexpr bool is_valid_2020cl(const ColorspaceDefinition &csp)
{
	return csp.matrix == MatrixCoefficients::REC_2020_CL && csp.transfer == TransferCharacteristics::REC_709;
}

constexpr bool is_valid_ictcp(const ColorspaceDefinition &csp)
{
	return csp.matrix == MatrixCoefficients::REC_2100_ICTCP &&
		(csp.transfer == TransferCharacteristics::ST_2084 || csp.transfer == TransferCharacteristics::ARIB_B67) &&
		csp.primaries == ColorPrimaries::REC_2020;
}

constexpr bool is_valid_lms(const ColorspaceDefinition &csp)
{
	return csp.matrix == MatrixCoefficients::REC_2100_LMS &&
		(csp.transfer == TransferCharacteristics::LINEAR || csp.transfer == TransferCharacteristics::ST_2084 || csp.transfer == TransferCharacteristics::ARIB_B67) &&
		csp.primaries == ColorPrimaries::REC_2020;
}

constexpr bool is_valid_csp(const ColorspaceDefinition &csp)
{
	// 1. Require matrix to be set if transfer is set.
	// 2. Require transfer to be set if primaries is set.
	// 3. Check requirements for Rec.2020 CL.
	// 4. Check requirements for Rec.2100 ICtCp.
	// 5. Check requirements for Rec.2100 LMS.
	return !(csp.matrix == MatrixCoefficients::UNSPECIFIED && csp.transfer != TransferCharacteristics::UNSPECIFIED) &&
		!(csp.transfer == TransferCharacteristics::UNSPECIFIED && csp.primaries != ColorPrimaries::UNSPECIFIED) &&
		!(csp.matrix == MatrixCoefficients::REC_2020_CL && !is_valid_2020cl(csp)) &&
		!(csp.matrix == MatrixCoefficients::REC_2100_ICTCP && !is_valid_ictcp(csp)) &&
		!(csp.matrix == MatrixCoefficients::REC_2100_LMS && !is_valid_lms(csp));
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
	using std::placeholders::_1;
	using std::placeholders::_2;

	zassert_d(is_valid_csp(csp), "invalid colorspace");

	std::vector<ColorspaceNode> edges;

	if (csp.matrix == MatrixCoefficients::RGB) {
		// RGB can be converted to YUV with exception of REC_2020_CL, REC_2100_ICTCP, REC_2100_LMS, and UNSPECIFIED
		for (auto coeffs : all_matrix()) {
			if (coeffs != MatrixCoefficients::RGB && coeffs != MatrixCoefficients::REC_2020_CL &&coeffs != MatrixCoefficients::REC_2100_ICTCP &&
			    coeffs != MatrixCoefficients::REC_2100_LMS && coeffs != MatrixCoefficients::UNSPECIFIED) {
				edges.emplace_back(csp.to(coeffs), std::bind(create_ncl_rgb_to_yuv_operation, coeffs, _1, _2));
			}
		}

		// Linear RGB can be converted to other transfer functions and primaries; also to REC_2020_CL and REC_2100_LMS.
		if (csp.transfer == TransferCharacteristics::LINEAR) {
			for (auto transfer : all_transfer()) {
				if (transfer != csp.transfer && transfer != TransferCharacteristics::UNSPECIFIED)
					edges.emplace_back(csp.to(transfer), std::bind(create_linear_to_gamma_operation, transfer, _1, _2));
			}
			for (auto primaries : all_primaries()) {
				if (primaries != csp.primaries && primaries != ColorPrimaries::UNSPECIFIED)
					edges.emplace_back(csp.to(primaries), std::bind(create_gamut_operation, csp.primaries, primaries, _1, _2));
			}

			edges.emplace_back(csp.to(MatrixCoefficients::REC_2020_CL).to(TransferCharacteristics::REC_709), create_2020_cl_rgb_to_yuv_operation);
			if (csp.primaries == ColorPrimaries::REC_2020)
				edges.emplace_back(csp.to(MatrixCoefficients::REC_2100_LMS), std::bind(create_ncl_rgb_to_yuv_operation, MatrixCoefficients::REC_2100_LMS, _1, _2));
		} else if (csp.transfer != TransferCharacteristics::UNSPECIFIED) {
			// Gamma RGB can be converted to linear RGB.
			edges.emplace_back(csp.to_linear(), std::bind(create_gamma_to_linear_operation, csp.transfer, _1, _2));
		}
	} else if (csp.matrix == MatrixCoefficients::REC_2020_CL) {
		edges.emplace_back(csp.to_rgb().to_linear(), create_2020_cl_yuv_to_rgb_operation);
	} else if (csp.matrix == MatrixCoefficients::REC_2100_LMS) {
		// LMS with ST_2084 or ARIB_B67 transfer functions can be converted to ICtCp and also to linear transfer function
		if (csp.transfer == TransferCharacteristics::ST_2084 || csp.transfer == TransferCharacteristics::ARIB_B67) {
			edges.emplace_back(csp.to(MatrixCoefficients::REC_2100_ICTCP), create_lms_to_ictcp_operation);
			edges.emplace_back(csp.to(TransferCharacteristics::LINEAR), std::bind(create_gamma_to_linear_operation, csp.transfer, _1, _2));
		}
		// LMS with linear transfer function can be converted to RGB matrix and to ARIB_B67 and ST_2084 transfer functions
		if (csp.transfer == TransferCharacteristics::LINEAR) {
			edges.emplace_back(csp.to_rgb(), std::bind(create_ncl_yuv_to_rgb_operation, csp.matrix, _1, _2));
			edges.emplace_back(csp.to(TransferCharacteristics::ST_2084), std::bind(create_linear_to_gamma_operation, TransferCharacteristics::ST_2084, _1, _2));
			edges.emplace_back(csp.to(TransferCharacteristics::ARIB_B67), std::bind(create_linear_to_gamma_operation, TransferCharacteristics::ARIB_B67, _1, _2));
		}
	} else if (csp.matrix == MatrixCoefficients::REC_2100_ICTCP) {
		// ICtCp with ST_2084 or ARIB_B67 transfer functions can be converted to LMS
		if (csp.transfer == TransferCharacteristics::ST_2084 || csp.transfer == TransferCharacteristics::ARIB_B67)
			edges.emplace_back(csp.to(MatrixCoefficients::REC_2100_LMS), create_ictcp_to_lms_operation);
	} else if (csp.matrix != MatrixCoefficients::UNSPECIFIED) {
		// YUV can be converted to RGB.
		edges.emplace_back(csp.to_rgb(), std::bind(create_ncl_yuv_to_rgb_operation, csp.matrix, _1, _2));
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
