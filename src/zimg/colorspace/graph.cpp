#include <algorithm>
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
	return{ MatrixCoefficients::UNSPECIFIED, MatrixCoefficients::REC_2020_CL };
}

EnumRange<TransferCharacteristics> all_transfer()
{
	return{ TransferCharacteristics::UNSPECIFIED, TransferCharacteristics::ARIB_B67 };
}

EnumRange<ColorPrimaries> all_primaries()
{
	return{ ColorPrimaries::UNSPECIFIED, ColorPrimaries::DCI_P3_D65 };
}

constexpr bool is_valid_csp(const ColorspaceDefinition &csp)
{
	return !(csp.matrix == MatrixCoefficients::REC_2020_CL && csp.transfer != TransferCharacteristics::REC_709) &&
	       !(csp.matrix == MatrixCoefficients::UNSPECIFIED && csp.transfer != TransferCharacteristics::UNSPECIFIED) &&
	       !(csp.transfer == TransferCharacteristics::UNSPECIFIED && csp.primaries != ColorPrimaries::UNSPECIFIED);
}


class ColorspaceGraph {
	typedef std::pair<size_t, OperationFactory> edge_type;

	std::vector<ColorspaceDefinition> m_vertices;
	std::vector<std::vector<edge_type>> m_edge;

	size_t index_of(const ColorspaceDefinition &csp) const
	{
		auto it = std::find(m_vertices.begin(), m_vertices.end(), csp);
		if (it == m_vertices.end())
			throw error::NoColorspaceConversion{ "colorspace not present in database" };

		return it - m_vertices.begin();
	}

	void link(const ColorspaceDefinition &a, const ColorspaceDefinition &b, OperationFactory op)
	{
		m_edge[index_of(a)].emplace_back(index_of(b), op);
	}

	std::vector<OperationFactory> bfs(size_t in, size_t out) const
	{
		std::vector<OperationFactory> path;
		std::vector<size_t> queue;
		std::vector<size_t> visited;
		std::vector<size_t> parents(m_vertices.size());

		visited.push_back(in);
		queue.push_back(in);

		while (!queue.empty()) {
			size_t vertex = queue.front();
			queue.erase(queue.begin());

			if (vertex == out) {
				size_t tail = vertex;
				size_t prev;

				while (tail != in) {
					prev = parents[tail];

					auto link = std::find_if(m_edge[prev].begin(), m_edge[prev].end(), [=](const edge_type &e) { return e.first == tail; });
					zassert_d(link != m_edge[prev].end(), "missing link in traversal path");
					path.insert(path.begin(), link->second);

					tail = prev;
				}

				return path;
			}

			for (auto &i : m_edge[vertex]) {
				size_t adj = i.first;
				if (std::find(visited.begin(), visited.end(), adj) == visited.end()) {
					visited.push_back(adj);
					queue.push_back(adj);

					parents[adj] = vertex;
				}
			}
		}

		throw error::NoColorspaceConversion{ "no path between colorspaces" };
	}
public:
	ColorspaceGraph()
	{
		using std::placeholders::_1;
		using std::placeholders::_2;

		static_assert(!is_valid_csp({ MatrixCoefficients::REC_2020_CL, TransferCharacteristics::LINEAR, ColorPrimaries::REC_2020 }), "accepted bad colorspace");
		static_assert(!is_valid_csp({ MatrixCoefficients::REC_709, TransferCharacteristics::UNSPECIFIED, ColorPrimaries::REC_709 }), "accepted bad colorspace");
		static_assert(!is_valid_csp({ MatrixCoefficients::UNSPECIFIED, TransferCharacteristics::REC_709, ColorPrimaries::REC_709 }), "accepted bad colorspace");

		// Insert all colorspaces.
		for (auto coeffs : all_matrix()) {
			for (auto transfer : all_transfer()) {
				for (auto primaries : all_primaries()) {
					if (is_valid_csp(ColorspaceDefinition{ coeffs, transfer, primaries }))
						m_vertices.emplace_back(ColorspaceDefinition{ coeffs, transfer, primaries });
				}
			}
		}
		m_edge.resize(m_vertices.size());

		// Find all possible conversions.
		for (auto &csp : m_vertices) {
			if (csp.matrix == MatrixCoefficients::RGB) {
				// RGB can be converted to YUV.
				for (auto coeffs : all_matrix()) {
					// Only linear RGB can be converted to CL.
					if (coeffs == MatrixCoefficients::REC_2020_CL && csp.transfer == TransferCharacteristics::LINEAR)
						link(csp, csp.to(coeffs).to(TransferCharacteristics::REC_709), create_2020_cl_rgb_to_yuv_operation);
					else if (coeffs != MatrixCoefficients::RGB && coeffs != MatrixCoefficients::REC_2020_CL && coeffs != MatrixCoefficients::UNSPECIFIED)
						link(csp, csp.to(coeffs), std::bind(create_ncl_rgb_to_yuv_operation, coeffs, _1, _2));
				}

				// Linear RGB can be converted to gamma to other primaries.
				if (csp.transfer == TransferCharacteristics::LINEAR) {
					for (auto transfer : all_transfer()) {
						if (transfer != csp.transfer && transfer != TransferCharacteristics::UNSPECIFIED)
							link(csp, csp.to(transfer), std::bind(create_linear_to_gamma_operation, transfer, _1, _2));
					}
					for (auto primaries : all_primaries()) {
						if (primaries != csp.primaries)
							link(csp, csp.to(primaries), std::bind(create_gamut_operation, csp.primaries, primaries, _1, _2));
					}
				}

				// Gamma RGB can be converted to linear.
				if (csp.transfer != TransferCharacteristics::LINEAR && csp.transfer != TransferCharacteristics::UNSPECIFIED)
					link(csp, csp.to_linear(), std::bind(create_gamma_to_linear_operation, csp.transfer, _1, _2));
			} else {
				// YUV can only be converted to RGB.
				if (csp.matrix == MatrixCoefficients::REC_2020_CL)
					link(csp, csp.to_rgb().to_linear(), create_2020_cl_yuv_to_rgb_operation);
				else if (csp.matrix != MatrixCoefficients::UNSPECIFIED)
					link(csp, csp.to_rgb(), std::bind(create_ncl_yuv_to_rgb_operation, csp.matrix, _1, _2));
			}
		}
	}

	std::vector<OperationFactory> shortest_path(const ColorspaceDefinition &in, const ColorspaceDefinition &out) const
	{
		return bfs(index_of(in), index_of(out));
	}
};

} // namespace


std::vector<OperationFactory> get_operation_path(const ColorspaceDefinition &in, const ColorspaceDefinition &out)
{
	static const ColorspaceGraph graph{};
	return graph.shortest_path(in, out);
}

} // namespace colorspace
} // namespace zimg
