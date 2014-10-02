#include "Common/except.h"
#include "graph.h"
#include "operation.h"

namespace zimg {;
namespace colorspace {;

namespace {

template <class T>
class EnumRange {
	class iterator {
		T x;
	public:
		iterator(T x) : x{ x } {}

		iterator &operator++() { x = static_cast<T>(static_cast<int>(x) + 1); return *this; }
		bool operator!=(const iterator &other) const { return x != other.x; }
		T operator*() const { return x; }
	};

	T m_first;
	T m_last;
public:
	EnumRange(T first, T last) : m_first{ first }, m_last{ last } {}

	iterator begin() const { return{ m_first }; }
	iterator end() const { return{ static_cast<T>(static_cast<int>(m_last) + 1) }; }
};

template <class T>
EnumRange<T> make_range(T first, T last) { return{ first, last }; }

EnumRange<MatrixCoefficients> all_matrix() { return make_range(MatrixCoefficients::MATRIX_RGB, MatrixCoefficients::MATRIX_2020_CL); }
EnumRange<TransferCharacteristics> all_transfer() { return make_range(TransferCharacteristics::TRANSFER_LINEAR, TransferCharacteristics::TRANSFER_709); }
EnumRange<ColorPrimaries> all_primaries() { return make_range(ColorPrimaries::PRIMARIES_SMPTE_C, ColorPrimaries::PRIMARIES_2020); }

bool is_valid_csp(const ColorspaceDefinition &csp)
{
	return !(csp.matrix == MatrixCoefficients::MATRIX_2020_CL && csp.transfer == TransferCharacteristics::TRANSFER_LINEAR);
}

class ColorspaceGraph {
	typedef std::function<Operation *(CPUClass)> operation_type;
	typedef std::pair<size_t, operation_type> edge_type;

	std::vector<ColorspaceDefinition> m_vertices;
	std::vector<std::vector<edge_type>> m_edge;

	size_t index_of(MatrixCoefficients coeffs, TransferCharacteristics transfer, ColorPrimaries primaries) const
	{
		return index_of(ColorspaceDefinition{ coeffs, transfer, primaries });
	}

	size_t index_of(const ColorspaceDefinition &csp) const
	{
		auto it = std::find(m_vertices.begin(), m_vertices.end(), csp);
		if (it == m_vertices.end())
			throw ZimgLogicError{ "colorspace not present in graph" };

		return it - m_vertices.begin();
	}

	void link(const ColorspaceDefinition &a, const ColorspaceDefinition &b, operation_type op)
	{
		m_edge[index_of(a)].emplace_back(index_of(b), op);
	}

	std::vector<operation_type> bfs(size_t in, size_t out) const
	{
		std::vector<operation_type> path;
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
					path.insert(path.begin(), std::find_if(m_edge[prev].begin(), m_edge[prev].end(), [=](const edge_type &e){ return e.first == tail; })->second);
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

		throw ZimgLogicError{ "no path between colorspaces" };
	}

	ColorspaceGraph()
	{
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
			if (csp.matrix == MatrixCoefficients::MATRIX_RGB) {
				// RGB can be converted to YUV.
				for (auto coeffs : all_matrix()) {
					// Only linear RGB can be converted to CL.
					if (coeffs == MatrixCoefficients::MATRIX_2020_CL && csp.transfer == TransferCharacteristics::TRANSFER_LINEAR)
						link(csp, csp.to(coeffs).to(TransferCharacteristics::TRANSFER_709), create_2020_cl_rgb_to_yuv_operation);
					else if (coeffs != MatrixCoefficients::MATRIX_RGB && coeffs != MatrixCoefficients::MATRIX_2020_CL)
						link(csp, csp.to(coeffs), std::bind(create_ncl_rgb_to_yuv_operation, coeffs, std::placeholders::_1));
				}

				// Linear RGB can be converted to gamma to other primaries.
				if (csp.transfer == TransferCharacteristics::TRANSFER_LINEAR) {
					for (auto transfer : all_transfer()) {
						if (transfer != csp.transfer)
							link(csp, csp.to(transfer), std::bind(create_linear_to_gamma_operation, transfer, std::placeholders::_1));
					}
					for (auto primaries : all_primaries()) {
						if (primaries != csp.primaries)
							link(csp, csp.to(primaries), std::bind(create_gamut_operation, csp.primaries, primaries, std::placeholders::_1));
					}
				}

				// Gamma RGB can be converted to linear.
				if (csp.transfer != TransferCharacteristics::TRANSFER_LINEAR)
					link(csp, csp.toLinear(), std::bind(create_gamma_to_linear_operation, csp.transfer, std::placeholders::_1));

			} else {
				// YUV can only be converted to RGB.
				if (csp.matrix == MatrixCoefficients::MATRIX_2020_CL)
					link(csp, csp.toRGB().toLinear(), create_2020_cl_yuv_to_rgb_operation);
				else
					link(csp, csp.toRGB(), std::bind(create_ncl_yuv_to_rgb_operation, csp.matrix, std::placeholders::_1));
			}
		}
	}
public:
	static const ColorspaceGraph g_instance;

	std::vector<operation_type> shortest_path(const ColorspaceDefinition &in, const ColorspaceDefinition &out) const
	{
		return bfs(index_of(in), index_of(out));
	}
};

const ColorspaceGraph ColorspaceGraph::g_instance;

} // namespace


std::vector<OperationFactory> get_operation_path(const ColorspaceDefinition &in, const ColorspaceDefinition &out)
{
	return ColorspaceGraph::g_instance.shortest_path(in, out);
}

} // namespace colorspace
} // namespace zimg
