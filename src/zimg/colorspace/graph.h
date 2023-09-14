#pragma once

#ifndef ZIMG_COLORSPACE_GRAPH_H_
#define ZIMG_COLORSPACE_GRAPH_H_

#include <functional>
#include <memory>
#include <vector>

namespace zimg {
enum class CPUClass;
}

namespace zimg::colorspace {

struct ColorspaceDefinition;
struct OperationParams;
class Operation;

typedef std::function<std::unique_ptr<Operation>(const OperationParams &, CPUClass)> OperationFactory;

/**
 * Find the shortest path between two colorspaces.
 *
 * @param in input colorspace
 * @param out output colorspace
 * @return vector of factory functors for operations
 */
std::vector<OperationFactory> get_operation_path(const ColorspaceDefinition &in, const ColorspaceDefinition &out);

} // namespace zimg::colorspace

#endif // ZIMG_COLORSPACE_GRAPH_H_
