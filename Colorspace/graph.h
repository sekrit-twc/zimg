#pragma once

#ifndef ZIMG_COLORSPACE_GRAPH_H_
#define ZIMG_COLORSPACE_GRAPH_H_

#include <functional>
#include <vector>
#include "Common/cpuinfo.h"
#include "colorspace_param.h"

namespace zimg {;
namespace colorspace {;

class Operation;

typedef std::function<Operation *(CPUClass)> OperationFactory;

/**
 * Find the shortest path between two colorspaces.
 *
 * @param in input colorspace
 * @param out output colorspace
 * @return vector of factory functors for operations
 */
std::vector<OperationFactory> get_operation_path(const ColorspaceDefinition &in, const ColorspaceDefinition &out);

} // namespace colorspace
} // namespace zimg

#endif // ZIMG_COLORSPACE_GRAPH_H_
