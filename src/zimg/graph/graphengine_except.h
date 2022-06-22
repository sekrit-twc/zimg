#pragma once

#ifndef ZIMG_GRAPH_GRAPHENGINE_EXCEPT_H_
#define ZIMG_GRAPH_GRAPHENGINE_EXCEPT_H_

namespace graphengine {
struct Exception;
}

namespace zimg {
namespace graph {

[[noreturn]] void rethrow_graphengine_exception(const graphengine::Exception &e);

} // namespace graph
} // namespace graphengine

#endif // ZIMG_GRAPH_GRAPHENGINE_EXCEPT_H_
