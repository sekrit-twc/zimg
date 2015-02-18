#pragma once

#ifndef ZIMG_EXCEPT_H_
#define ZIMG_EXCEPT_H_

#include <stdexcept>

namespace zimg {;

class ZimgException : private std::runtime_error {
public:
	ZimgException() : std::runtime_error{ "" } {}

	explicit ZimgException(const char *msg) : std::runtime_error{ msg } {}

	explicit ZimgException(const std::string &msg) : std::runtime_error{ msg } {}

	virtual ~ZimgException() {}

	using std::runtime_error::what;
};

#define ZIMG_DECLARE_EXCEPTION(x) \
class x : public ZimgException { \
public: \
	x() : ZimgException() {} \
	explicit x(const char *msg) : ZimgException(msg) {} \
	explicit x(const std::string &msg) : ZimgException(msg) {} \
};

ZIMG_DECLARE_EXCEPTION(ZimgUnknownError)
ZIMG_DECLARE_EXCEPTION(ZimgLogicError)
ZIMG_DECLARE_EXCEPTION(ZimgOutOfMemory)
ZIMG_DECLARE_EXCEPTION(ZimgIllegalArgument)
ZIMG_DECLARE_EXCEPTION(ZimgUnsupportedError)

#undef ZIMG_DECLARE_EXCEPTION

} // namespace zimg

#endif // ZIMG_EXCEPT_H_
