#pragma once

#ifndef ZIMG_EXCEPT_H_
#define ZIMG_EXCEPT_H_

#include <stdexcept>

namespace zimg {;

namespace error {;

class Exception : private std::runtime_error {
public:
	Exception() : std::runtime_error{ "" } {}

	explicit Exception(const char *msg) : std::runtime_error{ msg } {}

	explicit Exception(const std::string &msg) : std::runtime_error{ msg } {}

	virtual ~Exception() {}

	using std::runtime_error::what;
};


#define DECLARE_EXCEPTION(x, base) \
class x : public base { \
public: \
  x() : base{ "" } {} \
  explicit x(const char *msg) : base{ msg } {} \
  explicit x(const std::string &msg) : base{ msg } {} \
};

DECLARE_EXCEPTION(UnknownError, Exception)
DECLARE_EXCEPTION(InternalError, Exception)

DECLARE_EXCEPTION(OutOfMemory, Exception)
DECLARE_EXCEPTION(UserCallbackFailed, Exception)

DECLARE_EXCEPTION(LogicError, Exception)
DECLARE_EXCEPTION(GreyscaleSubsampling, LogicError)
DECLARE_EXCEPTION(ColorFamilyMismatch, LogicError)
DECLARE_EXCEPTION(ImageNotDivislbe, LogicError)
DECLARE_EXCEPTION(BitDepthOverflow, LogicError)

DECLARE_EXCEPTION(IllegalArgument, Exception)
DECLARE_EXCEPTION(EnumOutOfRange, IllegalArgument)
DECLARE_EXCEPTION(ZeroImageSize, IllegalArgument)

DECLARE_EXCEPTION(UnsupportedOperation, Exception)
DECLARE_EXCEPTION(UnsupportedSubsampling, UnsupportedOperation)
DECLARE_EXCEPTION(NoColorspaceConversion, UnsupportedOperation)
DECLARE_EXCEPTION(ResamplingNotAvailable, UnsupportedOperation)
DECLARE_EXCEPTION(NoFieldParityConversion, UnsupportedOperation)

#undef DECLARE_EXCEPTION

} // namespace error

typedef error::Exception ZimgException;
typedef error::UnknownError ZimgUnknownError;
typedef error::InternalError ZimgInternalError;
typedef error::OutOfMemory ZimgOutOfMemory;
typedef error::LogicError ZimgLogicError;
typedef error::IllegalArgument ZimgIllegalArgument;
typedef error::UnsupportedOperation ZimgUnsupportedError;

} // namespace zimg

#endif // ZIMG_EXCEPT_H_
