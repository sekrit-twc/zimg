#pragma once

#ifndef ZIMG_CHECKED_INT_H_
#define ZIMG_CHECKED_INT_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace zimg {

template <class From, class To, class T = void>
using _enable_if_convertible_t = typename std::enable_if<std::is_convertible<From, To>::value, T>::type;

// Integer wrapper that throws on overflow.
template <class T>
class checked_integer {
	static_assert(std::is_integral<T>::value && !std::is_same<T, bool>::value, "must be built-in integer type");

	T m_value;
public:
	// (1) Constructs the object with the given value.
	constexpr checked_integer(const T &value = T()) noexcept;
	// (2) Constructs the object with the given value, checking for overflow.
	template <class U, class = _enable_if_convertible_t<U, T>>
	checked_integer(const U &value);

	// (1) Assigns |value| to the object.
	checked_integer &operator=(const T &value) noexcept;
	// (2) Assigns |value| to the object, checking for overflow.
	template <class U>
	_enable_if_convertible_t<U, T, checked_integer> &operator=(const U &value);

	// Returns the stored value.
	constexpr T get() const noexcept;

	// Checks whether *this is zero.
	constexpr explicit operator bool() const noexcept;

	// (1) Pre-increments the object.
	checked_integer &operator++();
	// (2) Pre-decrements the object.
	checked_integer &operator--();
	// (3) Post-increments the object.
	checked_integer operator++(int);
	// (4) Post-decrements the object.
	checked_integer operator--(int);

	// (1) Adds |other| to *this.
	checked_integer &operator+=(const checked_integer &other);
	// (2) Subtracts |other| from *this.
	checked_integer &operator-=(const checked_integer &other);
	// (3) Multiplies *this by |other|.
	checked_integer &operator*=(const checked_integer &other);
	// (4) Divides *this by |other|.
	checked_integer &operator/=(const checked_integer &other);
	// (5) Modulos *this by |other|.
	checked_integer &operator%=(const checked_integer &other);
	// (6) Left-shifts *this by |n|.
	checked_integer &operator<<=(unsigned n);
	// (7) Right-shifts *this by |n|.
	checked_integer &operator>>=(unsigned n);
	// (8) Bitwise-ANDs *this with |other|.
	checked_integer &operator&=(const checked_integer &other);
	// (9) Bitwise-ORs *this with |other|.
	checked_integer &operator|=(const checked_integer &other);
	// (10) Bitwise-XORs *this with |other|.
	checked_integer &operator^=(const checked_integer &other);
};

// (1) Returns the value of the argument.
template <class T>
checked_integer<T> operator+(const checked_integer<T> &value);
// (2) Negates tbe argument.
template <class T>
checked_integer<T> operator-(const checked_integer<T> &value);
// (3) Returns the bitwise-NOT of the argument.
template <class T>
checked_integer<T> operator~(const checked_integer<T> &value);

// (1 - 3) Returns the sum of the arguments.
template <class T>
checked_integer<T> operator+(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator+(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator+(const U &lhs, const checked_integer<T> &rhs);
// (4 - 6) Returns the result of subtracting rhs from lhs.
template <class T>
checked_integer<T> operator-(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator-(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator-(const U &lhs, const checked_integer<T> &rhs);
// (7 - 9) Multiplies its arguments.
template <class T>
checked_integer<T> operator*(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator*(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator*(const U &lhs, const checked_integer<T> &rhs);
// (10 - 12) Divides lhs by rhs.
template <class T>
checked_integer<T> operator/(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator/(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator/(const U &lhs, const checked_integer<T> &rhs);
// (13 - 15) Returns lhs modulo rhs.
template <class T>
checked_integer<T> operator%(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator%(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator%(const U &lhs, const checked_integer<T> &rhs);

// (1) Left-shifts value by n.
template <class T>
checked_integer<T> operator<<(const checked_integer<T> &value, unsigned n);
// (2) Right-shifts value by n.
template <class T>
checked_integer<T> operator>>(const checked_integer<T> &value, unsigned n);

// (1 - 3) Bitwise-ANDs lhs with rhs.
template <class T>
checked_integer<T> operator&(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator&(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator&(const U &lhs, const checked_integer<T> &rhs);
// (4 - 6) Bitwise-ORs lhs with rhs.
template <class T>
checked_integer<T> operator|(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator|(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator|(const U &lhs, const checked_integer<T> &rhs);
// (7 - 9) Bitwise-XORs lhs with rhs.
template <class T>
checked_integer<T> operator^(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator^(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator^(const U &lhs, const checked_integer<T> &rhs);

// (1 - 3) Compares lhs and rhs for equality.
template <class T>
bool operator==(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator==(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator==(const U &lhs, const checked_integer<T> &rhs);
// (4 - 6) Compares lhs and rhs for inequality.
template <class T>
bool operator!=(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator!=(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator!=(const U &lhs, const checked_integer<T> &rhs);
// (5 - 9) Compares lhs and rhs with less-than operator.
template <class T>
bool operator<(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator<(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator<(const U &lhs, const checked_integer<T> &rhs);
// (10 - 12) Compares lhs and rhs with less-than-or-equals operator.
template <class T>
bool operator<=(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator<=(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator<=(const U &lhs, const checked_integer<T> &rhs);
// (13 - 15) Compares lhs and rhs with greater-than operator.
template <class T>
bool operator>(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator>(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator>(const U &lhs, const checked_integer<T> &rhs);
// (16 - 18) Compares lhs and rhs with greater-than-or-equals operator.
template <class T>
bool operator>=(const checked_integer<T> &lhs, const checked_integer<T> &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator>=(const checked_integer<T> &lhs, const U &rhs);
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator>=(const U &lhs, const checked_integer<T> &rhs);

// Returns the absolute value of the object.
template <class T>
checked_integer<T> abs(const checked_integer<T> &arg);

typedef checked_integer<char> checked_char;
typedef checked_integer<signed char> checked_schar;
typedef checked_integer<unsigned char> checked_uchar;
typedef checked_integer<short> checked_short;
typedef checked_integer<unsigned short> checked_ushort;
typedef checked_integer<int> checked_int;
typedef checked_integer<unsigned> checked_uint;
typedef checked_integer<long> checked_long;
typedef checked_integer<unsigned long> checked_ulong;
typedef checked_integer<long long> checked_llong;
typedef checked_integer<unsigned long long> checked_ullong;

typedef checked_integer<std::size_t> checked_size_t;
typedef checked_integer<std::ptrdiff_t> checked_ptrdiff_t;

typedef checked_integer<std::intmax_t> checked_intmax_t;
typedef checked_integer<std::uintmax_t> checked_uintmax_t;

typedef checked_integer<std::int_fast8_t> checked_int_fast8_t;
typedef checked_integer<std::int_least8_t> checked_int_least8_t;

typedef checked_integer<std::uint_fast8_t> checked_uint_fast8_t;
typedef checked_integer<std::uint_least8_t> checked_uint_least8_t;

typedef checked_integer<std::int_fast16_t> checked_int_fast16_t;
typedef checked_integer<std::int_least16_t> checked_int_least16_t;

typedef checked_integer<std::uint_fast16_t> checked_uint_fast16_t;
typedef checked_integer<std::uint_least16_t> checked_uint_least16_t;

typedef checked_integer<std::int_fast32_t> checked_int_fast32_t;
typedef checked_integer<std::int_least32_t> checked_int_least32_t;

typedef checked_integer<std::uint_fast32_t> checked_uint_fast32_t;
typedef checked_integer<std::uint_least32_t> checked_uint_least32_t;

typedef checked_integer<std::int_fast64_t> checked_int_fast64_t;
typedef checked_integer<std::int_least64_t> checked_int_least64_t;

typedef checked_integer<std::uint_fast64_t> checked_uint_fast64_t;
typedef checked_integer<std::uint_least64_t> checked_uint_least64_t;

#ifdef INT8_MAX
typedef checked_integer<std::int8_t> checked_int8_t;
#endif

#ifdef UINT8_MAX
typedef checked_integer<std::uint8_t> checked_uint8_t;
#endif

#ifdef INT16_MAX
typedef checked_integer<std::int16_t> checked_int16_t;
#endif

#ifdef UINT16_MAX
typedef checked_integer<std::uint16_t> checked_uint16_t;
#endif

#ifdef INT32_MAX
typedef checked_integer<std::int32_t> checked_int32_t;
#endif

#ifdef UINT32_MAX
typedef checked_integer<std::uint32_t> checked_uint32_t;
#endif

#ifdef INT64_MAX
typedef checked_integer<std::int64_t> checked_int64_t;
#endif

#ifdef UINT64_MAX
typedef checked_integer<std::uint64_t> checked_uint64_t;
#endif

#ifdef INTPTR_MAX
typedef checked_integer<std::intptr_t> checked_intptr_t;
#endif

#ifdef UINTPTR_MAX
typedef checked_integer<std::uintptr_t> checked_uintptr_t;
#endif

} // namespace zimg


// Implementation details follow.
#include <stdexcept>

namespace zimg {

template <class T, bool = std::numeric_limits<T>::is_signed>
struct _checked_arithmetic;

// Unsigned.
template <class T>
struct _checked_arithmetic<T, false> {
#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable : 4146)
#endif
	static T neg(const T &value)
	{
		return -value;
	}
#ifdef _MSC_VER
  #pragma warning(pop)
#endif

	static void add(T &lhs, const T &rhs)
	{
		if (rhs > std::numeric_limits<T>::max() - lhs)
			throw std::overflow_error("overflow_error");
		lhs += rhs;
	}

	static void sub(T &lhs, const T &rhs)
	{
		if (rhs > lhs)
			throw std::overflow_error("overflow_error");
		lhs -= rhs;
	}

	static void mul(T &lhs, const T &rhs)
	{
		if (lhs && rhs > std::numeric_limits<T>::max() / lhs)
			throw std::overflow_error("overflow_error");
		lhs *= rhs;
	}

	static void div(T &lhs, const T &rhs)
	{
		if (rhs == T())
			throw std::overflow_error("overflow_error");
		lhs /= rhs;
	}

	static void mod(T &lhs, const T &rhs)
	{
		if (rhs == T())
			throw std::overflow_error("overflow_error");
		lhs %= rhs;
	}

	static void lshift(T &value, unsigned n)
	{
		if (n >= std::numeric_limits<T>::digits || (value > (std::numeric_limits<T>::max() >> n)))
			throw std::overflow_error("overflow_error");
		value <<= n;
	}

	static void rshift(T &value, unsigned n)
	{
		if (n >= std::numeric_limits<T>::digits)
			throw std::overflow_error("overflow_error");
		value >>= n;
	}
};

// Signed.
template <class T>
struct _checked_arithmetic<T, true> {
	static T neg(const T &value)
	{
		if (value == std::numeric_limits<T>::min())
			throw std::overflow_error("overflow_error");
		return -value;
	}

	static void add(T &lhs, const T &rhs)
	{
		if ((rhs > T() && (lhs > std::numeric_limits<T>::max() - rhs)) ||
		    (rhs < T() && (lhs < std::numeric_limits<T>::min() - rhs)))
		{
			throw std::overflow_error("overflow_error");
		}

		lhs += rhs;
	}

	static void sub(T &lhs, const T &rhs)
	{
		if ((rhs > T() && (lhs < std::numeric_limits<T>::min() + rhs)) ||
		    (rhs < T() && (lhs > std::numeric_limits<T>::max() + rhs)))
		{
			throw std::overflow_error("overflow_error");
		}

		lhs -= rhs;
	}

	static void mul(T &lhs, const T &rhs)
	{
		if ((lhs > 0 && rhs > 0 && lhs > std::numeric_limits<T>::max() / rhs) ||
		    (lhs > 0 && rhs < 0 && rhs < std::numeric_limits<T>::min() / lhs) ||
		    (lhs < 0 && rhs > 0 && lhs < std::numeric_limits<T>::min() / rhs) ||
		    (lhs < 0 && rhs < 0 && rhs < std::numeric_limits<T>::max() / lhs))
		{
			throw std::overflow_error("overflow_error");
		}

		lhs *= rhs;
	}

	static void div(T &lhs, const T &rhs)
	{
		if (rhs == T() || (lhs == std::numeric_limits<T>::min() && rhs == -1))
			throw std::overflow_error("overflow_error");

		lhs /= rhs;
	}

	static void mod(T &lhs, const T &rhs)
	{
		if (rhs == T() || (lhs == std::numeric_limits<T>::min() && rhs == -1))
			throw std::overflow_error("overflow_error");

		lhs %= rhs;
	}

	static void lshift(T &value, unsigned n)
	{
		if (n >= std::numeric_limits<T>::digits || value < T() || (value > (std::numeric_limits<T>::max() >> n)))
			throw std::overflow_error("overflow_error");

		value <<= n;
	}

	static void rshift(T &value, unsigned n)
	{
		if (n >= std::numeric_limits<T>::digits || value < T())
			throw std::overflow_error("overflow_error");

		value >>= n;
	}
};


// T is unsigned, U is unsigned.
template <class T, class U>
T _checked_integer_cast(const U &value, std::false_type, std::false_type)
{
	typedef typename std::common_type<T, U>::type common_type;

	// coverity[result_independent_of_operands]
	if (common_type(value) > common_type(std::numeric_limits<T>::max()))
		throw std::overflow_error("overflow_error");
	return static_cast<T>(value);
}

// T is unsigned, U is signed.
template <class T, class U>
T _checked_integer_cast(const U &value, std::false_type, std::true_type)
{
	typedef typename std::common_type<T, U>::type common_type;

	// coverity[result_independent_of_operands]
	if (value < U() || common_type(value) > common_type(std::numeric_limits<T>::max()))
		throw std::overflow_error("overflow_error");
	return static_cast<T>(value);
}

// T is signed, U is unsigned.
template <class T, class U>
T _checked_integer_cast(const U &value, std::true_type, std::false_type)
{
	typedef typename std::common_type<T, U>::type common_type;

	// coverity[result_independent_of_operands]
	if (common_type(value) > common_type(std::numeric_limits<T>::max()))
		throw std::overflow_error("overflow_error");
	return static_cast<T>(value);
}

// T is signed, U is signed.
template <class T, class U>
T _checked_integer_cast(const U &value, std::true_type, std::true_type)
{
	typedef typename std::common_type<T, U>::type common_type;

	// coverity[result_independent_of_operands]
	if (common_type(value) < common_type(std::numeric_limits<T>::min()) ||
	    common_type(value) > common_type(std::numeric_limits<T>::max()))
	{
		throw std::overflow_error("overflow_error");
	}
	return static_cast<T>(value);
}

template <class T, class U>
T _checked_integer_cast(const U &value)
{
	return _checked_integer_cast<T>(
		value,
		std::integral_constant<bool, std::numeric_limits<T>::is_signed>(),
		std::integral_constant<bool, std::numeric_limits<U>::is_signed>());
}

// ztd::checked_integer::checked_integer
//
// Constructs the ztd::checked_integer object.
// (1) Constructs the object with the given value.
// (2) Constructs the object with the given value, checking for overflow.
//
// Parameters
// value - value to initialize with
//
// Exceptions
// std::overflow_error if value is not representable.

// (1)
template <class T>
constexpr checked_integer<T>::checked_integer(const T &value) noexcept : m_value(value) {}

// (2)
template <class T>
template <class U, class>
checked_integer<T>::checked_integer(const U &value) : m_value(_checked_integer_cast<T>(value)) {}

// ztd::checked_integer::operator=
//
// Assigns a new value to the contents.
// (1) Assigns value to the object.
// (2) Assigns value to the object, checking for overflow.
//
// Parameters
// value - value to assign
//
// Exceptions
// std::overflow_error if value is not representable.

// (1)
template <class T>
checked_integer<T> &checked_integer<T>::operator=(const T &value) noexcept
{
	m_value = value;
	return *this;
}

// (2)
template <class T>
template <class U>
_enable_if_convertible_t<U, T, checked_integer<T>> &checked_integer<T>::operator=(const U &value)
{
	m_value = _checked_integer_cast<T>(value);
	return *this;
}

// ztd::checked_integer::get
//
// Returns the stored value.
//
// Return value
// The stored value.
template <class T>
constexpr T checked_integer<T>::get() const noexcept { return m_value; }

// ztd::checked_integer::operator bool
//
// Checks whether *this is zero.
//
// Return value
// true if *this is non-zero, false otherwise.
template <class T>
constexpr checked_integer<T>::operator bool() const noexcept { return !!m_value; }

// ztd::checked_integer::operator++,--
//
// Increments or decremnets the object.
// (1 - 2) Pre-increments or pre-decrements the object by one respectively.
// (3 - 4) Post-increments or post-decrements the object by one respectively.
//
// Return value
// (1 - 2) *this
// (3 - 4) a copy of *this that was made before the change
//
// Exceptions
// std::overflow_error if operation would overflow

// (1)
template <class T>
checked_integer<T> &checked_integer<T>::operator++()
{
	if (m_value == std::numeric_limits<T>::max())
		throw std::overflow_error("overflow_error");
	++m_value;
	return *this;
}

// (2)
template <class T>
checked_integer<T> &checked_integer<T>::operator--()
{
	if (m_value == std::numeric_limits<T>::min())
		throw std::overflow_error("overflow_error");
	--m_value;
	return *this;
}

// (3)
template <class T>
checked_integer<T> checked_integer<T>::operator++(int)
{
	checked_integer ret = *this;
	++*this;
	return ret;
}

// (4)
template <class T>
checked_integer<T> checked_integer<T>::operator--(int)
{
	checked_integer ret = *this;
	--*this;
	return ret;
}

// ztd::checked_integer::operator+=,-=,*=,/=,%=,<<=,>>=,&=,|=,^=
//
// Implements the compound assignment operators.
// (1) Adds other to *this.
// (2) Subtracts other from *this.
// (3) Multiplies *this by other.
// (4) Divides *this by other.
// (5) Modulos *this by other.
// (6) Left-shifts *this by n.
// (7) Right-shifts *this by n.
// (8) Bitwise-ANDs *this with other.
// (9) Bitwise-ORs *this with other.
// (10) Bitwise-XORs *this with other.
//
// Parameters
// other - right-hand side argument to operator
//
// Return value
// *this
//
// Exceptions
// std::overflow_error if result would overflow

// (1)
template <class T>
checked_integer<T> &checked_integer<T>::operator+=(const checked_integer<T> &rhs)
{
	_checked_arithmetic<T>::add(m_value, rhs.m_value);
	return *this;
}

// (2)
template <class T>
checked_integer<T> &checked_integer<T>::operator-=(const checked_integer<T> &rhs)
{
	_checked_arithmetic<T>::sub(m_value, rhs.m_value);
	return *this;
}

// (3)
template <class T>
checked_integer<T> &checked_integer<T>::operator*=(const checked_integer<T> &rhs)
{
	_checked_arithmetic<T>::mul(m_value, rhs.m_value);
	return *this;
}

// (4)
template <class T>
checked_integer<T> &checked_integer<T>::operator/=(const checked_integer<T> &rhs)
{
	_checked_arithmetic<T>::div(m_value, rhs.m_value);
	return *this;
}

// (5)
template <class T>
checked_integer<T> &checked_integer<T>::operator%=(const checked_integer<T> &rhs)
{
	_checked_arithmetic<T>::mod(m_value, rhs.m_value);
	return *this;
}

// (6)
template <class T>
checked_integer<T> &checked_integer<T>::operator<<=(unsigned n)
{
	_checked_arithmetic<T>::lshift(m_value, n);
	return *this;
}

// (7)
template <class T>
checked_integer<T> &checked_integer<T>::operator>>=(unsigned n)
{
	_checked_arithmetic<T>::rshift(m_value, n);
	return *this;
}

// (8)
template <class T>
checked_integer<T> &checked_integer<T>::operator&=(const checked_integer<T> &rhs)
{
	m_value &= rhs.m_value;
	return *this;
}

// (9)
template <class T>
checked_integer<T> &checked_integer<T>::operator|=(const checked_integer<T> &rhs)
{
	m_value |= rhs.m_value;
	return *this;
}

// (10)
template <class T>
checked_integer<T> &checked_integer<T>::operator^=(const checked_integer<T> &rhs)
{
	m_value ^= rhs.m_value;
	return *this;
}

// operator+(unary), operator-(unary), operator~ (ztd::checked_integer)
//
// Implements the unary arithmetic operators.
// (1) Returns the value of the argument.
// (2) Negates tbe argument.
// (3) Returns the bitwise-NOT of the argument.
//
// Exceptions:
// (2) std::overflow_error if result of negation would overflow.

// (1)
template <class T>
checked_integer<T> operator+(const checked_integer<T> &value)
{
	return value;
}

// (2) Negates tbe argument.
template <class T>
checked_integer<T> operator-(const checked_integer<T> &value)
{
	return _checked_arithmetic<T>::neg(value.get());
}

// (3) Returns the bitwise-NOT of the argument.
template <class T>
checked_integer<T> operator~(const checked_integer<T> &value)
{
	return ~value.get();
}

// operator+,-,*,/,% (ztd::checked_integer)
//
// Implements the binary arithmetic operators.
// (1 - 3) Returns the sum of the arguments.
// (4 - 6) Returns the result of subtracting rhs from lhs.
// (7 - 9) Multiplies its arguments.
// (10 - 12) Divides lhs by rhs.
// (13 - 15) Returns lhs modulo rhs.
//
// Parameters
// lhs, rhs - the arguments
//
// Return value
// (1 - 3) checked_integer(lhs) += checked_integer(rhs)
// (4 - 6) checked_integer(lhs) -= checked_integer(rhs)
// (7 - 9) checked_integer(lhs) *= checked_integer(rhs)
// (10 - 12) checked_integer(lhs) /= checked_integer(rhs)
// (13 - 15) checked_integer(lhs) %= checked_integer(rhs)
//
// Exceptions
// std::overflow_error if result would overflow.

// (1)
template <class T>
checked_integer<T> operator+(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) += rhs;
}

// (2)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator+(const checked_integer<T> &lhs, const U &rhs)
{
	return checked_integer<T>(lhs) += rhs;
}

// (3)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator+(const U &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) += rhs;
}

// (4)
template <class T>
checked_integer<T> operator-(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) -= rhs;
}

// (5)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator-(const checked_integer<T> &lhs, const U &rhs)
{
	return checked_integer<T>(lhs) -= rhs;
}

// (6)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator-(const U &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) -= rhs;
}

// (7)
template <class T>
checked_integer<T> operator*(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) *= rhs;
}

// (8)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator*(const checked_integer<T> &lhs, const U &rhs)
{
	return checked_integer<T>(lhs) *= rhs;
}

// (9)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator*(const U &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) *= rhs;
}

// (10)
template <class T>
checked_integer<T> operator/(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) /= rhs;
}

// (11)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator/(const checked_integer<T> &lhs, const U &rhs)
{
	return checked_integer<T>(lhs) /= rhs;
}

// (12)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator/(const U &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) /= rhs;
}

// (13)
template <class T>
checked_integer<T> operator%(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) %= rhs;
}

// (14)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator%(const checked_integer<T> &lhs, const U &rhs)
{
	return checked_integer<T>(lhs) %= rhs;
}

// (15)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator%(const U &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) %= rhs;
}

// operator<<,>> (ztd::checked_integer)
//
// Implements the bit-shift operators.
//
// (1) Left-shifts value by n.
// (2) Right-shifts value by n.
//
// Parameters
// value - value to shift
//     n - shift count
//
// Return value
// (1) checked_integer<T>(value) <<= n
// (2) checked_integer<T>(value) >>= n
//
// Exceptions
// std::overflow_error if value is negative, if n is out of range, or if the result would overflow.

// (1)
template <class T>
checked_integer<T> operator<<(const checked_integer<T> &value, unsigned n)
{
	return checked_integer<T>(value) <<= n;
}

// (2)
template <class T>
checked_integer<T> operator>>(const checked_integer<T> &value, unsigned n)
{
	return checked_integer<T>(value) >>= n;
}

// operator&,|,^ (ztd::checked_integer)
//
// Implements the binary bitwise operators.
// (1 - 3) Bitwise-ANDs lhs with rhs.
// (4 - 6) Bitwise-ORs lhs with rhs.
// (7 - 9) Bitwise-XORs lhs with rhs.
//
// Parameters
// lhs, rhs - the arguments
//
// Return value
// (1 - 3) checked_integer<T>(lhs) &= rhs
// (4 - 6) checked_integer<T>(lhs) |= rhs
// (7 - 9) checked_integer<T>(lhs) ^= rhs
//
// Exceptions
// None.

// (1)
template <class T>
checked_integer<T> operator&(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) &= rhs;
}

// (2)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator&(const checked_integer<T> &lhs, const U &rhs)
{
	return checked_integer<T>(lhs) &= rhs;
}

// (3)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator&(const U &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) &= rhs;
}

// (4)
template <class T>
checked_integer<T> operator|(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) |= rhs;
}

// (5)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator|(const checked_integer<T> &lhs, const U &rhs)
{
	return checked_integer<T>(lhs) |= rhs;
}

// (6)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator|(const U &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) |= rhs;
}

// (7)
template <class T>
checked_integer<T> operator^(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) ^= rhs;
}

// (8)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator^(const checked_integer<T> &lhs, const U &rhs)
{
	return checked_integer<T>(lhs) ^= rhs;
}

// (9)
template <class T, class U>
_enable_if_convertible_t<U, T, checked_integer<T>> operator^(const U &lhs, const checked_integer<T> &rhs)
{
	return checked_integer<T>(lhs) ^= rhs;
}

// operator==,!=,<,<=,>,>= (ztd::checked_integer)
//
// Compares two objects.
// (1 - 3) Compares lhs and rhs for equality.
// (4 - 6) Compares lhs and rhs for inequality.
// (5 - 9) Compares lhs and rhs with less-than operator.
// (10 - 12) Compares lhs and rhs with less-than-or-equals operator.
// (13 - 15) Compares lhs and rhs with greater-than operator.
// (16 - 18) Compares lhs and rhs with greater-than-or-equals operator.
//
// Parameters
// lhs, rhs - the arguments to compare
//
// Return value
// true if comparison is true, else false

// (1)
template <class T>
bool operator==(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return lhs.get() == rhs.get();
}

// (2)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator==(const checked_integer<T> &lhs, const U &rhs)
{
	return lhs.get() == rhs;
}

// (3)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator==(const U &lhs, const checked_integer<T> &rhs)
{
	return lhs == rhs.get();
}

// (4)
template <class T>
bool operator!=(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return lhs.get() != rhs.get();
}

// (5)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator!=(const checked_integer<T> &lhs, const U &rhs)
{
	return lhs.get() != rhs;
}

// (6)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator!=(const U &lhs, const checked_integer<T> &rhs)
{
	return lhs != rhs.get();
}

// (7)
template <class T>
bool operator<(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return lhs.get() < rhs.get();
}

// (8)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator<(const checked_integer<T> &lhs, const U &rhs)
{
	return lhs.get() < rhs;
}

// (9)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator<(const U &lhs, const checked_integer<T> &rhs)
{
	return lhs < rhs.get();
}

// (10)
template <class T>
bool operator<=(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return lhs.get() <= rhs.get();
}

// (11)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator<=(const checked_integer<T> &lhs, const U &rhs)
{
	return lhs.get() <= rhs;
}

// (12)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator<=(const U &lhs, const checked_integer<T> &rhs)
{
	return lhs <= rhs.get();
}

// (13)
template <class T>
bool operator>(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return lhs.get() > rhs.get();
}

// (14)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator>(const checked_integer<T> &lhs, const U &rhs)
{
	return lhs.get() > rhs;
}

// (15)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator>(const U &lhs, const checked_integer<T> &rhs)
{
	return lhs > rhs.get();
}

// (16)
template <class T>
bool operator>=(const checked_integer<T> &lhs, const checked_integer<T> &rhs)
{
	return lhs.get() >= rhs.get();
}

// (17)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator>=(const checked_integer<T> &lhs, const U &rhs)
{
	return lhs.get() >= rhs;
}

// (18)
template <class T, class U>
_enable_if_convertible_t<U, T, bool> operator>=(const U &lhs, const checked_integer<T> &rhs)
{
	return lhs >= rhs.get();
}

// ztd::abs(ztd::checked_integer)
//
// Returns the absolute value of the object.
//
// Parameters
// arg - argument
//
// Return value
// Absolute value of argument.
//
// Exceptions
// std::overflow_error if result would overflow.
template <class T>
checked_integer<T> abs(const checked_integer<T> &arg)
{
	return arg < checked_integer<T>() ? -arg : arg;
}

} // namespace zimg

#endif // ZIMG_CHECKED_INT_H_
