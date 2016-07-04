#include <limits>

namespace zimg {
namespace depth {

#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable : 4146)
  #pragma warning(disable : 4244)
#endif

// Credits to: Ruslan
// http://stackoverflow.com/questions/32829512/c-hexfloat-compile-time-parsing/35968294#35968294
class HEX_LF_C
{
    using size_t=decltype(sizeof(0)); // avoid including extra headers
    static constexpr const long double _0x1p256=1.15792089237316195424e77L; // 2^256
    struct BadDigit{};
    // Unportable, but will work for ANSI charset
    static constexpr int hexDigit(char c)
    {
        return '0'<=c&&c<='9' ? c-'0' :
               'a'<=c&&c<='f' ? c-'a'+0xa :
               'A'<=c&&c<='F' ? c-'A'+0xA : throw BadDigit{};
    }
    // lightweight constexpr analogue of std::strtoull
    template<typename Int>
    static constexpr Int getNumber(const char* array,
                                   int base,
                                   size_t begin,
                                   size_t end,
                                   Int accumulated=Int(0))
    {
        return begin==end ? accumulated :
               array[begin]=='-' ? -getNumber<Int>(array,base,begin+1,end) :
               array[begin]=='+' ? +getNumber<Int>(array,base,begin+1,end) :
               getNumber<Int>(array,base,begin+1,end,
                                    accumulated*base+hexDigit(array[begin]));
    }
    // lightweight constexpr version of std::scalbn
    static constexpr long double scalbn(long double value, int exponent)
    {
        // Trying hard to avoid hitting compiler recursion limit
        return exponent==0 ? value : exponent>0 ?
            (exponent>+255 ? scalbn(value*_0x1p256,exponent-256) : scalbn(value*2,exponent-1)) :
            (exponent<-255 ? scalbn(value/_0x1p256,exponent+256) : scalbn(value/2,exponent+1));
    }
    // constexpr version of std::strlen
    static constexpr size_t strlen(const char* array)
    { return *array ? 1+strlen(array+1) : 0; }
    static constexpr size_t findChar(const char* array,
                                     char charToFind,
                                     size_t begin,
                                     size_t end)
    {
        return begin==end ? end :
               array[begin]==charToFind ? begin :
               findChar(array,charToFind,begin+1,end);
    }
    static constexpr size_t mantissaEnd(const char* str)
    { return findChar(str,'p',0,strlen(str)); }

    static constexpr size_t pointPos(const char* str)
    { return findChar(str,'.',0,mantissaEnd(str)); }

    static constexpr int exponent(const char* str)
    {
        return mantissaEnd(str)==strlen(str) ? 0 :
                getNumber<int>(str,10,mantissaEnd(str)+1,strlen(str));
    }
    static constexpr bool isSign(char ch) { return ch=='+'||ch=='-'; }
    static constexpr size_t mantissaBegin(const char* str)
    {
        return isSign(*str)+
               2*(str[isSign(*str)]=='0' && str[isSign(*str)+1]=='x');
    }
    static constexpr unsigned long long beforePoint(const char* str)
    {
        return getNumber<unsigned long long>(str,
                                             16,
                                             mantissaBegin(str),
                                             pointPos(str));
    }
    static constexpr long double addDigits(const char* str,
                                           size_t begin,
                                           size_t end,
                                           long double currentValue,
                                           long double currentFactor)
    {
        return begin==end ? currentValue :
               addDigits(str,begin+1,end,
                         currentValue+currentFactor*hexDigit(str[begin]),
                         currentFactor/16);
    }
    // If you don't need to force compile-time evaluation, you can use this
    // directly (having made it public)
    template<size_t N>
    static constexpr long double get(const char (&str)[N])
    {
        return (str[0]=='-' ? -1 : 1)*
            addDigits(str,pointPos(str)+1,mantissaEnd(str),
                      scalbn(beforePoint(str),exponent(str)),
                      scalbn(1.L/16,exponent(str)));
    }
    struct UnsupportedLiteralLength{};
public:
    // This helps to convert string literal to a valid template parameter
    // It just packs the given chunk (8 chars) of the string into a ulonglong.
    // We rely here and in LF_Evaluator on the fact that 32 chars is enough
    // for any useful long double hex literal (on x87 arch).
    // Will need tweaking if support for wider long double types is required.
    template<size_t N>
    static constexpr unsigned long long string_in_ull(const char (&array)[N],
                                                      size_t start,
                                                      size_t end,
                                                      size_t numIndex)
    {
        // relying on CHAR_BIT==8 here
        return N>32 ? throw UnsupportedLiteralLength{} :
               start==end || start>=N ? 0 :
               string_in_ull(array,start+1,end,numIndex) |
                    ((array[start]&0xffull)<<(8*(start-numIndex)));
    }
    // This is to force compile-time evaluation of the hex constant
    template<unsigned long long A,
             unsigned long long B,
             unsigned long long C,
             unsigned long long D>
    struct LF_Evaluator
    {
        static constexpr char ch(unsigned long long X,
                                 int charIndex) { return X>>charIndex*8; }
        static constexpr const char string[32]={
            ch(A,0),ch(A,1),ch(A,2),ch(A,3),ch(A,4),ch(A,5),ch(A,6),ch(A,7),
            ch(B,0),ch(B,1),ch(B,2),ch(B,3),ch(B,4),ch(B,5),ch(B,6),ch(B,7),
            ch(C,0),ch(C,1),ch(C,2),ch(C,3),ch(C,4),ch(C,5),ch(C,6),ch(C,7),
            ch(D,0),ch(D,1),ch(D,2),ch(D,3),ch(D,4),ch(D,5),ch(D,6),ch(D,7)
            };
        static constexpr long double value=get(string);
    };
};

#define HEX_LF_C(num) HEX_LF_C::LF_Evaluator<                    \
                        HEX_LF_C::string_in_ull(#num,0,8,0),     \
                        HEX_LF_C::string_in_ull(#num,8,16,8),    \
                        HEX_LF_C::string_in_ull(#num,16,24,16),  \
                        HEX_LF_C::string_in_ull(#num,24,32,24)>::value

#ifdef _MSC_VER
  #pragma warning(pop)
#endif

} // namespace depth
} // namespace zimg
