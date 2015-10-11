#ifndef BUILDER_MEMBER
#define BUILDER_MEMBER(type, name) \
  template <class T> \
  auto set_##name(T &&val) -> decltype(*this) { name = std::forward<T>(val); return *this; } \
  type name;
#endif // BUILDER_MEMBER
