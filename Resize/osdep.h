#ifndef OSDEP_H
#define OSDEP_H

#ifdef _MSC_VER 
#define FORCE_INLINE __forceinline
#define RESTRICT __restrict
#else
#define FORCE_INLINE
#define RESTRICT
#endif

#endif // OSDEP_H
