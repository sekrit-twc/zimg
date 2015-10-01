#ifdef _MSC_VER
  #include <float.h>
#endif

extern double _mysin(double x);
extern double _mycos(double y);
extern double _mypow(double x, double y);
extern float _mypowf(float x, float y);

#if defined(_MSC_VER) && !defined(_M_X64)
  #define fpu_set_single() _control87(_PC_24, MCW_PC)
  #define fpu_set_double() _control87(_PC_53, _MCW_PC)
  #define fpu_restore(x) _control87((x), _MCW_PC);
#else
  #define fpu_set_single() 0
  #define fpu_set_double() 0
  #define fpu_restore(x)
#endif /* _MSC_VER */

double mysin(double x)
{
	unsigned state = fpu_set_double();
	double y = _mysin(x);
	fpu_restore(state);
	return y;
}

double mycos(double x)
{
	unsigned state = fpu_set_double();
	double y = _mycos(x);
	fpu_restore(state);
	return y;
}

double mypow(double x, double y)
{
	unsigned state = fpu_set_double();
	double z = _mypow(x, y);
	fpu_restore(state);
	return z;
}

float mypowf(float x, float y)
{
	unsigned state = fpu_set_single();
	float z = _mypowf(x, y);
	fpu_restore(state);
	return z;
}
