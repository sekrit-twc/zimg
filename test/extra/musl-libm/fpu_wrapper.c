#ifdef _MSC_VER
  #include <float.h>
#endif

extern float _myexpf(float x);
extern float _mylogf(float x);
extern float _mylog10f(float x);

extern float _mypowf(float x, float y);

extern double _mysin(double x);
extern double _mycos(double y);

#if defined(_MSC_VER) && defined(_M_IX86)
  #define fpu_save() _control87(0, 0)
  #define fpu_set_single() _control87(_PC_24, _MCW_PC)
  #define fpu_set_double() _control87(_PC_53, _MCW_PC)
  #define fpu_restore(x) _control87((x), _MCW_PC)
#else
  #define fpu_save() 0
  #define fpu_set_single() (void)0
  #define fpu_set_double() (void)0
  #define fpu_restore(x) (void)x
#endif /* _MSC_VER */

float myexpf(float x)
{
	unsigned state = fpu_save();
	float y;
	fpu_set_single();
	y = _myexpf(x);
	fpu_restore(state);
	return y;
}

float mylogf(float x)
{
	unsigned state = fpu_save();
	float y;
	fpu_set_single();
	y = _mylogf(x);
	fpu_restore(state);
	return y;
}

float mylog10f(float x)
{
	unsigned state = fpu_save();
	float y;
	fpu_set_single();
	y = _mylog10f(x);
	fpu_restore(state);
	return y;
}

float mypowf(float x, float y)
{
	unsigned state = fpu_save();
	float z;
	fpu_set_single();
	z = _mypowf(x, y);
	fpu_restore(state);
	return z;
}

double mysin(double x)
{
	unsigned state = fpu_save();
	double y;
	fpu_set_double();
	y = _mysin(x);
	fpu_restore(state);
	return y;
}

double mycos(double x)
{
	unsigned state = fpu_save();
	double y;
	fpu_set_double();
	y = _mycos(x);
	fpu_restore(state);
	return y;
}
