#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>

class Timer {
	typedef std::chrono::high_resolution_clock hrclock;

	hrclock::time_point m_start;
	hrclock::time_point m_stop;
public:
	void start() { m_start = hrclock::now(); }

	void stop() { m_stop = hrclock::now(); }

	double elapsed()
	{
		std::chrono::duration<double> secs = m_stop - m_start;
		return secs.count();
	}
};

#endif // TIMER_H_
