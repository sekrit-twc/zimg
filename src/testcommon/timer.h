#pragma once

#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>
#include <cmath>
#include <utility>

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


template <class T, class U>
std::pair<double, double> measure_benchmark(unsigned times, T func, U callback)
{
	Timer timer;
	double min_time = INFINITY;
	double sum_time = 0.0;

	for (unsigned n = 0; n < times; ++n) {
		double elapsed_cur;

		timer.start();
		func();
		timer.stop();

		elapsed_cur = timer.elapsed();
		callback(n, elapsed_cur);

		sum_time += elapsed_cur;
		min_time = min_time < elapsed_cur ? min_time : elapsed_cur;
	}
	return{ sum_time / times, min_time };
}

template <class T>
std::pair<double, double> measure_benchmark(unsigned times, T func)
{
	return measure_benchmark(times, func, [](unsigned, double) {});
}

#endif // TIMER_H_
