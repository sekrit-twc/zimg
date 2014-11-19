#pragma once

#ifndef TIMER_H_
#define TIMER_H_

#ifndef _WIN32

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

#else

#include <windows.h>

class Timer {
	LARGE_INTEGER m_start;
	LARGE_INTEGER m_stop;
	LARGE_INTEGER m_frequency;
public:
	Timer() : m_start{}, m_stop{}, m_frequency{} { QueryPerformanceFrequency(&m_frequency); }

	void start() { QueryPerformanceCounter(&m_start); }

	void stop() { QueryPerformanceCounter(&m_stop); }

	double elapsed()
	{
		return (double)(m_stop.QuadPart - m_start.QuadPart) / (double)m_frequency.QuadPart;
	}
};

#endif // _WIN32

#endif // TIMER_H_
