#ifndef GCOPTER_BENCHMARK_TIMER_HPP
#define GCOPTER_BENCHMARK_TIMER_HPP

#include <chrono>

namespace firi_benchmark
{
    class SteadyTimer
    {
    public:
        SteadyTimer()
        {
            reset();
        }

        void reset()
        {
            start_ = std::chrono::steady_clock::now();
        }

        double elapsedMs() const
        {
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - start_)
                .count();
        }

    private:
        std::chrono::steady_clock::time_point start_;
    };
}

#endif
