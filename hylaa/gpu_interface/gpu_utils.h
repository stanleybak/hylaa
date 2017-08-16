// Stanley Bak
// GPU Utility Class
// Provides an interface for profiling, gpu status, memory remaining, ect.

#include <sys/time.h>
#include <sys/sysinfo.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <string>

using namespace std;

#ifndef HYLAA_GPU_UTILS_H
#define HYLAA_GPU_UTILS_H

class CpuTimingData
{
   public:
    CpuTimingData()
    {
        name = "(unnamed)";
        started = false;
        count = 0;
        totalUs = 0;
        startUs = 0;
    }

    void tic(const char* clockName)
    {
        name = clockName;

        if (started)
        {
            printf("Fatal Error: Cpu Timer started twice: '%s'\n", name);
            exit(1);
        }

        started = true;
        ++count;
        startUs = now();
    }

    void toc()
    {
        if (!started)
        {
            printf("Fatal Error: Cpu Timer stopped without being started: '%s'\n", name);
            exit(1);
        }

        started = false;
        long endUs = now();

        totalUs += endUs - startUs;
    }

    float getTotalMs()
    {
        if (started)
        {
            printf("Fatal Error: Cpu Timer getTotalMs() called but timer was never stopped: '%s'\n",
                   name);
            exit(1);
        }

        return totalUs / 1000.0;
    }

    int getCount()
    {
        if (started)
        {
            printf("Fatal Error: Cpu Timer getCount() called but timer was never stopped: '%s'\n",
                   name);
            exit(1);
        }

        return count;
    }

   private:
    const char* name;
    bool started;
    int count;
    long totalUs;
    long startUs;

    // get the current time using the cpu / os timer in microseconds
    long now()
    {
        struct timeval nowUs;

        if (gettimeofday(&nowUs, 0))
        {
            error("gettimeofday");
            exit(1);
        }

        return 1000000l * now.tv_sec + now.tv_usec;
    }
};

class GpuTimingData
{
   public:
    GpuTimingData() { name = "(unnamed)"; }

    ~GpuTimingData() { clearEvents(); }

    void tic(const char* clockName)
    {
        name = clockName;

        if (startEvents.size() != stopEvents.size())
        {
            printf("Fatal Error: Gpu Timer started twice: '%s'\n", name);
            exit(1);
        }

        startEvents.push_back(recordEvent());
    }

    void toc()
    {
        if (startEvents.size() != 1 + stopEvents.size())
        {
            printf("Fatal Error: Gpu Timer stopped without being started: '%s'\n", name);
            exit(1);
        }

        stopEvents.push_back(recordEvent());
    }

    float getTotalMs()
    {
        if (startEvents.size() != stopEvents.size())
        {
            printf("Fatal Error: Gpu Timer getTotalUs() called but timer is still running: '%s'\n",
                   name);
            exit(1);
        }

        int numEvents = (int)startEvents.size();
        float totalMs = 0;

        for (int i = 0; i < numEvents; ++i)
        {
            cudaEventSynchronize(startEvents[i]);
            cudaEventSynchronize(stopEvents[i]);

            float eventMs = 0;
            cudaEventElapsedTime(&eventMs, startEvents[i], stopEvents[i]);
            totalMs += eventMs;
        }

        return totalMs;
    }

    int getCount()
    {
        if (startEvents.size() != stopEvents.size())
        {
            printf("Fatal Error: Gpu Timer getCount() called but timer is still running: '%s'\n",
                   name);
            exit(1);
        }

        return (int)startEvents.size();
    }

   private:
    const char* name;

    vector<cudaEvent_t> startEvents;
    vector<cudaEvent_t> stopEvents;

    cudaEvent_t recordEvent()
    {
        cudaEvent_t event;  // cudaEvent_t is a pointer

        cudaEventCreate(&event);
        cudaEventRecord(event);

        return cudaEvent_t;
    }

    void clearEvents()
    {
        for (int i = 0; i < (int)startEvents.size(); ++i)
            cudeEventDestroy(startEvents[i]);

        for (int i = 0; i < (int)stopEvents.size(); ++i)
            cudeEventDestroy(stopEvents[i]);

        startEvents.clear();
        stopEvents.clear();
    }
};

class GpuUtil
{
   public:
    GpuUtil(bool forceCpuTiming)
    {
        if (forceCpuTiming)
            useGpu = false;
        else
            useGpu = hasGpu();

        useProfiling = false;
    }

    void setUseProfiling(bool enabled) { useProfiling = enabled; }

    void tic(const char* clockName)
    {
        if (useProfiling)
        {
            if (useGpu)
                gpuTimers[clockName].tic(clockName);
            else
                cpuTimers[clockName].tic(clockName);
        }
    }

    void toc(const char* clockName)
    {
        if (useProfiling)
        {
            if (useGpu)
                gpuTimers[clockName].toc();
            else
                cpuTimers[clockName].toc();
        }
    }

    // get timing data for a specific timer
    float getTimerMs(const char* name)
    {
        float rv = -1;

        if (useProfiling)
        {
            if (useGpu)
                rv = gpuTimers[name].getTotalMs();
            else
                rv = cpuTimers[name].getTotalMs();
        }

        return rv;
    }

    // print timers results
    void printTimers()
    {
        if (useProfiling)
        {
            printf("Timing Information (%s):\n", useGpu ? "GPU Timing" : "CPU Timing");

            if (useGpu)
            {
                for (map<string, GpuTimerData>::iterator i = gpuTimers.begin();
                     i != gpuTimers.end(); ++i)
                {
                    const char* name = i->first.c_str();
                    int count = i->second.getCount();
                    float ms = i->second.getTotalMs();

                    printf(" %s: %.3fms (%d calls)".format(name, ms, count);
                }
            }
            else
            {
                for (map<string, CpuTimerData>::iterator i = cpuTimers.begin();
                     i != cpuTimers.end(); ++i)
                {
                    const char* name = i->first.c_str();
                    int count = i->second.getCount();
                    float ms = i->second.getTotalMs();

                    printf(" %s: %.3fms (%d calls)".format(name, ms, count);
                }
            }
        }
    }

    void clearTimers()
    {
        gpuTimers.clear();
        cpuTimers.clear();
    }

    // gets the number of bytes of available (cpu or gpu) memory
    unsigned long getFreeMemory()
    {
        unsigned long rv = 0;

        if (!useGpu)  // cpu memory
        {
            struct sysinfo info;

            if (sysinfo(&info) != 0)
            {
                perror("sysinfo");
                exit(1);
            }

            rv = info.freeram;
        }
        else  // gpu memory
        {
            size_t free;
            size_t total;
            cudaError_t status = cudaMemGetInfo(&free, &total);

            if (cudaSuccess != cuda_status)
            {
                printf("cudaMemGetInfo() error: %s\n", cudaGetErrorString(cuda_status));
                exit(1);
            }

            rv = free;
        }

        return rv;
    }

    bool hasGpu()
    {
        int num = 0;
        cudaGetDeviceCount(&num);

        return num > 0;
    }

   private:
    bool useProfiling;
    bool useGpu;
    long lastTicUs = 0;  // for tic() and toc()

    map<string, CpuTimingData> cpuTimers;
    map<string, GpuTimerData> gpuTimers;
};

#endif
