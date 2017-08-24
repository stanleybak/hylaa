// Stanley Bak
// GPU Utility Class
// Provides an interface for profiling, gpu status, memory remaining, ect.

#include <sys/time.h>
#include <sys/sysinfo.h>
#include <stdlib.h>
#include <stdarg.h>
#include <vector>
#include <map>
#include <string>

using namespace std;

#ifndef HYLAA_GPU_UTILS_H
#define HYLAA_GPU_UTILS_H

// print an error and then exit
void error(const char* format, ...)
{
    va_list args;
    fprintf(stdout, "Fatal Error: ");
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
    fprintf(stdout, "\n");

    exit(1);
}

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
        totalOps = 0;
    }

    void tic(const char* clockName)
    {
        name = clockName;

        if (started)
            error("Cpu Timer started twice: '%s'\n", name);

        started = true;
        ++count;
        startUs = now();
    }

    void toc() { toc(0); }

    void toc(unsigned long ops)
    {
        if (!started)
            error("Cpu Timer stopped without being started: '%s'\n", name);

        started = false;
        long endUs = now();

        totalUs += endUs - startUs;
        totalOps += ops;
    }

    float getTotalMs()
    {
        if (started)
            error("Cpu Timer getTotalMs() called but timer was never stopped: '%s'\n", name);

        return totalUs / 1000.0;
    }

    int getCount()
    {
        if (started)
            error("Cpu Timer getCount() called but timer was never stopped: '%s'\n", name);

        return count;
    }

    unsigned long getOps()
    {
        if (started)
            error("Cpu Timer getOps() called but timer was never stopped: '%s'\n", name);

        return totalOps;
    }

   private:
    const char* name;
    bool started;
    int count;
    unsigned long totalOps;
    long totalUs;
    long startUs;

    // get the current time using the cpu / os timer in microseconds
    long now()
    {
        struct timeval nowUs;

        if (gettimeofday(&nowUs, 0))
        {
            perror("gettimeofday");
            error("gettimeofday failed");
        }

        return 1000000l * nowUs.tv_sec + nowUs.tv_usec;
    }
};

class GpuTimingData
{
   public:
    GpuTimingData()
    {
        name = "(unnamed)";
        totalOps = 0;
    }

    ~GpuTimingData() { clearEvents(); }

    void tic(const char* clockName)
    {
        name = clockName;

        if (startEvents.size() != stopEvents.size())
            error("Gpu Timer started twice: '%s'\n", name);

        startEvents.push_back(recordEvent());
    }

    void toc() { toc(0); }

    void toc(unsigned long ops)
    {
        if (startEvents.size() != 1 + stopEvents.size())
            error("Gpu Timer stopped without being started: '%s'\n", name);

        totalOps += ops;
        stopEvents.push_back(recordEvent());
    }

    float getTotalMs()
    {
        if (startEvents.size() != stopEvents.size())
            error("Gpu Timer getTotalUs() called but timer is still running: '%s'\n", name);

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
            error("Gpu Timer getCount() called but timer is still running: '%s'\n", name);

        return (int)startEvents.size();
    }

    unsigned long getOps()
    {
        if (startEvents.size() != stopEvents.size())
            error("Gpu Timer getOps() called but timer was never stopped: '%s'\n", name);

        return totalOps;
    }

   private:
    const char* name;
    unsigned long totalOps;

    vector<cudaEvent_t> startEvents;
    vector<cudaEvent_t> stopEvents;

    cudaEvent_t recordEvent()
    {
        cudaEvent_t event;  // cudaEvent_t is a pointer

        cudaEventCreate(&event);
        cudaEventRecord(event);

        return event;
    }

    void clearEvents()
    {
        for (int i = 0; i < (int)startEvents.size(); ++i)
            cudaEventDestroy(startEvents[i]);

        for (int i = 0; i < (int)stopEvents.size(); ++i)
            cudaEventDestroy(stopEvents[i]);

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

    void toc(const char* clockName) { toc(clockName, 0); }

    void toc(const char* clockName, unsigned long ops)
    {
        if (useProfiling)
        {
            if (useGpu)
                gpuTimers[clockName].toc(ops);
            else
                cpuTimers[clockName].toc(ops);
        }
    }

    // print timers results
    void printTimers()
    {
        if (useProfiling)
        {
            printf("Timing Information (%s):\n", useGpu ? "GPU Timing" : "CPU Timing");
            char buf[256];
            multimap<float, string> sorted;

            if (useGpu)
            {
                for (map<string, GpuTimingData>::iterator i = gpuTimers.begin();
                     i != gpuTimers.end(); ++i)
                {
                    const char* name = i->first.c_str();
                    float ms = i->second.getTotalMs();
                    int count = i->second.getCount();
                    unsigned long ops = i->second.getOps();

                    if (ops == 0)
                        snprintf(buf, sizeof(buf), " %s: %.3fms (%d calls)", name, ms, count);
                    else
                    {
                        double gigaFlops = ops / ms / 1000.0 / 1000.0;

                        if (gigaFlops > 1)
                            snprintf(buf, sizeof(buf), " %s: %.3fms (%d calls) (%f GFLOPS)", name,
                                     ms, count, gigaFlops);
                        else
                            snprintf(buf, sizeof(buf), " %s: %.3fms (%d calls) (%f MegaFlops)",
                                     name, ms, count, gigaFlops * 1000);
                    }

                    sorted.insert(make_pair(ms, buf));
                }
            }
            else
            {
                for (map<string, CpuTimingData>::iterator i = cpuTimers.begin();
                     i != cpuTimers.end(); ++i)
                {
                    const char* name = i->first.c_str();
                    float ms = i->second.getTotalMs();
                    int count = i->second.getCount();
                    unsigned long ops = i->second.getOps();

                    if (ops == 0)
                        snprintf(buf, sizeof(buf), " %s: %.3fms (%d calls)", name, ms, count);
                    else
                    {
                        double gigaFlops = ops / ms / 1000.0 / 1000.0;

                        if (gigaFlops > 1)
                            snprintf(buf, sizeof(buf), " %s: %.3fms (%d calls) (%f GFLOPS)", name,
                                     ms, count, gigaFlops);
                        else
                            snprintf(buf, sizeof(buf), " %s: %.3fms (%d calls) (%f MegaFlops)",
                                     name, ms, count, gigaFlops * 1000);
                    }

                    sorted.insert(make_pair(ms, buf));
                }
            }

            // print in descending order
            for (multimap<float, string>::reverse_iterator i = sorted.rbegin(); i != sorted.rend();
                 ++i)
            {
                const char* msg = i->second.c_str();
                printf("%s\n", msg);
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
                error("sysinfo failed");
            }

            rv = info.freeram;
        }
        else  // gpu memory
        {
            size_t free;
            size_t total;
            cudaError_t status = cudaMemGetInfo(&free, &total);

            if (status != cudaSuccess)
                error("cudaMemGetInfo() failed: %s\n", cudaGetErrorString(status));

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

    map<string, CpuTimingData> cpuTimers;
    map<string, GpuTimingData> gpuTimers;
};

#endif
