// Stanley Bak
// C++-based diagonal matrix multiplication

#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>
#include "ctpl_stl.h"

using namespace std;

ctpl::thread_pool* pool = 0;

void* amalloc(size_t size)
{
    void* rv = malloc(size);

    if (rv == 0)
    {
        printf("Fatal Error: malloc() failed\n");
        exit(1);
    }

    return rv;
}

void diaMultUnthreaded(double* result, double* vec, int matW, int matH, double* data, int* offsets,
                       int numOffsets)
{
    for (int o = 0; o < numOffsets; ++o)
    {
        int offset = offsets[o];
        int start = 0 > offset ? 0 : offset;
        int end = matH < matH + offset ? matH : matH + offset;

        for (int i = start; i < end; ++i)
            result[i] += vec[i - offset] * data[o * matW + i];
    }
}

void diaMultRecSplitOffsets(double* result, double* vec, int matW, int matH, double* data,
                            int* offsets, int numOffsets)
{
    if (numOffsets <= 1)
    {
        // base case
        for (int o = 0; o < numOffsets; ++o)
        {
            int offset = offsets[o];
            int start = 0 > offset ? 0 : offset;
            int end = matH < matH + offset ? matH : matH + offset;

            for (int i = start; i < end; ++i)
                result[i] += vec[i - offset] * data[o * matW + i];
        }
    }
    else
    {
        // recursive case
        int midOffset = numOffsets / 2;

        double* resultA = (double*)amalloc(sizeof(double) * matH);
        double* resultB = (double*)amalloc(sizeof(double) * matH);

        std::thread t1(diaMultRecSplitOffsets, resultA, vec, matW, matH, data, offsets, midOffset);
        std::thread t2(diaMultRecSplitOffsets, resultB, vec, matW, matH, data + matH * midOffset,
                       offsets + midOffset, numOffsets - midOffset);

        t1.join();
        t2.join();

        // combine resultA and resultB
        for (int i = 0; i < matH; ++i)
            result[i] = resultA[i] + resultB[i];

        free(resultA);
        free(resultB);
    }
}

// base case for split range
void diaMultRecSplitRangeBase(int id, double* result, const double* const vec, const int matW,
                              const int matH, const double* const data, const int* const offsets,
                              const int numOffsets, const int startIndex, const int endIndex)
{
    for (int o = 0; o < numOffsets; ++o)
    {
        int offset = offsets[o];

        int start = 0 > offset ? 0 : offset;
        int end = matH < matH + offset ? matH : matH + offset;

        // limit it between startIndex and endIndex
        start = start < startIndex ? startIndex : start;
        end = end > endIndex ? endIndex : end;

        for (int i = start; i < end; ++i)
            result[i] += vec[i - offset] * data[o * matW + i];
    }
}

void diaMultRecSplitRange(double* result, double* const vec, const int matW, const int matH,
                          double* const data, const int* const offsets, const int numOffsets,
                          const int numSplits)
{
    std::vector<std::future<void>> results(numSplits);

    for (int s = 0; s < numSplits; ++s)
    {
        int startIndex = s * matH / numSplits;
        int endIndex = (s + 1) * matH / numSplits;

        results[s] = pool->push(diaMultRecSplitRangeBase, result, vec, matW, matH, data, offsets,
                                numOffsets, startIndex, endIndex);

        // threads[s] = std::thread(diaMultRecSplitRangeBase, result, vec, matW, matH, data,
        // offsets,
        //                         numOffsets, startIndex, endIndex);

        // diaMultRecSplitRangeBase(result, vec, matW, matH, data, offsets, numOffsets, startIndex,
        // endIndex);
    }

    for (int s = 0; s < numSplits; ++s)
        results[s].get();
}

void poolInit(int num)
{
    pool = new ctpl::thread_pool(num);
}

extern "C" {
void init(int num)
{
    poolInit(num);
}

void diaMult(double* result, double* vec, int matW, int matH, double* data, int* offsets,
             int numOffsets, int numSplits)
{
    diaMultRecSplitRange(result, vec, matW, matH, data, offsets, numOffsets, numSplits);
}
}
