// Stanley Bak
// C++-based diagonal matrix multiplication

#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

using namespace std;

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
void diaMultRecSplitRangeBase(double* result, double* vec, long matW, long matH, double* data,
                              int* offsets, int numOffsets, long startIndex, long endIndex)
{
    if (startIndex < 0 || endIndex < 0)
    {
        printf("Invalid index range in split multiplication: %ld to %ld\n", startIndex, endIndex);
        exit(1);
    }

    for (long o = 0; o < numOffsets; ++o)
    {
        long offset = offsets[o];

        long start = 0 > offset ? 0 : offset;
        long end = matH < matH + offset ? matH : matH + offset;

        // limit it between startIndex and endIndex
        start = start < startIndex ? startIndex : start;
        end = end > endIndex ? endIndex : end;

        for (long i = start; i < end; ++i)
            result[i] += vec[i - offset] * data[o * matW + i];
    }
}

void diaMultRecSplitRange(double* result, double* vec, long matW, long matH, double* data,
                          int* offsets, int numOffsets, int numSplits)
{
    std::thread threads[numSplits];

    if (sizeof(long) < 8)
    {
        printf("Expected sizeof(long) >= 8, got %lu\n", sizeof(long));
        exit(1);
    }

    for (long s = 0; s < numSplits; ++s)
    {
        long startIndex = s * matH / numSplits;
        long endIndex = (s + 1) * matH / numSplits;

        threads[s] = std::thread(diaMultRecSplitRangeBase, result, vec, matW, matH, data, offsets,
                                 numOffsets, startIndex, endIndex);
    }

    for (int s = 0; s < numSplits; ++s)
        // results[s].get();
        threads[s].join();
}

extern "C" {

void diaMult(double* result, double* vec, int matW, int matH, double* data, int* offsets,
             int numOffsets, int numSplits)
{
    if (matW < 50 * 50 * 50)  // for small matrices it's quicker to do it in one thread
        diaMultRecSplitRangeBase(result, vec, matW, matH, data, offsets, numOffsets, 0, matW);
    else
        diaMultRecSplitRange(result, vec, (long)matW, (long)matH, data, offsets, numOffsets,
                             numSplits);
}
}
