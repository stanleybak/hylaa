// Stanley Bak
// C++-based parallel diagonal matrix multiplication

#include <stdio.h>
#include <stdlib.h>
#include <thread>

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

        if (o == 0)  // overwrite result[i] in case there was junk in there
        {
            for (long i = startIndex; i < start; ++i)
                result[i] = 0;

            for (long i = start; i < end; ++i)
                result[i] = vec[i - offset] * data[o * matW + i];

            for (long i = end; i < endIndex; ++i)
                result[i] = 0;
        }
        else  // accumulate sum into result[i]
        {
            for (long i = start; i < end; ++i)
                result[i] += vec[i - offset] * data[o * matW + i];
        }
    }
}

// (threaded) recursive case
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
        threads[s].join();
}

extern "C" {

void diaFastMult(double* result, double* vec, int matW, int matH, double* data, int* offsets,
             int numOffsets, int numSplits)
{
    if (result == vec)
    {
        printf("Error: result and vec point to the same memory in diaFastMult\n");
        exit(1);
    }
    
    if (matW < 50 * 50 * 50)  // for small matrices it's quicker to do it in one thread
        diaMultRecSplitRangeBase(result, vec, matW, matH, data, offsets, numOffsets, 0, matW);
    else
        diaMultRecSplitRange(result, vec, (long)matW, (long)matH, data, offsets, numOffsets,
                             numSplits);
}
}
