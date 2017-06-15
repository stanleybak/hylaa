// Stanley Bak
// GPU Matrix Multiplication interface using Cusp / Cuda
// June 2017

#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/multiply.h>

#include <sys/time.h>

typedef double FLOAT_TYPE;
// shared matrix in device memory
static cusp::hyb_matrix<int, FLOAT_TYPE, cusp::device_memory>* curMatrix = 0;

static int nonZeros = 0;

// timing shared variable
static long lastTicUs = 0;

void error(const char* msg)
{
    printf("Fatal Error: %s\n", msg);
    exit(1);
}

void tic()
{
    struct timeval now;
    
    if(gettimeofday( &now, 0))
        error("gettimeofday");
        
    lastTicUs = 1000000 * now.tv_sec + now.tv_usec;
}

// returns the us elaspsed
long toc(const char* label)
{
    struct timeval now;
    
    if(gettimeofday( &now, 0))
        error("gettimeofday");
        
    long nowUs = 1000000 * now.tv_sec + now.tv_usec;
    long dif = nowUs - lastTicUs;
    
    printf("%s: %.4f ms\n", label, dif / 1000.0);
    
    return dif;
}

void _loadMatrix(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries, int nonZeroCount)
{
    tic();
    cusp::coo_matrix<int, FLOAT_TYPE, cusp::host_memory> hostMatrix(w, h, nonZeroCount);
        
    printf("loadMatrix() called, estimated size in memory of sparse matrix: %.2f MB (%d nonzeros)\n", 
        nonZeroCount * (8 + 4 + 4) / 1024.0 / 1024.0, nonZeroCount);

    // initialize matrix entries on host
    int index = 0;
 
    for (unsigned int i = 0; i < nonZeroCount; ++i)
    {
        int row = nonZeroRows[i];
        int col = nonZeroCols[i];
        double val = nonZeroEntries[i];
        
        hostMatrix.row_indices[index] = row;
        hostMatrix.column_indices[index] = col;
        hostMatrix.values[index++] = val;
    }
    
    toc("creating host coo matrix");
    
    tic();
    if (curMatrix != 0)
    {
        delete curMatrix;
        curMatrix = 0;
    }
    
    curMatrix = new (std::nothrow) cusp::hyb_matrix<int, FLOAT_TYPE,cusp::device_memory>(hostMatrix);
        
    if (curMatrix == 0)
        error("allocation of heap-based csr matrix in device memory returned nullptr");
        
    toc("copying matrix to device memory");
    nonZeros = nonZeroCount;
}

void _multiply(double* vector, double* result, int size)
{
    if (curMatrix == 0)
        error("loadMatrix must be called before multiply");
    
    // initialize input vector
    tic();
    cusp::array1d<FLOAT_TYPE, cusp::host_memory> hostVec(size);
    
    for (int i = 0; i < size; ++i)
        hostVec[i] = vector[i];
    toc("creating hostVec vector");
    
    tic();
    cusp::array1d<FLOAT_TYPE,cusp::device_memory> deviceVec(hostVec);
    toc("copying vector to device memory");
    
    // create device vec; should be negligible time, don't even measure
    cusp::array1d<FLOAT_TYPE, cusp::device_memory> resultVec(size);
    
    // compute result = A * stat
    tic();
    cusp::multiply(*curMatrix, deviceVec, resultVec);
    cudaDeviceSynchronize(); // wait until prior kernel is finished
    long usElapsed = toc("matrix-vector multiplication");
    
    // each nonzero is 2 FLOPS (one for the multiplication, and another for an addition
    // microseconds (us) is 1000 * 1000, which is close to megaflops
    double megaFlopsPerSecond = 2 * nonZeros / usElapsed; 
    printf("achieved megaflops = %f\n", megaFlopsPerSecond);
    
    tic();
    cusp::array1d<FLOAT_TYPE, cusp::host_memory> resultHost(resultVec);
    toc("copying result to host memory");
    
    tic();
    for (int i = 0; i < size; ++i)
        result[i] = resultHost[i];
        
    toc("copying to np.ndarray");
}

int _hasGpu()
{
    int rv = 1;
    
    try
    {
        cusp::array1d<FLOAT_TYPE, cusp::host_memory> hostVec(10);
    
        for (int i = 0; i < 10; ++i)
            hostVec[i] = 0;

        cusp::array1d<FLOAT_TYPE,cusp::device_memory> deviceVec(hostVec);
    }
    catch(std::exception &e)
    {
        printf("hasGpu() Failed: %s\n", e.what());
        rv = 0;
    }
    
    return rv;
}

extern "C"
{
int hasGpu()
{
    return _hasGpu();
}

void loadMatrix(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries, int nonZeroCount)
{
    _loadMatrix(w, h, nonZeroRows, nonZeroCols, nonZeroEntries, nonZeroCount);
}

void multiply(double* vector, double* result, int size)\
{
    _multiply(vector, result, size);
}

}
