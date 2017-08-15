// Stanley Bak
// GPU Matrix Multiplication interface using Cusp / Cuda
// June 2017

#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/multiply.h>
#include <cusp/print.h>

#include <sys/time.h>

typedef double FLOAT_TYPE;
typedef cusp::device_memory MEMORY_TYPE;
//typedef cusp::host_memory MEMORY_TYPE;

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
 
    for (int i = 0; i < nonZeroCount; ++i)
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

void _dot1(double* matrix, double* vector, int num_rows, int num_cols, double* result)
{
    // compute dot product of two matrices dot(matrix_a,matrix_b) using cusp::blas::dot

    const char* events[] =
    {
        "copy to cpus array2d",
        "copy matrices to device memory",
        "dot product",
        "copy result to cpu",
    };

    const int NUM_EVENTS = sizeof(events) / sizeof(events[0]);

    cudaEvent_t startEvents[NUM_EVENTS];
    cudaEvent_t stopEvents[NUM_EVENTS];
    int curEvent = 0;

    for (int i = 0; i < NUM_EVENTS; ++i)
    {
        cudaEventCreate(&startEvents[i]);
        cudaEventCreate(&stopEvents[i]);
    }

    //tic();
    cudaEventRecord(startEvents[curEvent]);

    cusp::array2d<FLOAT_TYPE, cusp::host_memory> host_matrix(num_rows, num_cols, 0);
    cusp::array1d<FLOAT_TYPE, cusp::host_memory> host_vector(num_cols, 0);

    for (int i = 0; i < num_rows; i++)
    {
        for(int j = 0; j < num_cols; j++) {
            host_matrix(i,j) = matrix[i*num_cols + j];
        }
    }

    for(int j = 0; j < num_cols; j++) {
            host_vector[j] = vector[j];
        }

    cudaEventRecord(stopEvents[curEvent++]);
    cudaEventRecord(startEvents[curEvent]);

    cusp::array2d<FLOAT_TYPE, MEMORY_TYPE> device_matrix(host_matrix);
    cusp::array1d<FLOAT_TYPE, MEMORY_TYPE> device_vector(host_vector);
    cusp::array1d<FLOAT_TYPE, MEMORY_TYPE> device_result(num_cols, 0);

    cudaEventRecord(stopEvents[curEvent++]);
    //toc("copy matrices to device memory");

    // compute dot product of two matrices using cusp::blas::dot
    //tic();
    cudaEventRecord(startEvents[curEvent]);
    
    for (int j = 0; j < num_cols; j++)
    {
        device_result[j] = cusp::blas::dot(device_matrix.row(j), device_vector);
    }

    int flopEvent = curEvent;
    cudaEventRecord(stopEvents[curEvent++]);
    //toc("compute the dot product of two matrices using cusp::blas::dot");

    // compute dot product of two matrices using cusp::blas::dot
    //tic();
    cudaEventRecord(startEvents[curEvent]);
    
    for (int j = 0; j < num_cols; j++)
    {
        result[j] = device_result[j];
    }

    cudaEventRecord(stopEvents[curEvent++]);
    //toc("copy result back to cpu memory");

    for (int i = 0; i < NUM_EVENTS; ++i)
    {
      cudaEventSynchronize(stopEvents[i]);
      
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, startEvents[i], stopEvents[i]);
      printf("%s: %f ms\n", events[i], milliseconds);

      if (i == flopEvent)
      {
        // compute number of flops during flopEvent
        int opCount = 2 * num_rows * num_cols;
        float mflops = opCount / milliseconds / 1000.0;

        printf("MFLOPS: %f\n", mflops);
      }
    } 
}


void _dot2(double* matrix, double* vector, int num_rows, int num_cols, double* result)
{
    const char* events[] =
    {
        "copy matrices to device memory",
        "convert to sparse matrix",
        "matrix-vector multiplication",
        "copy result to cpu",
    };

    const int NUM_EVENTS = sizeof(events) / sizeof(events[0]);

    cudaEvent_t startEvents[NUM_EVENTS];
    cudaEvent_t stopEvents[NUM_EVENTS];
    int curEvent = 0;

    for (int i = 0; i < NUM_EVENTS; ++i)
    {
        cudaEventCreate(&startEvents[i]);
        cudaEventCreate(&stopEvents[i]);
    }
    
    cudaEventRecord(startEvents[curEvent]);

    cusp::array2d<FLOAT_TYPE, cusp::host_memory> host_matrix(num_rows, num_cols, 0);
    cusp::array1d<FLOAT_TYPE, cusp::host_memory> host_vector(num_cols, 0);

    for (int i = 0; i < num_rows; i++)
    {
        for(int j = 0; j < num_cols; j++) {
            host_matrix(i,j) = matrix[i*num_cols + j];
        }
    }

    for(int j = 0; j < num_cols; j++) {
            host_vector[j] = vector[j];
        }

    cusp::array2d<FLOAT_TYPE, MEMORY_TYPE> device_matrix(host_matrix);
    cusp::array1d<FLOAT_TYPE, MEMORY_TYPE> device_vector(host_vector);
    cusp::array1d<FLOAT_TYPE, MEMORY_TYPE> device_result(num_cols, 0);

    cudaEventRecord(stopEvents[curEvent++]); 
    // compute dot product of two matrices dot(matrix_a,matrix_b) using matrix multiplication

    cudaEventRecord(startEvents[curEvent]);
    // convert device_matrix_b to hybrid format for efficient multiplication
    cusp::ell_matrix<int, FLOAT_TYPE, MEMORY_TYPE> device_sparse_matrix(device_matrix);

    cudaEventRecord(stopEvents[curEvent++]);

    cudaEventRecord(startEvents[curEvent]);

    cusp::multiply(device_sparse_matrix, device_vector, device_result);

    cudaEventRecord(stopEvents[curEvent++]);
    
    cudaEventRecord(startEvents[curEvent]);
    
    for (int j = 0; j < num_cols; j++)
    {
        result[j] = device_result[j];
    }

    cudaEventRecord(stopEvents[curEvent++]);

    
    printf("\n");
    for (int i = 0; i < NUM_EVENTS; ++i)
    {
      cudaEventSynchronize(stopEvents[i]);
      
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, startEvents[i], stopEvents[i]);
      printf("%s: %f ms\n", events[i], milliseconds);
    }
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

int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if (devProp.minor == 1) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

void printGpuStats()
{
    size_t free_byte;
    size_t total_byte;
    
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){

        printf("Error: cudaMemGetInfo failed - %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
           used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);


    //////////// print device info
    
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device Number: %d\n", i);
      
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

        printf("  Major revision number:         %d\n", prop.major);
        printf("  Minor revision number:         %d\n", prop.minor);
        printf("  Total global memory:           %d", (int)(prop.totalGlobalMem / 1024 / 1024));
        printf(" MB\n");
        printf("  Number of multiprocessors:     %d\n", prop.multiProcessorCount);
        printf("  Total amount of shared memory per block: %d\n", (int)prop.sharedMemPerBlock);
        printf("  Total registers per block:     %d\n", prop.regsPerBlock);
        printf("  Warp size:                     %d\n", prop.warpSize);
        printf("  Maximum memory pitch:          %d\n", (int)prop.memPitch);
        printf("  Total amount of constant memory: %d\n", (int)prop.totalConstMem);

        printf("  SPCores: %d\n", getSPcores(prop));
        printf("\n");
    }
}

int _hasGpu()
{
    int rv = 1;

    printf("has gpu started\n");

    // show memory usage of GPU
    //printGpuStats();
    
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

    printf("has gpu ended\n");
    
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

 void dot1(double* matrix, double* vector, int num_rows, int num_cols, double* result)
{

    _dot1(matrix, vector, num_rows, num_cols, result);
}

    
 void dot2(double* matrix, double* vector, int num_rows, int num_cols, double* result)
{

    _dot2(matrix, vector, num_rows, num_cols, result);
}


}
