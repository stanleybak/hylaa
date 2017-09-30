// Dung Tran & Stanley Bak
// Krylov subspace - based simulation using Gpu- Cusp / Cuda for sparse ode
// June 2017

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/multiply.h>
#include <cusp/multiply.h>
#include <cusp/print.h>

// CUDA runtime
//#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gpu_util.h"

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

typedef double FLOAT_TYPE;

// anyArray is only used to identify if CPU (do nothing) or GPU (initialize handle)
void createCublasHandle(cublasHandle_t &cublasHandle,
                        cusp::array1d<FLOAT_TYPE, cusp::device_memory> *anyArray)
{
    if (cublasHandle != 0)
        error("cublasHandle initialized twice");

    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
        error("cublasCreate() failed");

    cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
}

// anyArray is only used to identify if CPU (do nothing) or GPU (initialize handle)
void createCublasHandle(cublasHandle_t &cublasHandle,
                        cusp::array1d<FLOAT_TYPE, cusp::host_memory> *anyArray)
{
    if (cublasHandle != 0)
        error("cublasHandle initialized twice");
}

void myDot(cublasHandle_t &cublasHandle, unsigned long size,
           cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view &a,
           cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view &b,
           cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view &resultView, int resultIndex)
{
    // cpu implementation
    FLOAT_TYPE d = cusp::blas::dot(a, b);

    resultView[resultIndex] = d;
}

void myDot(cublasHandle_t &cublasHandle, unsigned long size,
           cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view &a,
           cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view &b,
           cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view &resultView, int resultIndex)
{
    double *x = thrust::raw_pointer_cast(&a[0]);
    double *y = thrust::raw_pointer_cast(&b[0]);
    double *result = thrust::raw_pointer_cast(&resultView[resultIndex]);

    if (cublasDdot(cublasHandle, size, x, 1, y, 1, result) != CUBLAS_STATUS_SUCCESS)
        error("cublasDdot() failed");
}

// subtract dots * prevVec from curVec using axpy
void myAxpy(cublasHandle_t &cublasHandle,
            // numsView: [-1, 0, 1, temp storage]
            cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view &numsView,
            cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view &a,
            cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view &resView,
            cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view &hView, int hIndex)
{
    // cpu implementation
    cusp::blas::axpy(a, resView, -hView[hIndex]);
}

// subtract dots * prevVec from curVec using axpy
void myAxpy(cublasHandle_t &cublasHandle,
            // numsView: [-1, 0, 1, temp storage]
            cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view &numsView,
            cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view &a,
            cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view &resView,
            cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view &hView, int hIndex)
{
    // gpu implementation
    // cusp::blas::axpy(a, b, -hView[hIndex]);

    int count = a.size();

    double *minusOne = thrust::raw_pointer_cast(&numsView[0]);
    double *zero = thrust::raw_pointer_cast(&numsView[1]);
    double *temp = thrust::raw_pointer_cast(&numsView[3]);
    double *x = thrust::raw_pointer_cast(&a[0]);
    double *res = thrust::raw_pointer_cast(&resView[0]);
    double *h = thrust::raw_pointer_cast(&hView[hIndex]);

    // copy h to temp
    if (cublasDcopy(cublasHandle, 1, h, 1, temp, 1) != CUBLAS_STATUS_SUCCESS)
        error("cublasDcopy() failed");

    // scale temp by -1
    if (cublasDscal(cublasHandle, 1, minusOne, temp, 1) != CUBLAS_STATUS_SUCCESS)
        error("cublasDscal() failed");

    // do the axpy (alpha = temp)
    if (cublasDaxpy(cublasHandle, count, temp, x, 1, res, 1) != CUBLAS_STATUS_SUCCESS)
        error("cublasDaxpy() failed");
}

template <class MEMORY_TYPE>
class CuspData
{
    typedef cusp::array1d<FLOAT_TYPE, MEMORY_TYPE> Array1d;
    typedef typename Array1d::view Array1dView;

    typedef cusp::array2d<FLOAT_TYPE, MEMORY_TYPE> Array2d;
    typedef typename Array2d::view Array2dView;

    typedef cusp::array1d<FLOAT_TYPE, cusp::host_memory> HostFloatArray1d;
    typedef typename HostFloatArray1d::view HostFloatArray1dView;

    typedef cusp::array1d<int, cusp::host_memory> HostIntArray1d;
    typedef typename HostIntArray1d::view HostIntArray1dView;

    typedef cusp::coo_matrix<int, FLOAT_TYPE, MEMORY_TYPE> CooMatrix;

    typedef cusp::hyb_matrix<int, FLOAT_TYPE, MEMORY_TYPE> HybMatrix;

    typedef cusp::csr_matrix<int, FLOAT_TYPE, cusp::host_memory> HostCsrMatrix;
    typedef typename HostCsrMatrix::view HostCsrMatrixView;

   private:
    HybMatrix *aMatrix;
    HybMatrix *keyDirMatrix;

    Array1d *vMatrix;     // (i+1) * n
    Array1d *hMatrix;     // (i+1) * i
    Array1d *vProjected;  // k * (i+1)

    unsigned long _n;  // number of dimensions in the system
    unsigned long _k;  // number of key directions
    unsigned long _i;  // number of arnoldi iterations

    // profiling variables
    bool useProfiling;
    unsigned long aMatrixNonzeros;
    unsigned long keyDirMatrixNonzeros;

    // cublas variables
    cublasHandle_t cublasHandle;
    Array1d *cuspNums;  // [-1, 0, 1, temp_val]

   public:
    GpuUtil util;  // timers and other utility functions

    CuspData(bool useCpu) : util(useCpu), cublasHandle(0)
    {
        aMatrix = 0;
        keyDirMatrix = 0;

        vMatrix = 0;
        hMatrix = 0;
        vProjected = 0;

        cuspNums = 0;

        reset();  // this resets all variables
    }

    ~CuspData()
    {
        reset();

        cudaDeviceReset();
    }

    void reset()
    {
        if (aMatrix != 0)
        {
            delete aMatrix;
            aMatrix = 0;
        }

        if (keyDirMatrix != 0)
        {
            delete keyDirMatrix;
            keyDirMatrix = 0;
        }

        if (vMatrix != 0)
        {
            delete vMatrix;
            vMatrix = 0;
        }

        if (hMatrix != 0)
        {
            delete hMatrix;
            hMatrix = 0;
        }

        if (vProjected != 0)
        {
            delete vProjected;
            vProjected = 0;
        }

        if (cuspNums != 0)
        {
            delete cuspNums;
            cuspNums = 0;
        }

        if (cublasHandle)
        {
            if (cublasDestroy(cublasHandle) != CUBLAS_STATUS_SUCCESS)
                error("cublasDestroy failed");

            cublasHandle = 0;
        }

        util.clearTimers();

        _n = 0;
        _k = 0;
        _i = 0;

        setUseProfiling(false);
        aMatrixNonzeros = 0;
        keyDirMatrixNonzeros = 0;
    }

    void setUseProfiling(bool enabled)
    {
        useProfiling = enabled;
        util.setUseProfiling(enabled);
    }

    // load A matrix, passed in as a csr matrix
    void loadAMatrix(unsigned long w, unsigned long h, int *rowOffsets, unsigned long rowOffsetsLen,
                     int *colInds, unsigned long colIndsLen, FLOAT_TYPE *values,
                     unsigned long valuesLen)
    {
        if (_n == 0)
            error("loadKeyDirMatrix() called before preallocate() (_n==0)\n");

        if (w != h)
            error("loadAMatrix() expected square A matrix, got w=%lu, h=%lu", w, h);

        if (w != _n)
            error(
                "in loadAMatrix() size (%lu) not to equal dims from preallocate() "
                "(%lu)",
                w, _n);

        if (useProfiling)
        {
            FLOAT_TYPE size = rowOffsetsLen * sizeof(int);
            size += colIndsLen * sizeof(int);
            size += valuesLen * sizeof(FLOAT_TYPE);

            printf(
                "loadAMatrix() with sparse matrix size: %.2f MB (%lu nonzeros). "
                "Memory on device: "
                "%.2f MB\n",
                size / 1024.0 / 1024.0, valuesLen, getFreeMemoryMb());
        }

        aMatrixNonzeros = valuesLen;

        HostIntArray1dView rowOffsetsView(rowOffsets, rowOffsets + rowOffsetsLen);
        HostIntArray1dView colIndsView(colInds, colInds + colIndsLen);
        HostFloatArray1dView valuesView(values, values + colIndsLen);

        HostCsrMatrixView view(_n, _n, valuesLen, rowOffsetsView, colIndsView, valuesView);

        if (aMatrix != 0)
        {
            delete aMatrix;
            aMatrix = 0;
        }

        try
        {
            aMatrix = new HybMatrix(view);
        }
        catch (std::bad_alloc)
        {
            error("memory allocation of aMatrix failed\n");
        }

        // one-time preallocate cusp_nums (shouldn't fail)
        if (cuspNums == 0)
        {
            try
            {
                HostFloatArray1d temp(4);
                temp[0] = -1;
                temp[1] = 0;
                temp[2] = 1;
                temp[3] = 0;
                cuspNums = new Array1d(temp);
            }
            catch (std::bad_alloc)
            {
                error("cuspNums memory allocation failed\n");
            }
        }

        // one-time allocation of cublasHandle
        if (cublasHandle == 0)
            createCublasHandle(cublasHandle, vMatrix);
    }

    // load key dir matrix, passed in as a csr matrix
    void loadKeyDirMatrix(unsigned long w, unsigned long h, int *rowOffsets,
                          unsigned long rowOffsetsLen, int *colInds, unsigned long colIndsLen,
                          FLOAT_TYPE *values, unsigned long valuesLen)
    {
        if (_n == 0)
            error("loadKeyDirMatrix() called before preallocate() (_n==0)\n");

        if (w != _n)
            error(
                "in loadKeyDirMatrix() width (%lu) to equal dims from "
                "preallocate() (%lu)",
                w, _n);

        if (w != _n)
            error(
                "in loadKeyDirMatrix() height (%lu) to equal keyDirMatSize from "
                "preallocate() "
                "(%lu)",
                h, _k);

        if (useProfiling)
        {
            FLOAT_TYPE size = rowOffsetsLen * sizeof(int);
            size += colIndsLen * sizeof(int);
            size += valuesLen * sizeof(FLOAT_TYPE);

            printf(
                "loadKeyDirMatrix() with dense matrix size: %.2f MB, Memory on "
                "device: %.2f MB\n",
                size / 1024.0 / 1024.0, getFreeMemoryMb());
        }

        keyDirMatrixNonzeros = valuesLen;

        HostIntArray1dView rowOffsetsView(rowOffsets, rowOffsets + rowOffsetsLen);
        HostIntArray1dView colIndsView(colInds, colInds + colIndsLen);
        HostFloatArray1dView valuesView(values, values + colIndsLen);

        HostCsrMatrixView view(h, w, valuesLen, rowOffsetsView, colIndsView, valuesView);

        if (keyDirMatrix != 0)
        {
            delete keyDirMatrix;
            keyDirMatrix = 0;
        }

        try
        {
            keyDirMatrix = new HybMatrix(view);
        }
        catch (std::bad_alloc)
        {
            error("memory allocation of keyDirMatrix() failed\n");
        }
    }

    FLOAT_TYPE getFreeMemoryMb()
    {
        unsigned long bytes = util.getFreeMemory();
        FLOAT_TYPE rv = (bytes / 1024.0 / 1024.0);

        return rv;
    }

    // frees memory if it was previously allocated, returns false if memory error
    // occurs
    bool preallocateMemory(unsigned long arnoldiIt, unsigned long dims, unsigned long keyDirMatSize)
    {
        bool success = true;

        if (_n != 0 && dims != _n && aMatrix != 0)
        {
            // num dimensions changed... free aMatrix
            delete aMatrix;
            aMatrix = 0;
        }

        if (_k != 0 && keyDirMatSize != _k && keyDirMatrix != 0)
        {
            // num key directions changed
            delete keyDirMatrix;
            keyDirMatrix = 0;
        }

        _n = dims;
        _i = arnoldiIt;
        _k = keyDirMatSize;

        if (useProfiling)
            printf(
                "preallocateMemory() called with profiling, Free memory on "
                "device: %.2f MB\n",
                getFreeMemoryMb());

        // free memory which we will newly allocate

        if (hMatrix != 0)
        {
            delete hMatrix;
            hMatrix = 0;
        }

        if (vMatrix != 0)
        {
            delete vMatrix;
            vMatrix = 0;
        }

        if (vProjected != 0)
        {
            delete vProjected;
            vProjected = 0;
        }

        try
        {
            // preallocate hMatrix, numParInit * iterations * iterations
            unsigned long hMatrixSize = _i * (_i + 1);

            if (useProfiling)
                printf(
                    "Trying to allocate %.2f MB for hMatrix (remaining memory %.2f "
                    "MB)...\n",
                    sizeof(FLOAT_TYPE) * hMatrixSize / 1024.0 / 1024.0, getFreeMemoryMb());

            hMatrix = new Array1d(hMatrixSize, 0);

            // preallocate vMatrix, width = dims * iterations, height = numParInit
            unsigned long vMatrixSize = _n * (_i + 1);

            if (useProfiling)
                printf(
                    "Trying to allocate %.2f MB for vMatrix (remaining memory %.2f "
                    "MB)...\n",
                    sizeof(FLOAT_TYPE) * vMatrixSize / 1024.0 / 1024.0, getFreeMemoryMb());

            vMatrix = new Array1d(vMatrixSize, 0);

            // preallocate vProjected
            unsigned long vProjectedSize = _k * (_i + 1);

            if (useProfiling)
                printf(
                    "Trying to allocate %.2f MB for vProjected (remaining memory %.2f "
                    "MB)...\n",
                    sizeof(FLOAT_TYPE) * vProjectedSize / 1024.0 / 1024.0, getFreeMemoryMb());

            vProjected = new Array1d(vProjectedSize, 0);
        }
        catch (std::bad_alloc)
        {
            if (useProfiling)
                printf("memory allocation failed\n");

            _i = 0;
            _n = 0;
            _k = 0;
            success = false;
        }

        return success;
    }

    // reads/writes from/to vMatrix, writes to hMatrix
    void arnoldi(unsigned long iterations)
    {
        Array1dView cuspNumsView = cuspNums->subarray(0, 4);

        long start = now();

        for (unsigned long it = 1; it <= iterations; it++)
        {
            if (useProfiling && it % 100 == 0)
            {
                long elapsedUs = now() - start;  // microseconds
                double elapsedSec = elapsedUs / 1000.0 / 1000.0;
                double frac = ((double)it / iterations);
                double totalSec = elapsedSec / frac;
                double totalMin = totalSec / 60.0;
                double remainingSec = totalSec - elapsedSec;
                double remainingMin = remainingSec / 60.0;
                double remainingHr = remainingMin / 60.0;

                printf(
                    "iter %lu / %lu (%.2f). Total: %.1f min, ETA: %.1f sec (%.2f min) (%.2f hr)\n",
                    it, iterations, frac, totalMin, remainingSec, remainingMin, remainingHr);
            }

            util.tic("sparse matrix vector multiply");
            unsigned long prevRowOffset = _n * (it - 1);
            unsigned long curRowOffset = _n * it;

            Array1dView vecView = vMatrix->subarray(prevRowOffset, _n);
            Array1dView resultView = vMatrix->subarray(curRowOffset, _n);

            cusp::multiply(*aMatrix, vecView, resultView);
            util.toc("sparse matrix vector multiply", 2 * aMatrixNonzeros);

            ////////

            util.tic("dots & axpy");
            unsigned long rowOffset = (it - 1) * (_i + 1);
            resultView = hMatrix->subarray(rowOffset, it);

            rowOffset = _n * it;

            Array1dView curVec = vMatrix->subarray(rowOffset, _n);

            // combined dot/axpy to have modified gram-schmidt orthogonalization
            // (more stable)
            for (unsigned long row = 0; row < it; ++row)
            {
                Array1dView curRow = vMatrix->subarray(row * _n, _n);

                // util.tic("dot");
                myDot(cublasHandle, _n, curVec, curRow, resultView, row);
                // util.toc("dot", 2 * _n);

                rowOffset = _n * row;
                Array1dView prevVec = vMatrix->subarray(rowOffset, _n);

                // util.tic("axpy");
                myAxpy(cublasHandle, cuspNumsView, prevVec, curVec, resultView, row);
                // util.toc("axpy", 2 * _n);
            }

            util.toc("dots & axpy", 2 * 2 * _n * it);

            ////////

            rowOffset = _n * it;
            curVec = vMatrix->subarray(rowOffset, _n);

            util.tic("nrm2");
            FLOAT_TYPE magnitude = cusp::blas::nrm2(curVec);
            util.toc("nrm2");

            // store magnitude in H
            rowOffset = (it - 1) * (_i + 1);
            (*hMatrix)[rowOffset + it] = magnitude;

            // scale vector
            if (magnitude < 1e-13)
            {
                // cusp::blas::scal(curVec, 0.0);

                // printf("Break at it = %lu! Vec norm = %f Profile if this actually helps.\n", it,
                //       magnitude);
                break;
            }
            else
            {
                util.tic("scale");
                cusp::blas::scal(curVec, 1.0 / magnitude);
                util.toc("scale");
            }
        }
    }

    void projectV(unsigned long iterations)
    {
        // use vMatrix and keyDirMatrix to produce vProjected

        for (unsigned long iteration = 0; iteration <= iterations; ++iteration)
        {
            // Can we do this with a single matrix-matrix mult???
            unsigned long rowOffset = _n * (iteration);
            Array1dView vecView = vMatrix->subarray(rowOffset, _n);

            // result view is in vProjected
            rowOffset = _k * iteration;

            Array1dView resultView = vProjected->subarray(rowOffset, iterations);

            util.tic("project-v sparse matrix vector multiply");
            cusp::multiply(*keyDirMatrix, vecView, resultView);
            util.toc("project-v sparse matrix vector multiply", 2 * keyDirMatrixNonzeros);
        }
    }

    void getProfilingData(const char *name, double *ms, double *gflops)
    {
        util.getProfilingData(name, ms, gflops);
    }

    void printProfilingData()
    {
        util.printTimers();
        util.clearTimers();
    }

    // initialize with a passed-in vector
    void initVec(FLOAT_TYPE *vec, unsigned long len)
    {
        util.tic("init arnoldi");

        if (len != _n)
            error("initArnoldi called with bad len = %lu (_n = %lu)\n", len, _n);

        // initialize with zeros
        cusp::blas::fill(*vMatrix, 0.0);
        cusp::blas::fill(*hMatrix, 0.0);
        cusp::blas::fill(*vProjected, 0.0);

        HostFloatArray1dView vecView(vec, vec + len);
        Array1dView vMatrixView = vMatrix->subarray(0, _n);
        cusp::blas::copy(vecView, vMatrixView);

        // sanity check that norm of vec is 1
        FLOAT_TYPE magnitude = cusp::blas::nrm2(vMatrixView);
        FLOAT_TYPE tol = 1e-6;

        if (magnitude < 1.0 - tol || magnitude > 1.0 + tol)
            error("initial arnoldi vec must be normalized first (magnitude was %f)", magnitude);

        util.toc("init arnoldi");
    }

    // initialize with a unit-vector in the given dimention
    void initUnit(unsigned long dim)
    {
        util.tic("init arnoldi");

        if (dim > _n)
            error("initParallelArnoldiV called with single dim=%lu, but _n=%lu", dim, _n);

        // initialize with zeros
        cusp::blas::fill(*vMatrix, 0.0);
        cusp::blas::fill(*hMatrix, 0.0);
        cusp::blas::fill(*vProjected, 0.0);

        // put a 1.0 in the correct spot
        (*vMatrix)[dim] = 1.0;

        util.toc("init arnoldi");
    }

    // copy memory, run arnoldi, project results, copy memory back (everything but init)
    void compute(FLOAT_TYPE *resultH, unsigned long sizeResultH, FLOAT_TYPE *resultPV,
                 unsigned long sizeResultPV)
    {
        if (aMatrix == 0)
            error("arnoldiParallel() called before loadAMatrix()\n");

        if (keyDirMatrix == 0)
            error("arnoldiParallel() called before loadKeyDirMatrix()\n");

        if (_i == 0)
            error("arnoldiParallel() called before preallocate() (_i==0)\n");

        // check expected results sizes
        unsigned long expectedH = _i * (_i + 1);
        unsigned long expectedPV = (_i + 1) * _k;

        if (sizeResultH != expectedH)
            error("Wrong size for resultH with i = %lu. Got %lu, expected %lu.", _i, sizeResultH,
                  expectedH);

        if (sizeResultPV != expectedPV)
            error(
                "Wrong size for resultPV with (i, k) = (%lu, %lu, %lu). Got "
                "%lu, expected %lu.",
                _i, _k, sizeResultPV, expectedPV);

        util.tic("compute");

        util.tic("arnoldi()");
        arnoldi(_i);
        util.toc("arnoldi()");

        // project v_matrix onto keyDirMatrix
        util.tic("projectV()");
        projectV(_i);
        util.toc("projectV()");

        // copying H matrix to np.ndarray
        util.tic("copying H matrix to np.ndarray");
        HostFloatArray1dView hostHView(resultH, resultH + expectedH);
        cusp::blas::copy(*hMatrix, hostHView);  // hostHView = *hMatrix
        util.toc("copying H matrix to np.ndarray");

        // copy vProjected to np.ndarray

        util.tic("copying V-projected matrix to np.ndarray");
        HostFloatArray1dView hostPVView(resultPV, resultPV + expectedPV);
        cusp::blas::copy(*vProjected, hostPVView);  // hostPVView = *vProjected
        util.toc("copying V-projected matrix to np.ndarray");

        util.toc("compute");
    }
};

CuspData<cusp::host_memory> cuspDataCpu(true);
CuspData<cusp::device_memory> cuspDataGpu(false);

extern "C" {
unsigned long hasGpu()
{
    return cuspDataGpu.util.hasGpu();
}

void reset()
{
    cuspDataCpu.reset();
    cuspDataGpu.reset();
}

////// CPU Version
void setUseProfilingCpu(unsigned long enabled)
{
    cuspDataCpu.setUseProfiling(enabled != 0);
}

// as csr matrix
void loadAMatrixCpu(unsigned long w, unsigned long h, int *rowOffsets, unsigned long rowOffsetsLen,
                    int *colInds, unsigned long colIndsLen, FLOAT_TYPE *values,
                    unsigned long valuesLen)
{
    cuspDataCpu.loadAMatrix(w, h, rowOffsets, rowOffsetsLen, colInds, colIndsLen, values,
                            valuesLen);
}

void loadKeyDirMatrixCpu(unsigned long w, unsigned long h, int *rowOffsets,
                         unsigned long rowOffsetsLen, int *colInds, unsigned long colIndsLen,
                         FLOAT_TYPE *values, unsigned long valuesLen)
{
    cuspDataCpu.loadKeyDirMatrix(w, h, rowOffsets, rowOffsetsLen, colInds, colIndsLen, values,
                                 valuesLen);
}

FLOAT_TYPE getFreeMemoryMbCpu()
{
    return cuspDataCpu.getFreeMemoryMb();
}

unsigned long preallocateMemoryCpu(unsigned long arnoldiIt, unsigned long dims,
                                   unsigned long keyDirMatSize)
{
    return cuspDataCpu.preallocateMemory(arnoldiIt, dims, keyDirMatSize) ? 1 : 0;
}

void arnoldiUnitCpu(unsigned long dim, FLOAT_TYPE *resultH, unsigned long sizeResultH,
                    FLOAT_TYPE *resultPV, unsigned long sizeResultPV)
{
    cuspDataCpu.initUnit(dim);
    cuspDataCpu.compute(resultH, sizeResultH, resultPV, sizeResultPV);
}

void arnoldiVecCpu(FLOAT_TYPE *vec, unsigned long vecLen, FLOAT_TYPE *resultH,
                   unsigned long sizeResultH, FLOAT_TYPE *resultPV, unsigned long sizeResultPV)
{
    cuspDataCpu.initVec(vec, vecLen);
    cuspDataCpu.compute(resultH, sizeResultH, resultPV, sizeResultPV);
}

void getProfilingDataCpu(const char *name, FLOAT_TYPE *resultVec, unsigned long resultVecLen)
{
    if (resultVecLen != 2)
        error("Expected resultVecLen == 2 in getProfilingData()");

    cuspDataCpu.getProfilingData(name, &resultVec[0], &resultVec[1]);
}

void printProfilingDataCpu()
{
    cuspDataCpu.printProfilingData();
}

////// GPU Version
void setUseProfilingGpu(unsigned long enabled)
{
    cuspDataGpu.setUseProfiling(enabled != 0);
}

// as csr matrix
void loadAMatrixGpu(unsigned long w, unsigned long h, int *rowOffsets, unsigned long rowOffsetsLen,
                    int *colInds, unsigned long colIndsLen, FLOAT_TYPE *values,
                    unsigned long valuesLen)
{
    cuspDataGpu.loadAMatrix(w, h, rowOffsets, rowOffsetsLen, colInds, colIndsLen, values,
                            valuesLen);
}

// as csr matrix
void loadKeyDirMatrixGpu(unsigned long w, unsigned long h, int *rowOffsets,
                         unsigned long rowOffsetsLen, int *colInds, unsigned long colIndsLen,
                         FLOAT_TYPE *values, unsigned long valuesLen)
{
    cuspDataGpu.loadKeyDirMatrix(w, h, rowOffsets, rowOffsetsLen, colInds, colIndsLen, values,
                                 valuesLen);
}

FLOAT_TYPE getFreeMemoryMbGpu()
{
    return cuspDataGpu.getFreeMemoryMb();
}

unsigned long preallocateMemoryGpu(unsigned long arnoldiIterations, unsigned long dims,
                                   unsigned long keyDirMatSize)
{
    return cuspDataGpu.preallocateMemory(arnoldiIterations, dims, keyDirMatSize) ? 1 : 0;
}

void arnoldiUnitGpu(unsigned long dim, FLOAT_TYPE *resultH, unsigned long sizeResultH,
                    FLOAT_TYPE *resultPV, unsigned long sizeResultPV)
{
    cuspDataGpu.initUnit(dim);
    cuspDataGpu.compute(resultH, sizeResultH, resultPV, sizeResultPV);
}

void arnoldiVecGpu(FLOAT_TYPE *vec, unsigned long vecLen, FLOAT_TYPE *resultH,
                   unsigned long sizeResultH, FLOAT_TYPE *resultPV, unsigned long sizeResultPV)
{
    cuspDataGpu.initVec(vec, vecLen);
    cuspDataGpu.compute(resultH, sizeResultH, resultPV, sizeResultPV);
}

void getProfilingDataGpu(const char *name, FLOAT_TYPE *resultVec, unsigned long resultVecLen)
{
    if (resultVecLen != 2)
        error("Expected resultVecLen == 2 in getProfilingData()");

    cuspDataGpu.getProfilingData(name, &resultVec[0], &resultVec[1]);
}

void printProfilingDataGpu()
{
    cuspDataGpu.printProfilingData();
}

}  // end extern "C"
