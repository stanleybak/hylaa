// Dung Tran & Stanley Bak
// Krylov subspace - based simulation using Gpu- Cusp / Cuda for sparse ode
// June 2017

#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <sys/time.h>
#include "gpu_util.h"

template <class FLOAT_TYPE, class MEMORY_TYPE>
class CuspData
{
    typedef cusp::array1d<FLOAT_TYPE, MEMORY_TYPE> Array1d;
    typedef typename Array1d::view Array1dView;

    typedef cusp::array1d<FLOAT_TYPE, cusp::host_memory> HostFloatArray1d;
    typedef typename HostFloatArray1d::view HostFloatArray1dView;

    typedef cusp::array1d<int, cusp::host_memory> HostIntArray1d;
    typedef typename HostIntArray1d::view HostIntArray1dView;

    typedef cusp::csr_matrix<int, FLOAT_TYPE, MEMORY_TYPE> CsrMatrix;

    typedef cusp::csr_matrix<int, FLOAT_TYPE, cusp::host_memory> HostCsrMatrix;
    typedef typename HostCsrMatrix::view HostCsrMatrixView;

   public:
    GpuUtil util;  // timers and other utility functions

    CsrMatrix* aTranspose;
    CsrMatrix* keyDirMatrix;

    Array1d* vMatrix;
    Array1d* hMatrix;
    Array1d* vProjected;

    int _n;  // number of dimensions in the system
    int _k;  // number of key directions
    int _i;  // number of arnoldi iterations
    int _p;  // number of parallel initial vectors in arnoldi

    // profiling variables
    bool useProfiling;
    int aTransposeNonzeros;
    int keyDirMatrixNonzeros;

    CuspData(bool useCpu) : util(useCpu)
    {
        aTranspose = 0;
        keyDirMatrix = 0;

        vMatrix = 0;
        hMatrix = 0;
        vProjected = 0;

        reset();  // this resets all variables
    }

    ~CuspData() { reset(); }

    void reset()
    {
        if (aTranspose != 0)
        {
            delete aTranspose;
            aTranspose = 0;
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

        util.clearTimers();

        _n = 0;
        _k = 0;
        _i = 0;
        _p = 0;

        useProfiling = false;
        aTransposeNonzeros = 0;
        keyDirMatrixNonzeros = 0;
    }

    void setUseProfiling(bool enabled)
    {
        useProfiling = enabled;
        util.setUseProfiling(enabled);
    }

    // load a transpose, passed in as a csr matrix
    void loadATranspose(int dims, int* rowOffsets, int rowOffsetsLen, int* colInds, int colIndsLen,
                        FLOAT_TYPE* values, int valuesLen)
    {
        if (rowOffsetsLen != dims + 1)
            error("in loadATranspose(), rowOffsetsLen(%d) != dims(%d) + 1", rowOffsetsLen, dims);

        if (useProfiling)
            printf("loadATranspose() with sparse matrix size: %.2f MB (%d nonzeros)\n",
                   valuesLen * (8 + 4 + 4) / 1024.0 / 1024.0, valuesLen);

        util.tic("loadATranspose()");

        _n = dims;
        aTransposeNonzeros = valuesLen;

        HostIntArray1dView rowOffsetsView(rowOffsets, rowOffsets + rowOffsetsLen);
        HostIntArray1dView colIndsView(colInds, colInds + colIndsLen);
        HostFloatArray1dView valuesView(values, values + colIndsLen);

        HostCsrMatrixView view(_n, _n, valuesLen, rowOffsetsView, colIndsView, valuesView);

        if (aTranspose != 0)
        {
            delete aTranspose;
            aTranspose = 0;
        }

        util.tic("copy a_mat to gpu");
        aTranspose = new (std::nothrow) CsrMatrix(view);
        util.tic("copy a_mat to gpu");

        if (aTranspose == 0)
            error("memory allocation of aTranspose returned nullptr\n");

        util.toc("loadATranspose()");
        util.printTimers();
        util.clearTimers();
    }

    void loadKeyDirMatrixTranspose(int numKeyDirs, int* rowOffsets, int rowOffsetsLen, int* colInds,
                                   int colIndsLen, FLOAT_TYPE* values, int valuesLen)
    {
        if (rowOffsetsLen != _n + 1)
            error("in loadKeyDirMatrixTranspose(), rowOffsetsLen(%d) != dims(%d) + 1",
                  rowOffsetsLen, _n);

        if (useProfiling)
            printf("loadKeyDirMatrixTranspose() with sparse matrix size: %.2f MB (%d nonzeros)\n",
                   valuesLen * (8 + 4 + 4) / 1024.0 / 1024.0, valuesLen);

        util.tic("loadKeyDirMatrixTranspose()");

        _k = numKeyDirs;
        keyDirMatrixNonzeros = valuesLen;

        HostIntArray1dView rowOffsetsView(rowOffsets, rowOffsets + rowOffsetsLen);
        HostIntArray1dView colIndsView(colInds, colInds + colIndsLen);
        HostFloatArray1dView valuesView(values, values + colIndsLen);

        HostCsrMatrixView view(_n, _n, valuesLen, rowOffsetsView, colIndsView, valuesView);

        if (keyDirMatrix != 0)
        {
            delete keyDirMatrix;
            keyDirMatrix = 0;
        }

        util.tic("copy key_mat to gpu");
        keyDirMatrix = new (std::nothrow) CsrMatrix(view);
        util.tic("copy key_mat to gpu");

        if (keyDirMatrix == 0)
            error("memory allocation of keyDirMatrixTranspose returned nullptr\n");

        util.toc("loadKeyDirMatrixTranspose()");
        util.printTimers();
        util.clearTimers();
    }

    double getFreeMemoryMb()
    {
        unsigned long bytes = util.getFreeMemory();

        return bytes / 1024.0 / 1024.0;
    }

    // frees memory if it was previously allocated, returns false if memory error occurs
    bool preallocateMemory(int arnoldiIt, int numParallelInit)
    {
        if (_n == 0)
            error("preallocateMemory() called before loadMatrix() (_n==0)\n");

        if (_k == 0)
            error("preallocateMemory() called before loadKeyDirMatrix() (_k==0)\n");

        _i = arnoldiIt;
        _p = numParallelInit;

        // preallocate vMatrix, width = dims * iterations, height = numParInit
        if (vMatrix != 0)
        {
            delete vMatrix;
            vMatrix = 0;
        }

        unsigned long vMatrixSize = _p * _n * (_i + 1);
        vMatrix = new (std::nothrow) Array1d(vMatrixSize, 0);

        // preallocate hMatrix, numParInit * iterations * iterations
        if (hMatrix != 0)
        {
            delete hMatrix;
            hMatrix = 0;
        }

        unsigned long hMatrixSize = _p * _i * (_i + 1);
        hMatrix = new (std::nothrow) Array1d(hMatrixSize, 0);

        // preallocate vProjected
        if (vProjected != 0)
        {
            delete vProjected;
            vProjected = 0;
        }

        unsigned long vProjectedSize = _p * _n * (_i + 1);
        vProjected = new (std::nothrow) Array1d(vProjectedSize, 0);

        bool success = vMatrix != 0 && hMatrix != 0 && vProjected != 0;

        if (!success)
        {
            _i = 0;
            _p = 0;
        }

        return success;
    }

    void initParallelArnoldiV(int startDim, int numInitVecs)
    {
        util.tic("init parallel");

        if (startDim + numInitVecs > _n)
            error("initParallelArnoldiV called with startDim=%d, numInitVecs=%d, but dims=%d",
                  startDim, numInitVecs, _n);

        unsigned long rowWidth = _n * (_i + 1);

        // initialize each row
        for (unsigned long rowNum = 0; rowNum < numInitVecs; ++rowNum)
        {
            unsigned long rowOffset = rowNum * rowWidth;
            Array1dView initView = vMatrix->subarray(rowOffset, _n);

            cusp::blas::fill(initView, 0.0);
            initView[startDim + rowNum] = 1.0;
        }

        util.toc("init parallel");
    }

    // reads/writes from/to vMatrix, writes to hMatrix
    void runArnoldi(int iterations, int numInitVecs)
    {
        util.tic("arnoldi iteration");

        // Arnoldi parallel algorithm iteration
        for (int it = 1; it < iterations; it++)
        {
            // do all the multiplications up front
            // for cur_vec in xrange(num_init):
            //    vec = np.dot(prev_v[cur_vec, (cur_it-1)*size:cur_it*size], a_transpose)
            //    prev_v[cur_vec, cur_it*size:(cur_it+1)*size] = vec

            util.tic("arnoldi matrix vector multiply");

            for (int curInitVec = 0; curInitVec < numInitVecs; ++curInitVec)
            {
                unsigned long rowOffset = curInitVec * _n * iterations;
                unsigned long prevColOffset = _n * (it - 1);
                unsigned long curColOffset = _n * it;

                Array1dView vecView = vMatrix->subarray(rowOffset + prevColOffset, _n);
                Array1dView resultView = vMatrix->subarray(rowOffset + curColOffset, _n);

                // ????
                // cusp::multiply(A, V[j], V[j + 1]);

                // cusp::blas::dot()
                // v_1 = A * v_0
            }

            /*

            // multiply(a, b, c) -> c = a * b
            cusp::multiply(*curMatrix, V_all[j], V_all[j + 1]);

            unsigned long ops = aTransposeNonzeros * numInitVecs;

            util.toc("arnoldi matrix vector multiply", ops);

            for (int k = 0; k < numInitVec; k++)
            {
                // compute Hm-k

                tic();
                cusp::copy(H_all[k], Hmat_k);  // Load k-th Hmat matrix
                ioTime += toc();

                for (int i = 0; i <= j; i++)
                {
                    tic();
                    Hmat_k(i, j) = cusp::blas::dot(V_all[i].column(k), V_all[j + 1].column(k));
                    dotTime += toc();

                    tic();
                    // axpy(a, b, c) computes b += c*a
                    cusp::blas::axpy(V_all[i].column(k), V_all[j + 1].column(k), -Hmat_k(i, j));
                    axpyTime += toc();
                }

                Hmat_k(j + 1, j) = cusp::blas::nrm2(V_all[j + 1].column(k));

                if (Hmat_k(j + 1, j) < 1e-10)
                {
                    // an interesting problem: given a system matrix A, different initial vector can
                    // produce
                    // different number of iteration, i.e, the actual number of iteration
                    // To do arnoldi in parallel, we neglect the break condition as in the function
                    // arnoldi_initVectorPos
                    // We make all vector has the same number of iteration and equal to maxiter
                    // i.e., actual_numIter = maxiter (user input parameter)

                    if (mem == 1)
                    {
                        ;
                    }
                    else
                    {
                        printf(
                            "***Notice***: break condition of Arnoldi algorithm is neglected for "
                            "the initial vector V_%d \n",
                            k);
                        printf(
                            "***Notice***: the actual number of iteration corresponding to initial "
                            "vector V_%d is %d \n",
                            k, j + 1);
                        mem = 1;
                    }

                    Hmat_k(j + 1, j) = 0;
                    cusp::blas::scal(V_all[j + 1].column(k), float(0));
                    // break;
                }
                else
                    cusp::blas::scal(V_all[j + 1].column(k), float(1) / Hmat_k(j + 1, j));

                tic();
                cusp::copy(Hmat_k, H_all[k]);  // update the k-th Hmatrix
                ioTime += toc();
                }*/
        }

        util.toc("arnoldi iteration");
    }

    void arnoldiParallel(int startDim, double* result_H)
    {
        if (_n == 0)
            error("arnoldiParallel() called before loadMatrix() (_n==0)\n");

        if (_k == 0)
            error("arnoldiParrallel() called before loadKeyDirMatrix() (_k==0)\n");

        if (_i == 0 || _p == 0)
            error("arnoldiParrallel() called before preallocate() (_i==0 or _p==0)\n");

        if (startDim < 0 || startDim >= _n)
            error("invalid startDim in arnodli (%d dim system): %d", _n, startDim);

        util.tic("arnoldi parallel total");

        int parInitVecs = _p;

        if (startDim + parInitVecs > _n)
            parInitVecs = _n - startDim;

        initParallelArnoldiV(startDim, parInitVecs);

        runArnoldi(_i, parInitVecs);
        /*

        // copying H matrix to np.ndarray
        util.tic("copying H matrix to np.ndarray");

        // TODO: create a view of the h array... first implement arnoldi_parallel to figure out how
        // H is laid out in memory

        cusp::array2d<FLOAT_TYPE, cusp::host_memory> H(maxiter + 1, maxiter, 0);
        for (int k = 0; k < numInitVec; ++k)
        {
            cusp::copy(H_all[k], H);
            for (int i = 0; i < maxiter; ++i)
            {
                for (int l = 0; l < maxiter; ++l)
                    result_H[i * maxiter + l + k * maxiter * maxiter] = H(i, l);
            }
        }

        util.toc("copying H matrix to np.ndarray");

        // save all matrix Vm into V_all_final
        tic();
        for (int k = 0; k < numInitVec; k++)
            for (int i = 0; i < maxiter; i++)
                cusp::blas::copy(V_all[i].column(k), V_all_final[k].column(i));

        toc("save all matrix Vm into V_all_final");

        */

        util.toc("arnoldi parallel total");
        util.printTimers();
        util.clearTimers();
    }
};

CuspData<double, cusp::host_memory> cuspDataCpu(true);
CuspData<double, cusp::device_memory> cuspDataGpu(false);

extern "C" {
int hasGpu()
{
    return cuspDataGpu.util.hasGpu();
}

void reset()
{
    cuspDataCpu.reset();
    cuspDataGpu.reset();
}

////// CPU Version
void setUseProfilingCpu(int enabled)
{
    cuspDataCpu.setUseProfiling(enabled != 0);
}

void loadATransposeCpu(int dims, int* rowOffsets, int rowOffsetsLen, int* colInds, int colIndsLen,
                       double* values, int valuesLen)
{
    cuspDataCpu.loadATranspose(dims, rowOffsets, rowOffsetsLen, colInds, colIndsLen, values,
                               valuesLen);
}

void loadKeyDirMatrixTransposeCpu(int numKeyDirs, int* rowOffsets, int rowOffsetsLen, int* colInds,
                                  int colIndsLen, double* values, int valuesLen)
{
    cuspDataCpu.loadKeyDirMatrixTranspose(numKeyDirs, rowOffsets, rowOffsetsLen, colInds,
                                          colIndsLen, values, valuesLen);
}

double getFreeMemoryMbCpu()
{
    return cuspDataCpu.getFreeMemoryMb();
}

int preallocateMemoryCpu(int arnoldiIt, int numParallelInitVecs)
{
    return cuspDataCpu.preallocateMemory(arnoldiIt, numParallelInitVecs) ? 1 : 0;
}

void arnoldiParallelCpu(int startDim, double* resultH)
{
    cuspDataCpu.arnoldiParallel(startDim, resultH);
}

////// GPU Version
void setUseProfilingGpu(int enabled)
{
    cuspDataGpu.setUseProfiling(enabled != 0);
}

void loadATransposeGpu(int dims, int* rowOffsets, int rowOffsetsLen, int* colInds, int colIndsLen,
                       double* values, int valuesLen)
{
    cuspDataGpu.loadATranspose(dims, rowOffsets, rowOffsetsLen, colInds, colIndsLen, values,
                               valuesLen);
}

void loadKeyDirMatrixTransposeGpu(int numKeyDirs, int* rowOffsets, int rowOffsetsLen, int* colInds,
                                  int colIndsLen, double* values, int valuesLen)
{
    cuspDataGpu.loadKeyDirMatrixTranspose(numKeyDirs, rowOffsets, rowOffsetsLen, colInds,
                                          colIndsLen, values, valuesLen);
}

double getFreeMemoryMbGpu()
{
    return cuspDataGpu.getFreeMemoryMb();
}

int preallocateMemoryGpu(int arnoldiIterations, int numParallelInitVecs)
{
    return cuspDataGpu.preallocateMemory(arnoldiIterations, numParallelInitVecs) ? 1 : 0;
}

void arnoldiParallelGpu(int startDim, double* resultH)
{
    cuspDataGpu.arnoldiParallel(startDim, resultH);
}
}
