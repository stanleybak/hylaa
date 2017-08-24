// Dung Tran & Stanley Bak
// Krylov subspace - based simulation using Gpu- Cusp / Cuda for sparse ode
// June 2017

#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <sys/time.h>
#include "gpu_util.h"

template <class FLOAT_TYPE, class MEMORY_TYPE>
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

    typedef cusp::csr_matrix<int, FLOAT_TYPE, MEMORY_TYPE> CsrMatrix;

    typedef cusp::csr_matrix<int, FLOAT_TYPE, cusp::host_memory> HostCsrMatrix;
    typedef typename HostCsrMatrix::view HostCsrMatrixView;

   public:
    GpuUtil util;  // timers and other utility functions

    CsrMatrix* aMatrix;
    CsrMatrix* keyDirMatrix;

    Array1d* vMatrix;     // p * [(i+1) * n]
    Array1d* hMatrix;     // p * [(i+1) * i]
    Array1d* vProjected;  // p * [k * (i+1)]

    int _n;  // number of dimensions in the system
    int _k;  // number of key directions
    int _i;  // number of arnoldi iterations
    int _p;  // number of parallel initial vectors in arnoldi

    // profiling variables
    bool useProfiling;
    int aMatrixNonzeros;
    int keyDirMatrixNonzeros;

    CuspData(bool useCpu) : util(useCpu)
    {
        aMatrix = 0;
        keyDirMatrix = 0;

        vMatrix = 0;
        hMatrix = 0;
        vProjected = 0;

        reset();  // this resets all variables
    }

    ~CuspData() { reset(); }

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

        util.clearTimers();

        _n = 0;
        _k = 0;
        _i = 0;
        _p = 0;

        useProfiling = false;
        aMatrixNonzeros = 0;
        keyDirMatrixNonzeros = 0;
    }

    void setUseProfiling(bool enabled)
    {
        useProfiling = enabled;
        util.setUseProfiling(enabled);
    }

    // load A matrix, passed in as a csr matrix
    void loadAMatrix(int w, int h, int* rowOffsets, int rowOffsetsLen, int* colInds, int colIndsLen,
                     FLOAT_TYPE* values, int valuesLen)
    {
        if (w != h)
            error("loadAMatrix() expected square A matrix, got w=%d, h=%d", w, h);

        if (useProfiling)
            printf("loadAMatrix() with sparse matrix size: %.2f MB (%d nonzeros)\n",
                   valuesLen * (8 + 4 + 4) / 1024.0 / 1024.0, valuesLen);

        util.tic("loadAMatrix()");

        _n = w;
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

        util.tic("copy a_mat to gpu");
        aMatrix = new (std::nothrow) CsrMatrix(view);
        util.toc("copy a_mat to gpu");

        if (aMatrix == 0)
            error("memory allocation of aMatrix returned nullptr\n");

        util.toc("loadAMatrix()");
        util.printTimers();
        util.clearTimers();
    }

    // load key dir matrix, passed in as a csr matrix
    void loadKeyDirMatrix(int w, int h, int* rowOffsets, int rowOffsetsLen, int* colInds,
                          int colIndsLen, FLOAT_TYPE* values, int valuesLen)
    {
        if (w != _n)
            error("in loadKeyDirMatrix() width (%d) to equal dims (%d)", w, _n);

        if (useProfiling)
            printf("loadKeyDirMatrix() with dense matrix size: %.2f MB\n",
                   w * h * (8 + 4 + 4) / 1024.0 / 1024.0);

        util.tic("loadKeyDirMatrix()");

        _k = h;
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
        util.toc("copy key_mat to gpu");

        if (keyDirMatrix == 0)
            error("memory allocation of keyDirMatrix() returned nullptr\n");

        util.toc("loadKeyDirMatrix()");
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
            error("preallocateMemory() called before loadAMatrix() (_n==0)\n");

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

        unsigned long vProjectedSize = _p * _k * (_i + 1);
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

        // fill with zeros
        cusp::blas::fill(*vMatrix, 0.0);

        unsigned long rowWidth = _n * (_i + 1);

        for (unsigned long rowNum = 0; rowNum < numInitVecs; ++rowNum)
        {
            // initialize the "1.0" in each row
            unsigned long rowOffset = rowNum * rowWidth;

            (*vMatrix)[rowOffset + startDim + rowNum] = 1.0;
        }

        util.toc("init parallel");
    }

    // reads/writes from/to vMatrix, writes to hMatrix
    void runArnoldi(int iterations, int numInitVecs)
    {
        util.tic("arnoldi");

        // Arnoldi parallel algorithm iteration
        for (int it = 1; it < iterations; it++)
        {
            util.tic("sparse matrix vector multiply total");
            // do all the multiplications up front
            for (int curInitVec = 0; curInitVec < numInitVecs; ++curInitVec)
            {
                unsigned long rowOffset = curInitVec * _n * iterations;
                unsigned long prevColOffset = _n * (it - 1);
                unsigned long curColOffset = _n * it;

                Array1dView vecView = vMatrix->subarray(rowOffset + prevColOffset, _n);
                Array1dView resultView = vMatrix->subarray(rowOffset + curColOffset, _n);

                util.tic("sparse matrix vector multiply");
                cusp::multiply(*aMatrix, vecView, resultView);
                util.toc("sparse matrix vector multiply", 2 * aMatrixNonzeros);
            }
            util.toc("sparse matrix vector multiply total", 2 * aMatrixNonzeros * numInitVecs);

            util.tic("dots total");
            // do all the dot products (using dense matrix multiplication with cusp::blas::gemv)
            for (int curInitVec = 0; curInitVec < numInitVecs; ++curInitVec)
            {
                unsigned long rowOffset = curInitVec * _n * iterations;
                unsigned long curColOffset = _n * it;

                Array1dView vecView = vMatrix->subarray(rowOffset + curColOffset, _n);

                printf("cur_vec:\n");
                cusp::print(vecView);

                Array1dView matView1d = vMatrix->subarray(rowOffset, curColOffset);
                Array2dView matView2d = make_array2d_view(it, _n, _n, matView1d, cusp::row_major());

                printf("matView2d:\n");
                cusp::print(matView2d);

                rowOffset = curInitVec * (_i + 1) * _i;
                curColOffset = (it - 1) * (_i + 1);
                Array1dView resultView = hMatrix->subarray(rowOffset + curColOffset, it);

                printf("result-view:\n");
                cusp::print(resultView);

                util.tic("dots");
                cusp::blas::gemv(matView2d, vecView, resultView);
                util.toc("dots", 2 * _n * it);
            }
            util.toc("dots total", 2 * _n * it * numInitVecs);

            // multiply(a, b, c) -> c = a * b
            // cusp::multiply(*curMatrix, V_all[j], V_all[j + 1]);

            /*

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

        util.toc("arnoldi");
    }

    void arnoldiParallel(int startDim, double* resultH, int sizeResultH, double* resultPV,
                         int sizeResultPV)
    {
        if (_n == 0)
            error("arnoldiParallel() called before loadAMatrix() (_n==0)\n");

        if (_k == 0)
            error("arnoldiParrallel() called before loadKeyDirMatrix() (_k==0)\n");

        if (_i == 0 || _p == 0)
            error("arnoldiParrallel() called before preallocate() (_i==0 or _p==0)\n");

        // check expected results sizes
        int expectedH = _i * (_i + 1);
        int expectedPV = _i * _p * _k;

        if (sizeResultH != expectedH)
            error("Wrong size for resultH with i = %d. Got %d, expected %d.", _i, sizeResultH,
                  expectedH);

        if (sizeResultPV != expectedPV)
            error("Wrong size for resultPV with (i, p, k) = (%d, %d, %d). Got %d, expected %d.", _i,
                  _p, _k, sizeResultPV, expectedPV);

        if (startDim < 0 || startDim >= _n)
            error("invalid startDim in arnoldi (%d dim system): %d", _n, startDim);

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

// as csr matrix
void loadAMatrixCpu(int w, int h, int* rowOffsets, int rowOffsetsLen, int* colInds, int colIndsLen,
                    double* values, int valuesLen)
{
    cuspDataCpu.loadAMatrix(w, h, rowOffsets, rowOffsetsLen, colInds, colIndsLen, values,
                            valuesLen);
}

void loadKeyDirMatrixCpu(int w, int h, int* rowOffsets, int rowOffsetsLen, int* colInds,
                         int colIndsLen, double* values, int valuesLen)
{
    cuspDataCpu.loadKeyDirMatrix(w, h, rowOffsets, rowOffsetsLen, colInds, colIndsLen, values,
                                 valuesLen);
}

double getFreeMemoryMbCpu()
{
    return cuspDataCpu.getFreeMemoryMb();
}

int preallocateMemoryCpu(int arnoldiIt, int numParallelInitVecs)
{
    return cuspDataCpu.preallocateMemory(arnoldiIt, numParallelInitVecs) ? 1 : 0;
}

void arnoldiParallelCpu(int startDim, double* resultH, int sizeResultH, double* resultPV,
                        int sizeResultPV)
{
    cuspDataCpu.arnoldiParallel(startDim, resultH, sizeResultH, resultPV, sizeResultPV);
}

////// GPU Version
void setUseProfilingGpu(int enabled)
{
    cuspDataGpu.setUseProfiling(enabled != 0);
}

// as csr matrix
void loadAMatrixGpu(int w, int h, int* rowOffsets, int rowOffsetsLen, int* colInds, int colIndsLen,
                    double* values, int valuesLen)
{
    cuspDataGpu.loadAMatrix(w, h, rowOffsets, rowOffsetsLen, colInds, colIndsLen, values,
                            valuesLen);
}

// as csr matrix
void loadKeyDirMatrixGpu(int w, int h, int* rowOffsets, int rowOffsetsLen, int* colInds,
                         int colIndsLen, double* values, int valuesLen)
{
    cuspDataGpu.loadKeyDirMatrix(w, h, rowOffsets, rowOffsetsLen, colInds, colIndsLen, values,
                                 valuesLen);
}

double getFreeMemoryMbGpu()
{
    return cuspDataGpu.getFreeMemoryMb();
}

int preallocateMemoryGpu(int arnoldiIterations, int numParallelInitVecs)
{
    return cuspDataGpu.preallocateMemory(arnoldiIterations, numParallelInitVecs) ? 1 : 0;
}

void arnoldiParallelGpu(int startDim, double* resultH, int sizeResultH, double* resultPV,
                        int sizeResultPV)
{
    cuspDataGpu.arnoldiParallel(startDim, resultH, sizeResultH, resultPV, sizeResultPV);
}
}
