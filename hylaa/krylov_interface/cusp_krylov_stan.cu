// Dung Tran & Stanley Bak
// Krylov subspace - based simulation using Gpu- Cusp / Cuda for sparse ode
// June 2017

#include <new>
#include <stdio.h>
#include <stdlib.h>
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
    typedef cusp::hyb_matrix<int, FLOAT_TYPE, MEMORY_TYPE> HybMatrix;

    typedef cusp::coo_matrix<int, FLOAT_TYPE, cusp::host_memory> HostCooMatrix;

   public:
    HybMatrix* aTranspose;
    HybMatrix* keyDirMatrix;

    // each row is the arnoldi v matrix from a single initial vector
    Array1d* vMatrix;
    Array1d* hMatrix;

    GpuUtil util;  // timers and other utility functions

    int aTransposeNonzeros;

    int dims;
    int keyDirHeight;
    bool useProfiling;

    int arnoldiIterations;
    int numTimeSteps;
    int numParallelInitVecs;

    CuspData(bool useCpu) : util(useCpu)
    {
        aTranspose = 0;
        keyDirMatrix = 0;
        vMatrix = 0;
        hMatrix = 0;

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

        util.clearTimers();

        aTransposeNonzeros = 0;

        dims = 0;
        keyDirHeight = 0;
        useProfiling = false;

        arnoldiIterations = 0;
        numTimeSteps = 0;
        numParallelInitVecs = 0;
    }

    void setUseProfiling(bool enabled)
    {
        useProfiling = enabled;
        util.setUseProfiling(enabled);
    }

    void loadATranspose(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries,
                        int nonZeroCount)
    {
        dims = w;
        aTransposeNonzeros = nonZeroCount;

        HostCooMatrix hostMatrix(w, h, nonZeroCount);

        if (useProfiling)
        {
            printf(
                "loadATranspose() called, estimated size in memory of sparse matrix: %.2f MB (%d "
                "nonzeros)\n",
                nonZeroCount * (8 + 4 + 4) / 1024.0 / 1024.0, nonZeroCount);
        }

        util.tic("loadATranspose()");

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

        if (aTranspose != 0)
        {
            delete aTranspose;
            aTranspose = 0;
        }

        aTranspose = new (std::nothrow) cusp::hyb_matrix<int, FLOAT_TYPE, MEMORY_TYPE>(hostMatrix);

        if (aTranspose == 0)
            error("memory allocation of aTranspose returned nullptr\n");

        util.toc("loadATranspose()");
        util.printTimers();
        util.clearTimers();
    }

    void loadKeyDirMatrix(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries,
                          int nonZeroCount)
    {  // Load key Direction Sparse Matrix to get a particular direction of simulation result
        util.tic("loadKeyDirMatrix()");

        if (aTranspose == 0)
            error("loadKeyDirMat called before loadATranspose\n");

        if (w != dims)
            error("loadKeyDirMat called with width %d, but system dimensions was %d\n", w, dims);

        keyDirHeight = h;

        HostCooMatrix hostKeyMatrix(w, h, nonZeroCount);

        // initialize key matrix entries on host
        int index = 0;

        for (int i = 0; i < nonZeroCount; ++i)
        {
            int row = nonZeroRows[i];
            int col = nonZeroCols[i];
            double val = nonZeroEntries[i];

            hostKeyMatrix.row_indices[index] = row;
            hostKeyMatrix.column_indices[index] = col;
            hostKeyMatrix.values[index++] = val;
        }

        if (keyDirMatrix != 0)
        {
            delete keyDirMatrix;
            keyDirMatrix = 0;
        }

        keyDirMatrix =
            new (std::nothrow) cusp::hyb_matrix<int, FLOAT_TYPE, MEMORY_TYPE>(hostKeyMatrix);

        if (keyDirMatrix == 0)
            error("memory allocation of keyDirMatrix returned nullptr\n");

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
    bool preallocateMemory(int arnoldiIt, int numSteps, int numParInitVecs)
    {
        if (dims == 0)
            error("preallocateMemory() called before loadMatrix() (dims==0)\n");

        arnoldiIterations = arnoldiIt;
        numTimeSteps = numSteps;
        numParallelInitVecs = numParInitVecs;

        // preallocate vMatrix, width = dims * iterations, height = numParInit
        if (vMatrix != 0)
        {
            delete vMatrix;
            vMatrix = 0;
        }

        unsigned long vMatrixSize = dims * arnoldiIterations * numParallelInitVecs;
        vMatrix = new (std::nothrow) Array1d(vMatrixSize, 0);

        // preallocate hMatrix, numParInit * iterations * iterations
        if (hMatrix != 0)
        {
            delete hMatrix;
            hMatrix = 0;
        }

        unsigned long hMatrixSize = numParallelInitVecs * arnoldiIterations * arnoldiIterations;
        hMatrix = new (std::nothrow) Array1d(hMatrixSize, 0);

        return vMatrix != 0 && hMatrix != 0;
    }

    void initParallelArnoldiV(int startDim, int numInitVecs)
    {
        util.tic("init parallel");

        if (startDim + numInitVecs > dims)
            error("initParallelArnoldiV called with startDim=%d, numInitVecs=%d, but dims=%d",
                  startDim, numInitVecs, dims);

        unsigned long rowWidth = dims * arnoldiIterations;

        // initialize each row
        for (unsigned long rowNum = 0; rowNum < numInitVecs; ++rowNum)
        {
            unsigned long rowOffset = rowNum * rowWidth;
            Array1dView initView = vMatrix->subarray(rowOffset, dims);

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
                unsigned long rowOffset = curInitVec * dims * iterations;
                unsigned long prevColOffset = dims * (it - 1);
                unsigned long curColOffset = dims * it;

                Array1dView vecView = vMatrix->subarray(rowOffset + prevColOffset, dims);
                Array1dView resultView = vMatrix->subarray(rowOffset + curColOffset, dims);

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
        util.tic("arnoldi parallel total");

        if (vMatrix == 0 || hMatrix == 0)
            error("preallocateMemory() must be sucessfully called before running arnoldi");

        if (startDim < 0 || startDim >= dims)
            error("invalid startDim in arnodli (%d dim system): %d", dims, startDim);

        int parInitVecs = numParallelInitVecs;

        if (startDim + parInitVecs > dims)
            parInitVecs = dims - startDim;

        initParallelArnoldiV(startDim, parInitVecs);

        runArnoldi(arnoldiIterations, parInitVecs);
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

    /*void getKeySimResult_parallel(double* expHt_e1_tuples, double* keySimResult_tuples)
    {

        // get Simulation result in specific direction defined by keyDirMatrix
        // for one initial vector we have:
        // SimResult = V*exp(H*t)*e1, (V,H) are matrices obtained from Arnoldi algorithm
        // KeySimResult = keyDirMatrix*SimResult

        // we get keySimResult for all initial vectors at one time, the result is saved in
        // keySimResult_tuples

        std::vector<cusp::array1d<FLOAT_TYPE, MEMORY_TYPE> > V_expHt_e1(
            numInitVec);  // contain all V*exp(H*t)*e1
        std::vector<cusp::array1d<FLOAT_TYPE, MEMORY_TYPE> > device_keySimResult_tuples(numInitVec);

        cusp::array1d<FLOAT_TYPE, MEMORY_TYPE> expHt_e1_col_i(numIteration);
        cusp::hyb_matrix<int, FLOAT_TYPE, MEMORY_TYPE> Vi;

        // Check consitency

        if (keyDirMatrix_h !=
            systemSize)  // check consistency between the key direction matrix and system dimension
        {
            printf(
                "\n The number of column of key direction matrix is inconsistent with the system "
                "dimension");
            toc("check consistency");
        }
        else
        {
            // compute key Simulation result in parallel

            tic();

            for (int i = 0; i < numInitVec; i++)
            {
                V_expHt_e1[i].resize(systemSize);
                device_keySimResult_tuples[i].resize(keyDirMatrix_w);

                for (int k = 0; k < numIteration; k++)
                {
                    expHt_e1_col_i[k] = expHt_e1_tuples[i * numIteration + k];
                }

                // cusp::multiply(V_all_final[i], expHt_e1_col_i,V_expHt_e1[i]); // compute
                // V*exp(H*t)*e1
                // cusp::multiply(*keyDirMatrix,V_expHt_e1[i],device_keySimResult_tuples[i]); //
                // compute keyDirMatrix * V * exp(H*t) * e1

                cusp::convert(V_all_final[i], Vi);
                cusp::multiply(Vi, expHt_e1_col_i, V_expHt_e1[i]);  // compute V*exp(H*t)*e1
                cusp::multiply(
                    *keyDirMatrix, V_expHt_e1[i],
                    device_keySimResult_tuples[i]);  // compute keyDirMatrix * V * exp(H*t) * e1

                // printf("the %d-th key simulation result corresponding to the %d-th initial vector
                // is: \n", i, i );
                // cusp::print(device_keySimResult_tuples[i]);
            }

            toc("Compute keySimResult in parallel");

            // copy keySimulation Result to np.array

            tic();

            for (int i = 0; i < numInitVec; i++)
                for (int j = 0; j < keyDirMatrix_w; j++)
                    keySimResult_tuples[i * keyDirMatrix_w + j] = device_keySimResult_tuples[i][j];
        }

        }*/
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
void loadATransposeCpu(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries,
                       int nonZeroCount)
{
    cuspDataCpu.loadATranspose(w, h, nonZeroRows, nonZeroCols, nonZeroEntries, nonZeroCount);
}

void loadKeyDirMatrixCpu(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries,
                         int nonZeroCount)
{
    cuspDataCpu.loadKeyDirMatrix(w, h, nonZeroRows, nonZeroCols, nonZeroEntries, nonZeroCount);
}

double getFreeMemoryMbCpu()
{
    return cuspDataCpu.getFreeMemoryMb();
}

int preallocateMemoryCpu(int arnoldiIt, int numTimeSteps, int numParallelInitVecs)
{
    return cuspDataCpu.preallocateMemory(arnoldiIt, numTimeSteps, numParallelInitVecs) ? 1 : 0;
}

void arnoldiParallelCpu(int startDim, double* resultH)
{
    cuspDataCpu.arnoldiParallel(startDim, resultH);
}

/*void getProjectedResultCpu(double* expHt_tuples, double* keySimResult_tuples)
{
    cuspDataCpu.getProjectedResult(expHt_tuples, keySimResult_tuples);
    }*/

////// GPU Version
void loadATransposeGpu(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries,
                       int nonZeroCount)
{
    cuspDataGpu.loadATranspose(w, h, nonZeroRows, nonZeroCols, nonZeroEntries, nonZeroCount);
}

void loadKeyDirMatrixGpu(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries,
                         int nonZeroCount)
{
    cuspDataGpu.loadKeyDirMatrix(w, h, nonZeroRows, nonZeroCols, nonZeroEntries, nonZeroCount);
}

double getFreeMemoryMbGpu()
{
    return cuspDataGpu.getFreeMemoryMb();
}

int preallocateMemoryGpu(int arnoldiIterations, int numTimeSteps, int numParallelInitVecs)
{
    return cuspDataGpu.preallocateMemory(arnoldiIterations, numTimeSteps, numParallelInitVecs) ? 1
}

void arnoldiParallelGpu(int startDim, double* resultH)
{
    cuspDataGpu.arnoldiParallel(startDim, resultH);
}

/*void getProjectedResultGpu(double* expHt_tuples, double* keySimResult_tuples)
{
    cuspDataGpu.getProjectedResult(expHt_tuples, keySimResult_tuples);
    }*/
}

int main()
{
    printf("ran\n");

    return 0;
}
