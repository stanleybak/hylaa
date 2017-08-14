// Dung Tran
// An interface
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
#include <cusp/blas/blas.h>
#include <sys/time.h>

// timing shared variable
long lastTicUs = 0;

void error(const char* msg)
{
    printf("Fatal Error: %s\n", msg);
    exit(1);             
}

int _hasGpu()
{
    int rv = 1;

    try
    {
        cusp::array1d<double, cusp::host_memory> hostVec(10);

        for (int i = 0; i < 10; ++i)
            hostVec[i] = 0;

        cusp::array1d<double, cusp::device_memory> deviceVec(hostVec);
    }
    catch(std::exception &e)
    {
        printf("hasGpu() Failed: %s\n", e.what());
        rv = 0;
    }

    return rv;
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

// returns the us elaspsed
long toc()
{
    struct timeval now;

    if(gettimeofday( &now, 0))
        error("gettimeofday");

    long nowUs = 1000000 * now.tv_sec + now.tv_usec;
    long dif = nowUs - lastTicUs;

    return dif;
}

template <class FLOAT_TYPE, class MEMORY_TYPE>
class CuspData
{
public:
    //static int choose_GPU = 0; // choose_GPU == 1 means that user choose to use GPU, if not, using CPU
    // shared matrix in device memory
    cusp::hyb_matrix<int, FLOAT_TYPE, MEMORY_TYPE>* curMatrix;
    std::vector< cusp::array2d<FLOAT_TYPE,MEMORY_TYPE, cusp::column_major> > V_all_final; // contain all n- Vm matrix
    std::vector< cusp::array1d<FLOAT_TYPE,MEMORY_TYPE> > V_;

    std::vector< cusp::array1d<FLOAT_TYPE, MEMORY_TYPE> > device_sim_result;
    std::vector< cusp::array2d<FLOAT_TYPE,MEMORY_TYPE> >  device_keySimResult_tuples; // contain all keySimResult
    cusp::hyb_matrix<int, FLOAT_TYPE, MEMORY_TYPE>* keyDirMatrix;
    int systemSize;
    int keyDirMatrix_w;
    int keyDirMatrix_h;
    int numIteration;
    int numInitVec;

    CuspData()
    {
        curMatrix = 0;
        keyDirMatrix = 0;
        systemSize = 0;
        keyDirMatrix_w = 0;
        keyDirMatrix_h = 0;
        numIteration = 0;
        numInitVec = 0; 
    }

    ~CuspData()
    {
        reset();
    }

    void reset()
    {
        if (keyDirMatrix != 0)
        {
            delete keyDirMatrix;
            keyDirMatrix = 0;
        }

        if (curMatrix != 0)
        {
            delete curMatrix;
            curMatrix = 0;
        }

        // empty all variables
        V_.clear();
        V_all_final.clear();

        device_sim_result.clear();
        device_keySimResult_tuples.clear(); // contain all keySimResult
    }

    void _loadMatrix(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries, int nonZeroCount)
    {
        systemSize = w;

        cusp::coo_matrix<int, FLOAT_TYPE, cusp::host_memory> hostMatrix(w, h, nonZeroCount);

        //printf("loadMatrix() called, estimated size in memory of sparse matrix: %.2f MB (%d nonzeros)\n", 
        //    nonZeroCount * (8 + 4 + 4) / 1024.0 / 1024.0, nonZeroCount);

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

        if (curMatrix != 0)
        {
            delete curMatrix;
            curMatrix = 0;
        }

        curMatrix = new (std::nothrow) cusp::hyb_matrix<int, FLOAT_TYPE,MEMORY_TYPE>(hostMatrix);

        if (curMatrix == 0)
            error("allocation of heap-based csr matrix in device memory returned nullptr");
    }

    void _loadKeyDirMatrix(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries, int nonZeroCount)
    {   // Load key Direction Sparse Matrix to get a particular direction of simulation result

        tic();
        keyDirMatrix_w = w;
        keyDirMatrix_h = h;

        cusp::coo_matrix<int, FLOAT_TYPE, cusp::host_memory> hostKeyMatrix(w, h, nonZeroCount);

        //printf("loadKeyDirMatrix() called, estimated size in memory of sparse matrix: %.2f MB (%d nonzeros)\n", 
        //    nonZeroCount * (8 + 4 + 4) / 1024.0 / 1024.0, nonZeroCount);

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

        toc("creating host coo key matrix");

        tic();
        if (keyDirMatrix != 0)
        {
            delete keyDirMatrix;
            keyDirMatrix = 0;
        }

        keyDirMatrix = new (std::nothrow) cusp::hyb_matrix<int, FLOAT_TYPE,MEMORY_TYPE>(hostKeyMatrix);

        if (keyDirMatrix == 0)
            error("allocation of heap-based csr key matrix in device memory returned nullptr");

        toc("copying key matrix to device memory");

    }

    void _arnoldi_parallel2(int start_pos, int final_pos, int numIter,double* result_H)
    {   
        if (curMatrix == 0)
            error("loadMatrix must be called before running arnoldi algorithm");

        if (final_pos >= systemSize)
            error("start_pos + final_pos >= systemSize (too many initial vectors)");

        numIteration = numIter;
        numInitVec  = final_pos - start_pos + 1; 

        // maximum number of Iteration of Arnoldi algorithm
        int maxiter = std::min(systemSize, numIter);

        // create matrix V_all to contain all matrix V: V_all = [V0 V1 ...Vm]
        // V0 = [V0_1 ... V0_n] is (n x n) matrix containing all initial vectors, n = numInitVec 
        // Vi = [Vi_1 ... Vi_n] is (n x n) matrix containing all i-th vectors in step i of Arnoldi algorithm

        std::vector< cusp::array2d<FLOAT_TYPE,MEMORY_TYPE, cusp::column_major> > V_all; // use to compute n- Vm matrix in parallel
        V_all.resize(maxiter+1);

        // n = numInitVec
        // create matrix V_all_final to contain all matrix V; V_all_final = [Vm_0 Vm_2 ...Vm_(n-1)]
        // Vm_0 is the matrix (n x m) V (obtained from Arnoldi algorithm) that corresponds to the 0-th initial vector  
        // Vm_i is the (n x m) matrix V (obtained from Arnoldi algorithm) that corresponds to the i-th initial vector

        V_all_final.resize(numInitVec);

        // create matrix H_all to contain all matrix H: H_all = [Hm_1 Hm_2 ...Hm_n]
        // Hm_1, Hm_2 , ... Hm_n are m x m matrices, Hm_i is conresponding to the initial vector i 

        std::vector< cusp::array2d<FLOAT_TYPE,MEMORY_TYPE> > H_all; // contain all n Hm matrix
        H_all.resize(numInitVec+1);

         // create initial basic vector V_all[0] = n-dimension identity mat

        cusp::array2d<FLOAT_TYPE, MEMORY_TYPE> Hmat_k(maxiter+1,maxiter,0);
        cusp::array2d<FLOAT_TYPE,MEMORY_TYPE> Imat(systemSize,numInitVec,0);
        cusp::array2d<FLOAT_TYPE,MEMORY_TYPE,cusp::column_major> Vm(systemSize,maxiter,0);

        for (int i = 0; i< numInitVec; i++){
            cusp::copy(Hmat_k,H_all[i]); // initialize H_all[i]
            cusp::copy(Vm,V_all_final[i]); // initialize V_finall_all[k]
        }
        for (int i = 0; i < maxiter+1; i++)
            cusp::copy(Imat,V_all[i]);

        int pos = start_pos;
        for (int i = 0; i < numInitVec; i++){
            Imat(pos,i) = 1;
            pos = pos+1;
        }

        cusp::copy(Imat,V_all[0]);

        // Arnoldi parallel algorithm iteration

        int j;
        int mem = 0; // memorize where break condition happens

        printf("running Arnoldi Algorithm in parallel ...\n");
        long multiplyTime = 0;
        long dotTime = 0;
        long axpyTime = 0;
        long ioTime = 0;

        for (j = 0; j < maxiter; j++){

            tic();
            // multiply(a, b, c) -> c = a * b
            cusp::multiply(*curMatrix, V_all[j], V_all[j+1]);
            multiplyTime += toc("arnoldi multiply");   

            for(int k = 0; k < numInitVec; k++){
                // compute Hm-k

                tic();
                cusp::copy(H_all[k], Hmat_k); // Load k-th Hmat matrix
                ioTime += toc();

                
                for(int i = 0; i <= j; i++){

                    tic();
                    Hmat_k(i,j) = cusp::blas::dot(V_all[i].column(k), V_all[j+1].column(k));
                    dotTime += toc();

                    tic();
                    // axpy(a, b, c) computes b += c*a
                    cusp::blas::axpy(V_all[i].column(k), V_all[j+1].column(k), -Hmat_k(i,j));
                    axpyTime += toc();
                }


                Hmat_k(j+1,j) = cusp::blas::nrm2(V_all[j+1].column(k));

                if(Hmat_k(j+1,j) < 1e-10) {

                    // an interesting problem: given a system matrix A, different initial vector can produce
                    // different number of iteration, i.e, the actual number of iteration
                    // To do arnoldi in parallel, we neglect the break condition as in the function arnoldi_initVectorPos
                    // We make all vector has the same number of iteration and equal to maxiter
                    // i.e., actual_numIter = maxiter (user input parameter)


                    if (mem == 1){
                        ;
                    }
                    else {
                        printf("***Notice***: break condition of Arnoldi algorithm is neglected for the initial vector V_%d \n", k);
                        printf("***Notice***: the actual number of iteration corresponding to initial vector V_%d is %d \n", k, j+1);
                        mem = 1;
                    }

                    Hmat_k(j+1,j) = 0;
                    cusp::blas::scal(V_all[j+1].column(k),float(0));
                    //break;
                }
                else  cusp::blas::scal(V_all[j+1].column(k), float(1) / Hmat_k(j+1,j));           

                tic();
                cusp::copy(Hmat_k,H_all[k]); // update the k-th Hmatrix           
                ioTime += toc();
            }

        }

        printf("Multiply time: %.3f seconds\n", multiplyTime / 1000.0 / 1000.0);
        printf("Dot time: %.3f seconds\n", dotTime / 1000.0 / 1000.0);
        printf("Axpy time: %.3f seconds\n", axpyTime / 1000.0 / 1000.0);
        printf("I/O time: %.3f seconds\n", ioTime / 1000.0 / 1000.0);

          // copying H matrix to np.ndarray
         tic();
         
         cusp::array2d<FLOAT_TYPE, cusp::host_memory> H(maxiter+1,maxiter,0);
         for (int k = 0; k< numInitVec; ++k){
             cusp::copy(H_all[k], H);
             for (int i = 0; i < maxiter; ++i){
                 for(int l = 0; l < maxiter; ++l)
                     result_H[i*maxiter + l + k*maxiter*maxiter] = H(i,l);
             }
         }

         toc("copying H matrix to np.ndarray");

         // save all matrix Vm into V_all_final
         tic();
         for (int k = 0; k < numInitVec; k++)     
             for(int i = 0; i < maxiter; i++)
                 cusp::blas::copy(V_all[i].column(k),V_all_final[k].column(i));

         toc("save all matrix Vm into V_all_final");

    }

    
    void _arnoldi_parallel(int start_pos, int final_pos, int numIter,double* result_H)
    {   
        if (curMatrix == 0)
            error("loadMatrix must be called before running arnoldi algorithm");

        if (final_pos >= systemSize)
            error("start_pos + final_pos >= systemSize (too many initial vectors)");

        numIteration = numIter;
        numInitVec  = final_pos - start_pos + 1; 

        // maximum number of Iteration of Arnoldi algorithm
        int maxiter = std::min(systemSize, numIter);

        // create matrix V_all to contain all matrix V: V_all = [V0 V1 ...Vm]
        // V0 = [V0_1 ... V0_n] is (n x n) matrix containing all initial vectors, n = numInitVec 
        // Vi = [Vi_1 ... Vi_n] is (n x n) matrix containing all i-th vectors in step i of Arnoldi algorithm

        std::vector< cusp::array2d<FLOAT_TYPE,MEMORY_TYPE, cusp::column_major> > V_all; // use to compute n- Vm matrix in parallel
        V_all.resize(maxiter+1);

        // n = numInitVec
        // create matrix V_all_final to contain all matrix V; V_all_final = [Vm_0 Vm_2 ...Vm_(n-1)]
        // Vm_0 is the matrix (n x m) V (obtained from Arnoldi algorithm) that corresponds to the 0-th initial vector  
        // Vm_i is the (n x m) matrix V (obtained from Arnoldi algorithm) that corresponds to the i-th initial vector

        V_all_final.resize(numInitVec);

        // create matrix H_all to contain all matrix H: H_all = [Hm_1 Hm_2 ...Hm_n]
        // Hm_1, Hm_2 , ... Hm_n are m x m matrices, Hm_i is conresponding to the initial vector i 

        std::vector< cusp::array2d<FLOAT_TYPE,MEMORY_TYPE,cusp::row_major> > H_all; // contain all n Hm matrix
        H_all.resize(maxiter);

         // create initial basic vector V_all[0] = n-dimension identity mat

        cusp::array2d<FLOAT_TYPE, MEMORY_TYPE> Hmat_k(maxiter+1,numInitVec,0);
        cusp::array2d<FLOAT_TYPE,MEMORY_TYPE> Imat(systemSize,numInitVec,0);
        cusp::array2d<FLOAT_TYPE,MEMORY_TYPE,cusp::column_major> Vm(systemSize,maxiter,0);
        
        for (int i = 0; i< numInitVec; i++){
            cusp::copy(Hmat_k,H_all[i]); // initialize H_all[i]
            cusp::copy(Vm,V_all_final[i]); // initialize V_finall_all[k]
        }
        for (int i = 0; i < maxiter+1; i++)
            cusp::copy(Imat,V_all[i]);

        int pos = start_pos;
        for (int i = 0; i < numInitVec; i++){
            Imat(pos,i) = 1;
            pos = pos+1;
        }
        
        cusp::copy(Imat,V_all[0]);

        // Arnoldi parallel algorithm iteration

        int j;
        int mem = 0; // memorize where break condition happens

        printf("running Arnoldi Algorithm in parallel ...\n");
        long multiplyTime = 0;
        long dotTime = 0;
        long axpyTime = 0;
        long ioTime = 0;

        cusp::array1d<FLOAT_TYPE,MEMORY_TYPE> nrm_array(numInitVec);
        cusp::array1d<FLOAT_TYPE,MEMORY_TYPE> dot_array(numInitVec); 
        for (j = 0; j < maxiter; j++){

            tic();
            // multiply(a, b, c) -> c = a * b
             cusp::multiply(*curMatrix, V_all[j], V_all[j+1]);
             multiplyTime += toc("arnoldi multiply");   
                
              for(int i = 0; i <= j; i++){
                  // cusp::blas::dot(V_all[i], V_all[j+1]);
                    // cusp::blas::copy(dot_array, H_all[j].row(i));
                }

            //    for(int i = 0; i <= j; i++){
                    // cusp::blas::axpy(V_all[i], V_all[j+1], H_all[j]);                   
            //    }
                
                // nrm_array = cusp::blas::nrm2(V_all[j+1]); 

            //    for (int k = 0; k < numInitVec; k++){
            //        if (nrm_array[k] < 1e-10){
            //            nrm_array[k] = 0;
                        //cusp::blas::scal(V_all[j+1].column(k), float(0));
            //        }
                    // else cusp::blas::scal(V_all[j+1].column(k), float(1) / nrm_array[k]);
            
            //    }

                //  cusp::blas::copy(nrm_array, H_all[j].row(j+1));

        }

        // copy and save matrice H and V

         // copying H matrix to np.ndarray

                
        // cusp::array2d<FLOAT_TYPE, cusp::host_memory,cusp::column_major> H(maxiter+1,maxiter,0);

        // for (int k = 0; k < numInitVec; ++k)
        //     for(int i = 0; i < maxiter; ++i){
                 // cusp::blas::copy(H_all[k].column(i),H.column(i));
        //         for(int l = 0; l < maxiter; ++l)                     
                     //    result_H[i*maxiter + l + k*maxiter*maxiter] = H(i,l);
        //     }
           
        
         // save all matrix Vm into V_all_final
         //tic();
         //for (int k = 0; k < numInitVec; k++)     
         //    for(int i = 0; i < maxiter; i++)
                 //   cusp::blas::copy(V_all[i].column(k),V_all_final[k].column(i));

        //toc("save all matrix Vm into V_all_final");

         
        

    }


    void _getKeySimResult_parallel(double* expHt_e1_tuples, double* keySimResult_tuples)
    {
        // get Simulation result in specific direction defined by keyDirMatrix
        // for one initial vector we have:
        // SimResult = V*exp(H*t)*e1, (V,H) are matrices obtained from Arnoldi algorithm
        // KeySimResult = keyDirMatrix*SimResult

        // we get keySimResult for all initial vectors at one time, the result is saved in keySimResult_tuples   

        std::vector< cusp::array1d <FLOAT_TYPE,MEMORY_TYPE> > V_expHt_e1(numInitVec); // contain all V*exp(H*t)*e1
        std::vector< cusp::array1d <FLOAT_TYPE, MEMORY_TYPE> > device_keySimResult_tuples(numInitVec);

        cusp::array1d<FLOAT_TYPE,MEMORY_TYPE > expHt_e1_col_i(numIteration);
        cusp::hyb_matrix<int, FLOAT_TYPE, MEMORY_TYPE> Vi;

        // Check consitency

        if (keyDirMatrix_h != systemSize) // check consistency between the key direction matrix and system dimension
            {
                 printf("\n The number of column of key direction matrix is inconsistent with the system dimension");
                 toc("check consistency");
            }
        else{
            // compute key Simulation result in parallel

            tic();

            for (int i = 0; i < numInitVec; i++){
                V_expHt_e1[i].resize(systemSize);
                device_keySimResult_tuples[i].resize(keyDirMatrix_w);

                for(int k = 0; k < numIteration; k++){

                    expHt_e1_col_i[k] = expHt_e1_tuples[i*numIteration + k];

                }

                // cusp::multiply(V_all_final[i], expHt_e1_col_i,V_expHt_e1[i]); // compute V*exp(H*t)*e1
                //cusp::multiply(*keyDirMatrix,V_expHt_e1[i],device_keySimResult_tuples[i]); // compute keyDirMatrix * V * exp(H*t) * e1           
                
                cusp::convert(V_all_final[i],Vi);
                cusp::multiply(Vi, expHt_e1_col_i,V_expHt_e1[i]); // compute V*exp(H*t)*e1
                cusp::multiply(*keyDirMatrix,V_expHt_e1[i],device_keySimResult_tuples[i]); // compute keyDirMatrix * V * exp(H*t) * e1 

                // printf("the %d-th key simulation result corresponding to the %d-th initial vector is: \n", i, i );
                // cusp::print(device_keySimResult_tuples[i]);
            }

            toc("Compute keySimResult in parallel");

            // copy keySimulation Result to np.array

            tic();

            for (int i = 0; i < numInitVec; i++)
                for (int j = 0; j < keyDirMatrix_w; j++)        
                    keySimResult_tuples[i*keyDirMatrix_w + j] = device_keySimResult_tuples[i][j];
          }


    }
};

CuspData <double, cusp::host_memory> cuspDataCpu;
CuspData <double, cusp::device_memory> cuspDataGpu;

extern "C"
{
int hasGpu()
{
    return _hasGpu();
}

void choose_GPU_or_CPU(char* msg)
{
    printf("void choose_GPU_or_CPU(char* msg) unimplemented!\n");
}

void reset()
{
    cuspDataCpu.reset();
    cuspDataGpu.reset();
}

////// CPU Version
void loadMatrixCpu(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries, int nonZeroCount)
{
    cuspDataCpu._loadMatrix(w, h, nonZeroRows, nonZeroCols, nonZeroEntries, nonZeroCount);
}

void loadKeyDirMatrixCpu(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries, int nonZeroCount)
{
    cuspDataCpu._loadKeyDirMatrix(w, h, nonZeroRows, nonZeroCols, nonZeroEntries, nonZeroCount);
}   

void arnoldiParallelCpu( int start_pos, int final_pos, int numIter, double* result_H)
{
    cuspDataCpu._arnoldi_parallel(start_pos, final_pos, numIter,result_H);
}

    
void getKeySimResultParallelCpu( double* expHt_tuples, double* keySimResult_tuples)    
{
    cuspDataCpu._getKeySimResult_parallel( expHt_tuples, keySimResult_tuples);   
}

////// GPU Version
void loadMatrixGpu(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries, int nonZeroCount)
{
    cuspDataGpu._loadMatrix(w, h, nonZeroRows, nonZeroCols, nonZeroEntries, nonZeroCount);
}

void loadKeyDirMatrixGpu(int w, int h, int* nonZeroRows, int* nonZeroCols, double* nonZeroEntries, int nonZeroCount)
{
    cuspDataGpu._loadKeyDirMatrix(w, h, nonZeroRows, nonZeroCols, nonZeroEntries, nonZeroCount);
}   

void arnoldiParallelGpu( int start_pos, int final_pos, int numIter, double* result_H)
{
    cuspDataGpu._arnoldi_parallel(start_pos, final_pos, numIter,result_H);
}
    
void getKeySimResultParallelGpu( double* expHt_tuples, double* keySimResult_tuples)    
{
    cuspDataGpu._getKeySimResult_parallel( expHt_tuples, keySimResult_tuples);   
}

}
