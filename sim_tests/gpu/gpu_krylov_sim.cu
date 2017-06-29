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
#include <cusp/blas.h>
#include <sys/time.h>

typedef double FLOAT_TYPE;
typedef cusp::host_memory MEMORY_TYPE;
//typedef cusp::device_memory MEMORY_TYPE;

// shared matrix in device memory
static cusp::hyb_matrix<int, FLOAT_TYPE, MEMORY_TYPE>* curMatrix = 0;
static std::vector< cusp::array1d<FLOAT_TYPE,MEMORY_TYPE> > V_;
static std::vector< cusp::array1d<FLOAT_TYPE, MEMORY_TYPE> > device_sim_result;
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
    
    curMatrix = new (std::nothrow) cusp::hyb_matrix<int, FLOAT_TYPE,MEMORY_TYPE>(hostMatrix);
        
    if (curMatrix == 0)
        error("allocation of heap-based csr matrix in device memory returned nullptr");
        
    toc("copying matrix to device memory");
}

int _arnoldi_initVector(double* init_vector, double* result_H, int size, int numIter)
{
    if (curMatrix == 0)
        error("loadMatrix must be called before running arnoldi algorithm");
    
    // initialize input vector
    tic();
    cusp::array1d<FLOAT_TYPE, cusp::host_memory> hostInitVec(size);
    
    for (int i = 0; i < size; ++i)
        hostInitVec[i] = init_vector[i];
    toc("creating hostVec initial vector");
    
    // copy initial vector to device memory
    tic();
    cusp::array1d<FLOAT_TYPE,MEMORY_TYPE> deviceInitVec(hostInitVec);
    toc("copying initial vetocctor to device memory");

    // system dimension 
    tic();
    int N = size;
    toc("get system dimension");

    // maximum number of Iteration of Arnoldi algorithm
    tic();
    int maxiter = std::min(N, numIter);
    toc("get maximum number of iteration of arnoldi algorithm");

    // create matrix H_ in device memory for iteration
    tic();	
    cusp::array2d<FLOAT_TYPE,MEMORY_TYPE> H_(maxiter + 1, maxiter, 0);
    toc("create matrix H_ in device memory for iteration");

    // returned matrix H after iteration -- Hm in the algorithm -- (m x m) matrix 
    tic();
    cusp::array2d<FLOAT_TYPE,MEMORY_TYPE> H(maxiter, maxiter); 
    toc("create returned matrix H after iteration -- Hm in the algorithm -- (m x m) matrix ");

    // create matrix V_ for iteration
    tic();
    V_.resize(maxiter+1);
    for (int i = 0; i < maxiter + 1; i++)
        V_[i].resize(N);
    toc("create matrix V_ for iteration");

    // copy initial vector into V_[0]
    tic(); 
    cusp::copy(deviceInitVec,V_[0]); 
    toc("copy initial vector into V_[0]"); 

    // compute beta 
    tic();
    FLOAT_TYPE beta = cusp::blas::nrm2(deviceInitVec);
    toc("compute beta");
   

    // normalize initial vector
    cusp::blas::scal(V_[0], float(1)/beta);

    // iteration
    tic();
    int j;
    for(j = 0; j < maxiter; j++)
    {
	cusp::multiply(*curMatrix, V_[j], V_[j + 1]);
	
	for(int i = 0; i <= j; i++)
	{
		H_(i,j) = cusp::blas::dot(V_[i], V_[j + 1]);

		cusp::blas::axpy(V_[i], V_[j + 1], -H_(i,j));
	}

		H_(j+1,j) = cusp::blas::nrm2(V_[j + 1]);

		if(H_(j+1,j) < 1e-10) break;

		cusp::blas::scal(V_[j + 1], float(1) / H_(j+1,j));

     }
     toc("iteration");


     // scale V_ with beta, i.e. beta*V_, used later for computing simulation trace
     tic();
     for(int i = 0; i < maxiter; i++)
     {
        cusp::blas::scal(V_[i],beta);
     }
     toc("scaling matrix V with beta");
     

     // get matrix H (m x m dimension)
     tic(); 
     for(int rowH=0;rowH < maxiter; rowH++)
	for(int colH = 0; colH <maxiter; colH++)
		H(rowH,colH) = H_(rowH,colH);
     toc("get matrix H -- (m x m) dimension");


     // copying H matrix to np.ndarray
     tic();
    
     for (int i = 0; i < numIter; ++i )
	for (int k = 0; k < numIter; ++k)
		result_H[i*numIter + k] = H_(i,k);       
     toc("copying H to np.ndarray");
     
     if(j < maxiter)
     return j+1;
     else return j;
}

int _arnoldi_initVectorPos(int basic_initVector_pos, double* result_H, int size, int numIter)
{
    if (curMatrix == 0)
        error("loadMatrix must be called before running arnoldi algorithm");
    
    // create initial basic vector on device memory
    tic();

    cusp::array1d<FLOAT_TYPE,MEMORY_TYPE> deviceInitVec(size,0);

    printf("size of system %d compare with basic intial vector position %d",size,basic_initVector_pos);
    deviceInitVec[basic_initVector_pos] = 1;
    toc("create initial basic vector on device memory based on its position, i.e. basic_initVector_pos");
    
    // system dimension 
    tic();
    int N = size;
    toc("get system dimension");

    // maximum number of Iteration of Arnoldi algorithm
    tic();
    int maxiter = std::min(N, numIter);
    toc("get maximum number of iteration of arnoldi algorithm");

    // create matrix H_ in device memory for iteration
    tic();	
    cusp::array2d<FLOAT_TYPE,MEMORY_TYPE> H_(maxiter + 1, maxiter, 0);
    toc("create matrix H_ in device memory for iteration");

    // returned matrix H after iteration -- Hm in the algorithm -- (m x m) matrix 
    tic();
    cusp::array2d<FLOAT_TYPE,MEMORY_TYPE> H(maxiter, maxiter); 
    toc("create returned matrix H after iteration -- Hm in the algorithm -- (m x m) matrix ");

    // create matrix V_ for iteration
    tic();
    V_.resize(maxiter+1);
    for (int i = 0; i < maxiter + 1; i++)
        V_[i].resize(N);
    toc("create matrix V_ for iteration");

    // copy initial vector into V_[0]
    tic(); 
    cusp::copy(deviceInitVec,V_[0]); 
    toc("copy initial vector into V_[0]"); 

    // compute beta 
    tic();
    FLOAT_TYPE beta = 1;
    //FLOAT_TYPE beta = cusp::blas::nrm2(deviceInitVec);
    toc("compute beta");
   

    // normalize initial vector
    cusp::blas::scal(V_[0], float(1)/beta);

    // iteration
    tic();
    int j;
    for(j = 0; j < maxiter; j++)
    {
	cusp::multiply(*curMatrix, V_[j], V_[j + 1]);
	
	for(int i = 0; i <= j; i++)
	{
		H_(i,j) = cusp::blas::dot(V_[i], V_[j + 1]);

		cusp::blas::axpy(V_[i], V_[j + 1], -H_(i,j));
	}

		H_(j+1,j) = cusp::blas::nrm2(V_[j + 1]);

		if(H_(j+1,j) < 1e-10) break;

		cusp::blas::scal(V_[j + 1], float(1) / H_(j+1,j));

     }
     toc("iteration");


     // scale V_ with beta, i.e. beta*V_, used later for computing simulation trace
     tic();
     for(int i = 0; i < maxiter; i++)
     {
        cusp::blas::scal(V_[i],beta);
     }
     toc("scaling matrix V with beta");
     

     // get matrix H (m x m dimension)
     tic(); 
     for(int rowH=0;rowH < maxiter; rowH++)
	for(int colH = 0; colH <maxiter; colH++)
		H(rowH,colH) = H_(rowH,colH);
     toc("get matrix H -- (m x m) dimension");


     // copying H matrix to np.ndarray
     tic();
    
     for (int i = 0; i < numIter; ++i )
	for (int k = 0; k < numIter; ++k)
		result_H[i*numIter + k] = H_(i,k);       
     toc("copying H to np.ndarray");

     if(j < maxiter)
     return j+1;
     else return j;
}


void _sim(double* matrix_Hf, double* sim_result, int size, int actual_numIter, int numStep)
{
     	  
    // copy matrix Hf to device memory
    tic();
    cusp::array2d<FLOAT_TYPE,MEMORY_TYPE> deviceMatrix_Hf(actual_numIter,numStep);
    for(int k=0; k< numStep ; k++)
       for(int i = 0; i < actual_numIter; i++)
          deviceMatrix_Hf(i,k) = matrix_Hf[i*numStep+k];
    toc("copying matrix Hf to device memory");

    // compute simulation result 
    tic();
    device_sim_result.resize(numStep);
    for (int i = 0; i < numStep; i++)
        device_sim_result[i].resize(size);        
    toc("create simulation result matrix");

    tic();
    for (int i=0; i< numStep; i++)
    {
        for(int k=0; k < actual_numIter; k++)
        {
	  cusp::blas::axpy(V_[k], device_sim_result[i], deviceMatrix_Hf(k,i));
        }

    }
    toc("compute simulation result");

    // copy simulation result to np.ndarray
    tic();
    for(int i = 0; i < numStep; i++)
       for(int k = 0; k < size; k++)
       {
		sim_result[i*numStep + k] = device_sim_result[i][k];		
		
       }
    toc("copy simulation result to np.ndarray");
           
}

int _hasGpu()
{
    int rv = 1;
    
    try
    {
        cusp::array1d<FLOAT_TYPE, cusp::host_memory> hostVec(10);
    
        for (int i = 0; i < 10; ++i)
            hostVec[i] = 0;

        cusp::array1d<FLOAT_TYPE,MEMORY_TYPE> deviceVec(hostVec);
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

int arnoldi_initVector(double* init_vector, double* result_H, int size, int numIter)
{
   return _arnoldi_initVector(init_vector, result_H, size, numIter);
}

int arnoldi_initVectorPos(int basic_initVector_pos, double* result_H, int size, int numIter)
{
   return _arnoldi_initVectorPos(basic_initVector_pos, result_H, size, numIter);
}

void sim(double* matrix_Hf, double* sim_result, int size, int actual_numIter, int numStep)\
{
    _sim(matrix_Hf, sim_result,  size, actual_numIter, numStep);
}

}
