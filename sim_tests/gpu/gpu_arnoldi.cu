// Dunf Tran
// GPU Arnoldi Algorithm interface using Cusp / Cuda
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

void _arnoldi(double* init_vector, double* result_V, double* result_H, int size, int numIter)
{
    if (curMatrix == 0)
        error("loadMatrix must be called before multiply");
    
    // initialize input vector
    tic();
    cusp::array1d<FLOAT_TYPE, cusp::host_memory> hostInitVec(size);
    
    for (int i = 0; i < size; ++i)
        hostInitVec[i] = init_vector[i];
    toc("creating hostVec initial vector");
    
    // copy initial vector to device memory
    tic();
    cusp::array1d<FLOAT_TYPE,cusp::device_memory> deviceInitVec(hostInitVec);
    toc("copying initial vector to device memory");

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
    cusp::array2d<FLOAT_TYPE,cusp::device_memory> H_(maxiter + 1, maxiter, 0);
    toc("create matrix H_ in device memory for iteration");

    // returned matrix H after iteration -- Hm in the algorithm -- (m x m) matrix 
    tic();
    cusp::array2d<FLOAT_TYPE,cusp::device_memory> H(maxiter, maxiter); 
    toc("create returned matrix H after iteration -- Hm in the algorithm -- (m x m) matrix ");

    // create matrix V_ for iteration
    tic();
    std::vector< cusp::array1d<FLOAT_TYPE,cusp::device_memory> > V_(maxiter + 1);
    for (size_t i = 0; i < maxiter + 1; i++)
        V_[i].resize(N);
    toc("create matrix V_ for iteration");

    // returned matrix V after iteration -- Vm in the algorithm -- (N x m) matrix
    tic();
    cusp::array2d<FLOAT_TYPE,cusp::device_memory> V(N,maxiter);
    toc("create returned matrix V after iteration -- Vm in the algorithm -- (N x m) matrix"); 

    // copy initial vector into V_[0]
    tic(); 
    cusp::copy(deviceInitVec,V_[0]); 
    toc("copy initial vector into V_[0]"); 

    // compute beta 
    tic();
    FLOAT_TYPE beta = float(1) / cusp::blas::nrm2(deviceInitVec);
    toc("compute beta");

    // normalize initial vector
    cusp::blas::scal(V_[0], beta);

    // iteration
    tic();
    size_t j;
    for(j = 0; j < maxiter; j++)
    {
	cusp::multiply(*curMatrix, V_[j], V_[j + 1]);
	cusp::print(V_[j]); 

	for(size_t i = 0; i <= j; i++)
	{
		H_(i,j) = cusp::blas::dot(V_[i], V_[j + 1]);

		cusp::blas::axpy(V_[i], V_[j + 1], -H_(i,j));
	}

		H_(j+1,j) = cusp::blas::nrm2(V_[j + 1]);

		if(H_(j+1,j) < 1e-10) break;

		cusp::blas::scal(V_[j + 1], float(1) / H_(j+1,j));

     }
     toc("iteration");	

     // get matrix H (m x m dimension)
     tic(); 
     for(size_t rowH=0;rowH < maxiter; rowH++)
	for(size_t colH = 0; colH <maxiter; colH++)
		H(rowH,colH) = H_(rowH,colH);
     toc("get matrix H -- (m x m) dimension");

     // get matrix V (N x m dimension) 
     tic();
     cusp::array1d<FLOAT_TYPE,cusp::device_memory> x1(N);	

     for(size_t colV = 0; colV < maxiter; colV++)
     {	cusp::copy(V_[colV],x1);
	cusp::print(x1);		
	for(size_t rowV=0;rowV < N; rowV++)
		V(rowV, colV) = x1[rowV];
     }
     toc("get matrix V -- (N x m) dimension");

     // copy result to host memory	
     tic();
     cusp::array2d<FLOAT_TYPE, cusp::host_memory> result_V_Host(V);
     cusp::array2d<FLOAT_TYPE, cusp::host_memory> result_H_Host(H);
     toc("copying result to host memory");

     // copying to np.ndarray
     tic();
     for (int i = 0; i < N; ++i)
        for (int k = 0; k < numIter; ++k)
        	result_V[i,k] = result_V_Host(i,k);
    
     for (int i = 0; i < numIter; ++i )
	for (int k = 0; k < numIter; ++k)
		result_H[i,k] = result_H_Host(i,k);    
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

void arnoldi(double* init_vector, double* result_V, double* result_H, int size, int numIter)\
{
    _arnoldi(init_vector,  result_V,  result_H, size,
    numIter);
}

}
