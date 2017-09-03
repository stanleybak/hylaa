////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

// Utilities and system includes
#include <assert.h>
#include <stdlib.h>
//#include <helper_string.h>  // helper for shared functions common to CUDA SDK
// samples
#include <stdio.h>
#include <sys/time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
//#include <helper_functions.h>

#ifndef min
#define min(a, b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a, b) ((a > b) ? a : b)
#endif

typedef struct Matrix {
  double *data;
  unsigned int w;
  unsigned int h;

  void print(bool rowMajor) {

    if (rowMajor) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; ++x) {
          printf("%f ", data[y * w + x]);
        }

        printf("\n");
      }
    } else {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; ++x) {
          printf("(%d)%f ", x * h + y, data[x * h + y]);
        }

        printf("\n");
      }
    }
  }
} Matrix;

// compute C = A * B... assume row major
void matrixMultCPU(Matrix *A, Matrix *B, Matrix *C) {
  if (A->w != B->h) {
    printf("A <-> B size mismatch in matrixMult\n");
    exit(1);
  } else if (A->h != C->h) {
    printf("C <-> A size mismatch in matrixMult\n");
    exit(1);
  } else if (B->w != C->w) {
    printf("C <-> B size mismatch in matrixMult\n");
    exit(1);
  }

  for (unsigned int bCol = 0; bCol < B->w; ++bCol) {
    for (unsigned int aRow = 0; aRow < A->h; ++aRow) {
      double sum = 0;

      for (unsigned int aCol = 0; aCol < A->w; ++aCol) {
        double a = A->data[aRow * A->w + aCol];
        double b = B->data[aCol * B->w + bCol];
        sum += a * b;
      }

      printf("sum with bCol = %d and aRow = %d is %f... storing in %d\n", bCol,
             aRow, sum, aRow * C->w + bCol);

      C->data[aRow * C->w + bCol] = (double)sum;
    }
  }
}

void gpuMatrixVecMult(cublasHandle_t handle, Matrix *dA, Matrix *dB,
                      Matrix *dC) {
  const double alpha = 1.0f;
  const double beta = 0.0f;

  if (dB->w != 1) {
    printf("error: expected b.width == 1\n");
    exit(1);
  }

  // m = width of B
  // k = height of B
  // n = height of A
  int m = dA->h;
  int n = dA->w;

  int lda = m;

  printf("m, n = %d, %d\n", m, n);
  printf("lda = %d\n", lda);

  /*
cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
       int m, int n,
       const double          *alpha,
       const double          *A, int lda,
       const double          *x, int incx,
       const double          *beta,
       double          *y, int incy)
   */

  cublasStatus_t ret = cublasDgemv(handle, CUBLAS_OP_N, // not transposed
                                   m, n,                // m and n
                                   &alpha,              // 1.0
                                   dA->data, lda,       // a matrix
                                   dB->data, 1,         // b vector
                                   &beta,               // 0.0
                                   dC->data, 1);

  if (ret != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
    exit(EXIT_FAILURE);
  }
}

void gpuMatrixMult(cublasHandle_t handle, Matrix *dA, Matrix *dB, Matrix *dC) {
  const double alpha = 1.0f;
  const double beta = 0.0f;

  // we switch to column major by computing B^T * A^T which results in C^T

  // in Dgemm, A: m x k , B: k x n and C: m x n

  // m = width of B
  // k = height of B
  // n = height of A
  int m = dA->h;
  int n = dB->w;
  int k = dA->w;

  int lda = m;
  int ldb = k;
  int ldc = m;

  printf("m, n, k = %d, %d, %d\n", m, n, k);
  printf("lda, ldb, ldc = %d, %d, %d\n", lda, ldb, ldc);

  cublasStatus_t ret =
      cublasDgemm(handle, CUBLAS_OP_N,
                  CUBLAS_OP_N,    // transa, transb n = don't transpose
                  m, n, k,        // m, n, k
                  &alpha,         // 1.0
                  dA->data, lda,  // first term
                  dB->data, ldb,  // second term
                  &beta,          // 0.0
                  dC->data, ldc); // result

  if (ret != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
    exit(EXIT_FAILURE);
  }
}

long now() {
  struct timeval nowUs;

  if (gettimeofday(&nowUs, 0)) {
    perror("gettimeofday");
    exit(1);
  }

  return 1000000l * nowUs.tv_sec + nowUs.tv_usec;
}

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions (in addition to helper_cuda.h)

void inline checkError(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("%s", msg);
    exit(EXIT_FAILURE);
  }
}
// end of CUDA Helper Functions

// Allocates a matrix with random double entries.
void randomInit(double *data, int size) {
  for (int i = 0; i < size; ++i)
    data[i] = rand() / (double)RAND_MAX;
}

void printDiff(double *data1, double *data2, int width, int height,
               int iListLength, double fListTol) {
  printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
  int i, j, k;
  int error_count = 0;

  for (j = 0; j < height; j++) {
    /*if (error_count < iListLength)
    {
        printf("\n  Row %d:\n", j);
        }*/

    for (i = 0; i < width; i++) {
      k = j * width + i;
      double fDiff = fabs(data1[k] - data2[k]);

      if (fDiff > fListTol) {
        if (error_count < iListLength) {
          printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j,
                 data1[k], data2[k], fDiff);
        }

        error_count++;
      }
    }
  }

  printf(" \n  Total Errors = %d\n", error_count);
}

void initializeCUDA(int argc, char **argv, int &devID) {
  // By default, we use device 0, otherwise we override the device ID based on
  // what is provided at the command line
  cudaError_t error;
  devID = 0;

  /*if (checkCmdLineFlag(argc, (const char **)argv, "device"))
  {
      devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
      error = cudaSetDevice(devID);

      if (error != cudaSuccess)
      {
          printf("cudaSetDevice returned error code %d, line(%d)\n", error,
  __LINE__);
          exit(EXIT_FAILURE);
      }
  }*/

  // get number of SMs on this GPU
  error = cudaGetDevice(&devID);

  if (error != cudaSuccess) {
    printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    exit(EXIT_FAILURE);
  }

  /*if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
  {
      iSizeMultiple = getCmdLineArgumentInt(argc, (const char **)argv,
  "sizemult");
  }*/

  cudaDeviceProp deviceProp;

  error = cudaGetDeviceProperties(&deviceProp, devID);

  if (error != cudaSuccess) {
    printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error,
           __LINE__);
    exit(EXIT_FAILURE);
  }

  /*printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
         deviceProp.name, deviceProp.major, deviceProp.minor);

  // use a larger block size for Fermi and above
  int block_size = (deviceProp.major < 2) ? 16 : 32;

  matrix_size.uiWA = 2 * block_size * iSizeMultiple;
  matrix_size.uiHA = 4 * block_size * iSizeMultiple;
  matrix_size.uiWB = 2 * block_size * iSizeMultiple;
  matrix_size.uiHB = 4 * block_size * iSizeMultiple;
  matrix_size.uiWC = 2 * block_size * iSizeMultiple;
  matrix_size.uiHC = 4 * block_size * iSizeMultiple;

  //     unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;

  // C = ( A ) * ( B ) +  C
  //  op ( A ) m × k , op ( B ) k × n and C m × n ,

  matrix_size.uiWA = 1000000; // k
  matrix_size.uiHA = 10;      // m

  matrix_size.uiWB = 1;       // n
  matrix_size.uiHB = 1000000; // k

  matrix_size.uiWC = 10; // n
  matrix_size.uiHC = 1;  // m

  printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n", matrix_size.uiWA,
         matrix_size.uiHA, matrix_size.uiWB, matrix_size.uiHB, matrix_size.uiWC,
         matrix_size.uiHC);*/
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(int argc, char **argv, int devID) {
  cudaDeviceProp deviceProp;
  cudaError_t error;

  error = cudaGetDeviceProperties(&deviceProp, devID);

  if (error != cudaSuccess) {
    printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error,
           __LINE__);
    exit(EXIT_FAILURE);
  }

  Matrix A, B, C, C2;

  A.h = 10;
  A.w = 1000 * 1000;

  B.h = A.w;
  B.w = 1;

  C.h = A.h;
  C.w = B.w;

  /*A.w = 3;
  A.h = 2;

  B.w = 1;
  B.h = 3;

  C.w = 1;
  C.h = 2;*/

  C2.w = C.w;
  C2.h = C.h;

  // allocate host memory for matrices A and B
  unsigned int size_A = A.w * A.h;
  unsigned int mem_size_A = sizeof(double) * size_A;
  A.data = (double *)malloc(mem_size_A);

  unsigned int size_B = B.w * B.h;
  unsigned int mem_size_B = sizeof(double) * size_B;
  B.data = (double *)malloc(mem_size_B);

  unsigned int size_C = C.w * C.h;
  unsigned int mem_size_C = sizeof(double) * size_C;
  C.data = (double *)malloc(mem_size_C);

  C2.data = (double *)malloc(mem_size_C);

  double flopsPerMatrixMul = 2.0 * (double)A.w * (double)A.h * (double)B.w;

  // set seed for rand()
  srand(2006);

  // initialize host memory
  randomInit(A.data, size_A);
  randomInit(B.data, size_B);

  /*A.data[0] = 1;
  A.data[1] = 4;
  A.data[2] = 2;
  A.data[3] = 5;
  A.data[4] = 3;
  A.data[5] = 6;

  B.data[0] = 1;
  B.data[1] = 2;
  B.data[2] = 3;*/

  for (int y = 0; y < C.h; ++y)
    for (int x = 0; x < C.w; ++x) {
      int i = x + C.w * y;
      C.data[i] = C2.data[i] = 0;
    }

  // compute reference solution
  /*printf("Computing result using host CPU...\n");
  long start = now();

  matrixMultCPU(&A, &B, &C);

  printf("done.\n");
  long diffMs = (now() - start) / 1000;

  double msecPerMatrixMul = diffMs;
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

  // printf("\n C Flat: %f %f %f %f\n", C.data[0], C.data[1], C.data[2],
  //       C.data[3]);

  printf("\nCPU Performance= %.2f GFlop/s, Time= %.3f msec, Size= "
         "%.0f Ops\n",
         gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);*/

  // allocate device memory
  Matrix dA, dB, dC;
  dA.w = A.w;
  dA.h = A.h;

  dB.w = B.w;
  dB.h = B.h;

  dC.w = C.w;
  dC.h = C.h;

  // allocate host memory for the result

  error = cudaMalloc((void **)&dA.data, mem_size_A);

  if (error != cudaSuccess) {
    printf("cudaMalloc d_A returned error code %d, line(%d)\n", error,
           __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaMalloc((void **)&dB.data, mem_size_B);

  if (error != cudaSuccess) {
    printf("cudaMalloc d_B returned error code %d, line(%d)\n", error,
           __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaMalloc((void **)&dC.data, mem_size_C);

  if (error != cudaSuccess) {
    printf("cudaMalloc d_C returned error code %d, line(%d)\n", error,
           __LINE__);
    exit(EXIT_FAILURE);
  }

  // copy host memory to device
  error = cudaMemcpy(dA.data, A.data, mem_size_A, cudaMemcpyHostToDevice);

  if (error != cudaSuccess) {
    printf("cudaMemcpy d_A h_A returned error code %d, line(%d)\n", error,
           __LINE__);
    exit(EXIT_FAILURE);
  }

  error = cudaMemcpy(dB.data, B.data, mem_size_B, cudaMemcpyHostToDevice);

  if (error != cudaSuccess) {
    printf("cudaMemcpy d_B h_B returned error code %d, line(%d)\n", error,
           __LINE__);
    exit(EXIT_FAILURE);
  }

  // setup execution parameters
  // dim3 threads(block_size, block_size);
  // dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

  // create and start timer
  printf("Computing result using CUBLAS...\n");

  // CUBLAS version 2.0
  {
    cublasHandle_t handle;

    cublasStatus_t ret;

    ret = cublasCreate(&handle);

    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__);
      exit(EXIT_FAILURE);
    }

    ret = CUBLAS_STATUS_SUCCESS;
    // Perform warmup operation with cublas
    // ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB,
    // matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A,
    // matrix_size.uiWA, &beta, d_C, matrix_size.uiWA);
    // ret = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB,
    // matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A,
    // matrix_size.uiWA, &beta, d_C, matrix_size.uiWA);
    // cublasDgemm

    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
      exit(EXIT_FAILURE);
    }

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess) {
      fprintf(stderr, "Failed to create start event (error code %s)!\n",
              cudaGetErrorString(error));
      exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess) {
      fprintf(stderr, "Failed to create stop event (error code %s)!\n",
              cudaGetErrorString(error));
      exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess) {
      fprintf(stderr, "Failed to record start event (error code %s)!\n",
              cudaGetErrorString(error));
      exit(EXIT_FAILURE);
    }

    int nIter = 10;

    for (int i = 0; i < nIter; ++i) {
      gpuMatrixVecMult(handle, &dA, &dB, &dC);
      //      gpuMatrixMult(handle, &dA, &dB, &dC);
    }

    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    printf("done.\n");

    if (error != cudaSuccess) {
      fprintf(stderr, "Failed to record stop event (error code %s)!\n",
              cudaGetErrorString(error));
      exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess) {
      fprintf(stderr,
              "Failed to synchronize on the stop event (error code %s)!\n",
              cudaGetErrorString(error));
      exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess) {
      fprintf(stderr,
              "Failed to get time elapsed between events (error code %s)!\n",
              cudaGetErrorString(error));
      exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    double msecPerMatrixMul = msecTotal / nIter;
    double gigaFlops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("GPU Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
           gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

    // copy result from device to host

    // error = cudaMemcpy(dB.data, B.data, mem_size_B, cudaMemcpyHostToDevice);
    error = cudaMemcpy(C2.data, dC.data, mem_size_C, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) {
      printf("cudaMemcpy h_CUBLAS d_C returned error code %d, line(%d)\n",
             error, __LINE__);
      exit(EXIT_FAILURE);
    }

    /*printf("A:\n");
    A.print(false);

    printf("\nB:\n");
    B.print(false);
    printf("\ncuda C:\n");
    C2.print(false);
    printf("\n");*/

    checkError(cublasDestroy(handle), "cublasDestroy() error!\n");
  }

  // check result (CUBLAS)
  // bool resCUBLAS = true; // sdkCompareL2fe(reference, h_CUBLAS,
  // size_C,
  // 1.0e-6f);

  // if (resCUBLAS != true)
  //{
  // printDiff(C.data, C2.data, C.w, C.h, 10, 1e-6);
  //}

  // clean up memory
  free(A.data);
  free(B.data);
  free(C.data);
  free(C2.data);
  cudaFree(dA.data);
  cudaFree(dB.data);
  cudaFree(dC.data);

  cudaDeviceReset();

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("[Matrix Multiply CUBLAS] - Starting...\n");
  printf("If init is slow use: sudo nvidia-smi -pm 1\n");

  int devID = 0, sizeMult = 5;

  initializeCUDA(argc, argv, devID);

  int matrix_result = matrixMultiply(argc, argv, devID);

  exit(matrix_result);
}
