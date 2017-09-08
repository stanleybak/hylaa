#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/multiply.h>
#include <cusp/multiply.h>
#include <cusp/print.h>

#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

typedef double FLOAT_TYPE;

// print an error and then exit
void error(const char* format, ...)
{
    va_list args;
    fprintf(stdout, "Fatal Error: ");
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
    fprintf(stdout, "\n");

    exit(1);
}

void dot_product(cublasHandle_t cublasHandle, cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view a,
                 cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view b,
                 cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view resultView, int resultIndex)
{
    FLOAT_TYPE d = cusp::blas::dot(a, b);

    resultView[resultIndex] = d;
}

void dot_product_gpu(cublasHandle_t cublasHandle,
                 cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view a,
                 cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view b,
                 cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view resultView, int resultIndex)
{
    // FLOAT_TYPE d = cusp::blas::dot(a, b);

    // resultView[resultIndex] = d;
    int count = a.size();

    double *x = thrust::raw_pointer_cast(&a[0]);
    double *y = thrust::raw_pointer_cast(&b[0]);
    double *result = thrust::raw_pointer_cast(&resultView[0]);

    if (cublasDdot(cublasHandle, count, x, 1, y, 1, result) != CUBLAS_STATUS_SUCCESS)
        error("cublasDdot() failed");
}

int main()
{
    cublasHandle_t cublasHandle;
    
    cublasStatus_t status = cublasCreate(&cublasHandle);
    if (status != CUBLAS_STATUS_SUCCESS)
        error("cublasCreate() failed: %d", status);

    cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
    
    cusp::array1d<FLOAT_TYPE, cusp::host_memory> a(3);
    cusp::array1d<FLOAT_TYPE, cusp::host_memory> b(3);
    cusp::array1d<FLOAT_TYPE, cusp::host_memory> res(1);
    
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;
    
    b[0] = 3;
    b[1] = 2;
    b[2] = 1;
    
    res[0] = 0;
 
    cusp::array1d<FLOAT_TYPE, cusp::device_memory> gpu_a(a);
    cusp::array1d<FLOAT_TYPE, cusp::device_memory> gpu_b(b);
    cusp::array1d<FLOAT_TYPE, cusp::device_memory> gpu_res(res);
   
    ///////// cpu computation
    
    //dot_product(cublasHandle, a, b, res, 0);
    
    printf("cpu result = %f\n", res[0]); // 10
    
    ///////// gpu computation
    
     dot_product_gpu(cublasHandle, gpu_a, gpu_b, gpu_res, 0);
     
     cusp::array1d<FLOAT_TYPE, cusp::host_memory> copied_gpu_res(gpu_res);
     
     printf("gpu result = %f\n", copied_gpu_res[0]); // 10
    
    
    
     //cudaDeviceReset();
     
     return 0;
}
