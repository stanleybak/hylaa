#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/multiply.h>
#include <cusp/multiply.h>
#include <cusp/print.h>

#include <cublas_v2.h>

#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/time.h>

typedef double FLOAT_TYPE;

// print an error and then exit
void error(const char *format, ...) {
  va_list args;
  fprintf(stdout, "Fatal Error: ");
  va_start(args, format);
  vfprintf(stdout, format, args);
  va_end(args);
  fprintf(stdout, "\n");

  exit(1);
}

long now() {
  struct timeval nowUs;

  if (gettimeofday(&nowUs, 0)) {
    perror("gettimeofday");
    exit(1);
  }

  return 1000000l * nowUs.tv_sec + nowUs.tv_usec;
}

void dot_product(cublasHandle_t cublasHandle,
                 cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view a,
                 cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view b,
                 cusp::array1d<FLOAT_TYPE, cusp::host_memory>::view resultView,
                 int resultIndex) {
  FLOAT_TYPE d = cusp::blas::dot(a, b);

  resultView[resultIndex] = d;
}

void dot_product_gpu1(
    cublasHandle_t cublasHandle,
    cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view a,
    cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view b,
    cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view resultView,
    int resultIndex) {
  // FLOAT_TYPE d = cusp::blas::dot(a, b);

  // resultView[resultIndex] = d;
  int count = a.size();

  double *x = thrust::raw_pointer_cast(&a[0]);
  double *y = thrust::raw_pointer_cast(&b[0]);
  double *result = thrust::raw_pointer_cast(&resultView[0]);

  if (cublasDdot(cublasHandle, count, x, 1, y, 1, result) !=
      CUBLAS_STATUS_SUCCESS)
    error("cublasDdot() failed");
}

void dot_product_gpu2(
    cublasHandle_t cublasHandle,
    cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view a,
    cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view b,
    cusp::array1d<FLOAT_TYPE, cusp::device_memory>::view resultView,
    int resultIndex) {
  // FLOAT_TYPE d = cusp::blas::dot(a, b);

  // resultView[resultIndex] = d;
  int count = a.size();

  double *x = thrust::raw_pointer_cast(&a[0]);
  double *y = thrust::raw_pointer_cast(&b[0]);

  thrust::device_ptr<double> d_x(x), d_y(y);

  unsigned long size = a.size();
  // inner_product implements a mathematical dot product
  resultView[resultIndex] = thrust::inner_product(d_x, d_x + size, d_y, 0.0);
}

int main(int argc, char **argv) {
  cublasHandle_t cublasHandle;

  cublasStatus_t status = cublasCreate(&cublasHandle);
  if (status != CUBLAS_STATUS_SUCCESS)
    error("cublasCreate() failed: %d", status);

  printf("If Init is slow, use 'sudo nvidia-smi -pm 1'\n");

  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

  long size = 1024 * 1024;

  if (argc > 1) {
    size = 1024 * 1024 * atoi(argv[1]);
  }

  printf("Using size %.2f million \n", size / 1000.0 / 1000.0);

  cusp::array1d<FLOAT_TYPE, cusp::host_memory> a(size);
  cusp::array1d<FLOAT_TYPE, cusp::host_memory> b(size);
  cusp::array1d<FLOAT_TYPE, cusp::host_memory> res(1);

  for (int i = 0; i < size; ++i) {
    a[i] = (i + 1.0) / size;
    b[i] = (size - i) / (float)size;
  }

  res[0] = 0;

  cusp::array1d<FLOAT_TYPE, cusp::device_memory> gpu_a(a);
  cusp::array1d<FLOAT_TYPE, cusp::device_memory> gpu_b(b);
  cusp::array1d<FLOAT_TYPE, cusp::device_memory> gpu_res(res);

  float ops = 2.0 * (float)size;

  ///////// cpu computation

  long start = now();
  dot_product(cublasHandle, a, b, res, 0);
  float diff = now() - start;

  printf("cpu result = %f\n", res[0]);
  printf("diff = %fms, gflops = %f\n", diff / 1000.0, ops / diff / 1000.0);

  ///////// gpu computation

  start = now();
  dot_product_gpu2(cublasHandle, gpu_a, gpu_b, gpu_res, 0);
  diff = now() - start;

  cusp::array1d<FLOAT_TYPE, cusp::host_memory> copied_gpu_res(gpu_res);

  printf("gpu result = %f\n", copied_gpu_res[0]);
  printf("diff = %fms, gflops = %f\n", diff / 1000.0, ops / diff / 1000.0);

  // cudaDeviceReset();

  return 0;
}
