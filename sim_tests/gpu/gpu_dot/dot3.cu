/* File:     dot3.cu
 * Purpose:  Implement dot product on a gpu using cuda.  This version
 *           uses a binary tree reduction in which we attempt to reduce
 *           thread divergence.  It also uses shared memory to store
 *           intermediate results.  Assumes both threads_per_block and
 *           blocks_per_grid are powers of 2.
 *
 * Compile:  nvcc  -arch=sm_21 -o dot3 dot3.cu
 * Run:      ./dot3 <n> <blocks> <threads_per_block>
 *              n is the vector length
 *
 * Input:    None
 * Output:   Result of dot product of a collection of random floats
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_BLOCK_SZ 512
#define THREADS_PER_BLOCK 512

long now() {
  struct timeval nowUs;

  if (gettimeofday(&nowUs, 0)) {
    perror("gettimeofday");
    exit(1);
  }

  return 1000000l * nowUs.tv_sec + nowUs.tv_usec;
}

/*-------------------------------------------------------------------
 * Function:    Dev_dot  (kernel)
 * Purpose:     Implement a dot product of floating point vectors
 *              using atomic operations for the global sum
 * In args:     x, y, n
 * Out arg:     z
 *
 */
__global__ void Dev_dot(float x[], float y[], float z[], int n) {
  /* Use tmp to store products of vector components in each block */
  /* Can't use variable dimension here                            */
  __shared__ float tmp[MAX_BLOCK_SZ];
  int t = blockDim.x * blockIdx.x + threadIdx.x;
  int loc_t = threadIdx.x;

  if (t < n)
    tmp[loc_t] = x[t] * y[t];
  __syncthreads();

  /* This uses a tree structure to do the addtions */
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (loc_t < stride)
      tmp[loc_t] += tmp[loc_t + stride];
    __syncthreads();
  }

  /* Store the result from this cache block in z[blockIdx.x] */
  if (threadIdx.x == 0) {
    z[blockIdx.x] = tmp[0];
  }
} /* Dev_dot */

/*-------------------------------------------------------------------
 * Host code
 */
void Get_args(int argc, char *argv[], int *n_p, int *threads_per_block_p,
              int *blocks_per_grid_p);
void Setup(int n, int blocks, float **x_h_p, float **y_h_p, float **x_d_p,
           float **y_d_p, float **z_d_p);
float Serial_dot(float x[], float y[], int n);
void Free_mem(float *x_h, float *y_h, float *x_d, float *y_d, float *z_d);
float Dot_wrapper(float x_d[], float y_d[], float z_d[], int n, int blocks,
                  int threads);
float Dot_wrapper2(float x_d[], float y_d[], float z_d[], int n, int blocks,
                   int threads);

/*-------------------------------------------------------------------
 * main
 */
int main(int argc, char *argv[]) {
  int n, threads_per_block, blocks_per_grid;
  float *x_h, *y_h, dot = 0;
  float *x_d, *y_d, *z_d;
  double start, finish; /* Only used on host */

  Get_args(argc, argv, &n, &threads_per_block, &blocks_per_grid);
  Setup(n, blocks_per_grid, &x_h, &y_h, &x_d, &y_d, &z_d);

  start = now();
  dot = Dot_wrapper2(x_d, y_d, z_d, n, blocks_per_grid, threads_per_block);
  finish = now();

  printf("The dot product as computed by cuda is: %e\n", dot);
  printf("Elapsed time for cuda = %f us\n", finish - start);
  printf("gflops = %f\n", 2 * n / (finish - start) / 1000.0);

  start = now();
  dot = Serial_dot(x_h, y_h, n);
  finish = now();
  printf("The dot product as computed by cpu is: %e\n", dot);
  printf("Elapsed time for cpu = %f us\n", finish - start);
  printf("gflops = %f\n", 2 * n / (finish - start) / 1000.0);

  Free_mem(x_h, y_h, x_d, y_d, z_d);

  return 0;
} /* main */

/*-------------------------------------------------------------------
 * Function:  Get_args
 * Purpose:   Get and check command line args.  If there's an error
 *            quit.
 */
void Get_args(int argc, char *argv[], int *n_p, int *threads_per_block_p,
              int *blocks_per_grid_p) {

  if (argc != 2) {
    fprintf(stderr, "usage: %s <vector size>\n", argv[0]);
    exit(0);
  }
  *n_p = strtol(argv[1], NULL, 10);

  *blocks_per_grid_p = *n_p / THREADS_PER_BLOCK;
  *threads_per_block_p = THREADS_PER_BLOCK;
} /* Get_args */

/*-------------------------------------------------------------------
 * Function:  Setup
 * Purpose:   Allocate and initialize host and device memory
 */
void Setup(int n, int blocks, float **x_h_p, float **y_h_p, float **x_d_p,
           float **y_d_p, float **z_d_p) {
  int i;
  size_t size = n * sizeof(float);

  /* Allocate input vectors in host memory */
  *x_h_p = (float *)malloc(size);
  *y_h_p = (float *)malloc(size);

  /* Initialize input vectors */
  srandom(1);
  for (i = 0; i < n; i++) {
    (*x_h_p)[i] = random() / ((double)RAND_MAX);
    (*y_h_p)[i] = random() / ((double)RAND_MAX);
  }

  /* Allocate vectors in device memory */
  cudaMalloc(x_d_p, size);
  cudaMalloc(y_d_p, size);
  cudaMalloc(z_d_p, blocks * sizeof(float));

  /* Copy vectors from host memory to device memory */
  cudaMemcpy(*x_d_p, *x_h_p, size, cudaMemcpyHostToDevice);
  cudaMemcpy(*y_d_p, *y_h_p, size, cudaMemcpyHostToDevice);
} /* Setup */

/*-------------------------------------------------------------------
 * Function:  Dot_wrapper
 * Purpose:   CPU wrapper function for GPU dot product
 * Note:      Assumes x_d, y_d have already been
 *            allocated and initialized on device.  Also
 *            assumes z_d has been allocated.
 */
float Dot_wrapper(float x_d[], float y_d[], float z_d[], int n, int blocks,
                  int threads) {
  int i;
  float dot = 0.0;
  float z_h[blocks];

  /* Invoke kernel */
  Dev_dot << <blocks, threads>>> (x_d, y_d, z_d, n);
  cudaThreadSynchronize();

  cudaMemcpy(&z_h, z_d, blocks * sizeof(float), cudaMemcpyDeviceToHost);

  for (i = 0; i < blocks; i++)
    dot += z_h[i];
  return dot;
} /* Dot_wrapper */

__global__ void mydot(float *a, float *b, float *c, int n) {
  __shared__ float temp[THREADS_PER_BLOCK];

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  temp[threadIdx.x] = a[index] * b[index];

  __syncthreads();

  if (threadIdx.x == 0) {
    float sum = 0;
    for (int i = 0; i < THREADS_PER_BLOCK; i++)
      sum += temp[i];

    atomicAdd(c, sum);
  }
}

/*-------------------------------------------------------------------
 * Function:  Dot_wrapper
 * Purpose:   CPU wrapper function for GPU dot product
 * Note:      Assumes x_d, y_d have already been
 *            allocated and initialized on device.  Also
 *            assumes z_d has been allocated.
 */
float Dot_wrapper2(float x_d[], float y_d[], float z_d[], int n, int blocks,
                   int threads) {
  // use z_d[0] to store the result

  /* Invoke kernel */
  // Dev_dot << <blocks, threads>>> (x_d, y_d, z_d, n);
  mydot << <blocks, threads>>> (x_d, y_d, z_d, n);

  // copy result
  float rv = 0;
  cudaMemcpy(&rv, z_d, sizeof(float), cudaMemcpyDeviceToHost);
  return rv;

} /* Dot_wrapper */

/*-------------------------------------------------------------------
 * Function:  Serial_dot
 * Purpose:   Compute a dot product on the cpu
 */
float Serial_dot(float x[], float y[], int n) {
  int i;
  float dot = 0;

  for (i = 0; i < n; i++)
    dot += x[i] * y[i];

  return dot;
} /* Serial_dot */

/*-------------------------------------------------------------------
 * Function:  Free_mem
 * Purpose:   Free host and device memory
 */
void Free_mem(float *x_h, float *y_h, float *x_d, float *y_d, float *z_d) {

  /* Free device memory */
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);

  /* Free host memory */
  free(x_h);
  free(y_h);

} /* Free_mem */
