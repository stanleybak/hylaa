/*-------------------------------------------------------------------
 * Function:  Get_args
 * Purpose:   Get and check command line args.  If there's an error
 *            quit.
 */
void Get_args(int argc, char* argv[], int* n_p, int* threads_per_block_p,
      int* blocks_per_grid_p) {

   if (argc != 4) {
      fprintf(stderr, "usage: %s <vector order> <blocks> <threads>\n", 
            argv[0]);
      exit(0);
   }
   *n_p = strtol(argv[1], NULL, 10);
   *blocks_per_grid_p = strtol(argv[2], NULL, 10);
   *threads_per_block_p = strtol(argv[3], NULL, 10);
}  /* Get_args */

/*-------------------------------------------------------------------
 * Function:  Setup
 * Purpose:   Allocate and initialize host and device memory
 */
void Setup(int n, int blocks, float** x_h_p, float** y_h_p, float** x_d_p, 
      float** y_d_p, float** z_d_p) {
   int i;
   size_t size = n*sizeof(float);

   /* Allocate input vectors in host memory */
   *x_h_p = (float*) malloc(size);
   *y_h_p = (float*) malloc(size);
   
   /* Initialize input vectors */
   srandom(1);
   for (i = 0; i < n; i++) {
      (*x_h_p)[i] = random()/((double) RAND_MAX);
      (*y_h_p)[i] = random()/((double) RAND_MAX);
   }

   /* Allocate vectors in device memory */
   cudaMalloc(x_d_p, size);
   cudaMalloc(y_d_p, size);
   cudaMalloc(z_d_p, blocks*sizeof(float));

   /* Copy vectors from host memory to device memory */
   cudaMemcpy(*x_d_p, *x_h_p, size, cudaMemcpyHostToDevice);
   cudaMemcpy(*y_d_p, *y_h_p, size, cudaMemcpyHostToDevice);
}  /* Setup */

/*-------------------------------------------------------------------
 * Function:  Dot_wrapper
 * Purpose:   CPU wrapper function for GPU dot product
 * Note:      Assumes x_d, y_d have already been 
 *            allocated and initialized on device.  Also
 *            assumes z_d has been allocated.
 */
float Dot_wrapper(float x_d[], float y_d[], float z_d[], 
      int n, int blocks, int threads) {
   int i;
   float dot = 0.0;
   float z_h[blocks];

   /* Invoke kernel */
   Dev_dot<<<blocks, threads>>>(x_d, y_d, z_d, n);
   cudaThreadSynchronize();

   cudaMemcpy(&z_h, z_d, blocks*sizeof(float), cudaMemcpyDeviceToHost);

   for (i = 0; i < blocks; i++)
      dot += z_h[i];
   return dot;
}  /* Dot_wrapper */


/*-------------------------------------------------------------------
 * Function:  Serial_dot
 * Purpose:   Compute a dot product on the cpu
 */
float Serial_dot(float x[], float y[], int n) {
   int i;
   float dot = 0;

   for (i = 0; i < n; i++)
      dot += x[i]*y[i];

   return dot;
}  /* Serial_dot */

/*-------------------------------------------------------------------
 * Function:  Free_mem
 * Purpose:   Free host and device memory
 */
void Free_mem(float* x_h, float* y_h, float* x_d, float* y_d,
      float* z_d) {

   /* Free device memory */
   cudaFree(x_d);
   cudaFree(y_d);
   cudaFree(z_d);

   /* Free host memory */
   free(x_h);
   free(y_h);

}  /* Free_mem */
