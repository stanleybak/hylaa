
// Arnoldi algorithm to get V and H matrix 
// Dung Tran 
// Jun 21/2017

#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>
#include <cusp/gallery/grid.h>
#include <cusp/coo_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/detail/random.inl>

int main(void)
{

	// system matrix A		
	//cusp::hyb_matrix<int, float, cusp::host_memory> A;
        //cusp::gallery::grid2d(A, 4, 4);
	//cusp::random_array<float> rand(16);
  	//cusp::blas::copy(rand, A);
	cusp::array2d<int,cusp::host_memory> A(3,3);
	A(0,0) = 0; A(0,1) = 1; A(0,2) = 2;
  	A(1,0) = 3; A(1,1) = 4; A(1,2) = 5;
  	A(2,0) = 6; A(2,1) = 7; A(2,2) = 8;
	cusp::print(A);    
	
	// initial vector    
	cusp::array1d<float,cusp::host_memory> x0(3);
	x0[0] = 0.745578; 
	x0[1] = 0.570053; 
	x0[2] = 0.345184;
	cusp::print(x0);	

	// krylov supspace dimension, number of iteration
	int m = 2;  
	int N = A.num_rows;  // system dimension 
	int maxiter = std::min(N, m);   
		
	// create matrix H_ for iteration	
	cusp::array2d<float,cusp::host_memory> H_(maxiter + 1, maxiter, 0);
	
	// return matrix H after iteration -- Hm in the algorithm -- (m x m) matrix 
	cusp::array2d<float,cusp::host_memory> H(maxiter, maxiter);

        // create matrix V_ for iteration
        std::vector< cusp::array1d<float,cusp::host_memory> > V_(maxiter + 1);
        for (int i = 0; i < maxiter + 1; i++)
           V_[i].resize(N);

	// returned matrix V after iteration -- Vm in the algorithm -- (N x m) matrix
	cusp::array2d<float,cusp::host_memory> V(N,maxiter);

	// copy x0 into V_[0] 
        //cusp::copy(cusp::detail::random_reals<float>(N), V_[0]);
	cusp::copy(x0,V_[0]); 


	// get beta 
	float beta = float(1) / cusp::blas::nrm2(x0);
	printf("%f\n", beta);

	// normalize x0
	cusp::blas::scal(V_[0], beta);
	
	// iteration
	
	int j;
	for(j = 0; j < maxiter; j++)
	{
		cusp::multiply(A, V_[j], V_[j + 1]);

		cusp::print(V_[j]); 

		for(int i = 0; i <= j; i++)
		{
			H_(i,j) = cusp::blas::dot(V_[i], V_[j + 1]);

			cusp::blas::axpy(V_[i], V_[j + 1], -H_(i,j));
		}

		H_(j+1,j) = cusp::blas::nrm2(V_[j + 1]);

		if(H_(j+1,j) < 1e-10) break;

		cusp::blas::scal(V_[j + 1], float(1) / H_(j+1,j));
	}
	
	// get matrix H (m x m dimension) and print it
	for(int rowH=0;rowH < maxiter; rowH++)
		for(int colH = 0; colH <maxiter; colH++)
			H(rowH,colH) = H_(rowH,colH);

	cusp::print(H_);
		
	cusp::print(H);

	// get matrix V (N x m dimension) and print it

	cusp::array1d<float,cusp::host_memory> x1(N);	

	for(int colV = 0; colV < maxiter; colV++)
	{	cusp::copy(V_[colV],x1);
		cusp::print(x1);		
		for(int rowV=0;rowV < N; rowV++)
			V(rowV, colV) = x1[rowV];
	}

	cusp::print(V);
	
        // compute Vb = beta*V -- (N x m) matrix
	cusp::array2d<float,cusp::host_memory> Vb(N,maxiter);

	for(int colV2 = 0; colV2 <j; colV2++)
		cusp::blas::scal(V_[colV2], float(1) / cusp::blas::nrm2(V_[0]));
	
	
	for(int rowV3=0;rowV3 < N; rowV3++)
		for(int colV3 = 0; colV3 <j; colV3++)
			Vb(rowV3,colV3) = V_[colV3][rowV3];

	cusp::print(Vb);

	// todo: compute Hms = i*step*Hm; compute exp(Hms);  
	

    return 0;
}

