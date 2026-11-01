/**
 * gramschmidt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <math.h>
#include "accpol.h"
#include "../common/UVMBench/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define M PSIZE
#define N PSIZE

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;
size_t PSIZE;


void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q)
{
	size_t i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < N; k++)
	{
		nrm = 0;
		for (i = 0; i < M; i++)
		{
			nrm += A[i*N + k] * A[i*N + k];
		}
		
		R[k*N + k] = sqrt(nrm);
		for (i = 0; i < M; i++)
		{
			Q[i*N + k] = A[i*N + k] / R[k*N + k];
		}
		
		for (j = k + 1; j < N; j++)
		{
			R[k*N + j] = 0;
			for (i = 0; i < M; i++)
			{
				R[k*N + j] += Q[i*N + k] * A[i*N + j];
			}
			for (i = 0; i < M; i++)
			{
				A[i*N + j] = A[i*N + j] - Q[i*N + k] * R[k*N + j];
			}
		}
	}
}


void init_array(DATA_TYPE* A)
{
	size_t i, j;

    #pragma omp parallel for simd collapse(2) private(i, j) shared(A)
	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
		}
	}
}


void compareResults(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu)
{
	size_t i, j, fail;
	fail = 0;

	for (i=0; i < M; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(A[i*N + j], A_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{				
				fail++;
				printf("i: %lu j: %lu \n1: %f\n 2: %f\n", i, j, A[i*N + j], A_outputFromGpu[i*N + j]);
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %lu\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );	
	return;
}


__global__ void gramschmidt_kernel1(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k,size_t PSIZE)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid==0)
	{
		DATA_TYPE nrm = 0.0;
		int i;
		for (i = 0; i < M; i++)
		{
			nrm += a[i * N + k] * a[i * N + k];
		}
      		r[k * N + k] = sqrt(nrm);
	}
}


__global__ void gramschmidt_kernel2(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k, size_t PSIZE)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < M)
	{	
		q[i * N + k] = a[i * N + k] / r[k * N + k];
	}
}


__global__ void gramschmidt_kernel3(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k, size_t PSIZE)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((j > k) && (j < N))
	{
		r[k*N + j] = 0.0;

		int i;
		for (i = 0; i < M; i++)
		{
			r[k*N + j] += q[i*N + k] * a[i*N + j];
		}
		
		for (i = 0; i < M; i++)
		{
			a[i*N + j] -= q[i*N + k] * r[k*N + j];
		}
	}
}


// pointers by reference: a per-kernel 'd' swaps them between launches
void gramschmidtCuda(DATA_TYPE*& A_gpu, DATA_TYPE*& R_gpu, DATA_TYPE*& Q_gpu, double timeout, AccessPolicy& acp)
{
	double t_start, t_end;
	size_t k;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 gridKernel1(1, 1);
	dim3 gridKernel2((size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)), 1);
	dim3 gridKernel3((size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)), 1);
	
	t_start = rtclock();
	for (k = 0; k < N; k++)
	{
		acp.applyKernelPolicy(0);
		gramschmidt_kernel1<<<gridKernel1,block>>>(A_gpu, R_gpu, Q_gpu, k, PSIZE);
		CUDA_CHECK(cudaDeviceSynchronize());
		acp.applyKernelPolicy(1);
		gramschmidt_kernel2<<<gridKernel2,block>>>(A_gpu, R_gpu, Q_gpu, k, PSIZE);
		CUDA_CHECK(cudaDeviceSynchronize());
		acp.applyKernelPolicy(2);
		gramschmidt_kernel3<<<gridKernel3,block>>>(A_gpu, R_gpu, Q_gpu, k, PSIZE);
		CUDA_CHECK(cudaDeviceSynchronize());
		
		if(timeout > 0 && rtclock() - t_start > timeout)
			break;
	}
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
  
}


int main(int argc, char *argv[])
{
	//Set problem size with argv[1]
        if(argc>=2){
                if(strcmp(argv[1],"-h")==0){
                        printf("Usage: %s <psize (GB)> [1: cpu, 0: no cpu]\n",argv[0]);
                        exit(0);
                }
                double bytes = 1024. * 1024. * 1024. * atof(argv[1]);
                //printf("Bytes: %.2f\n", bytes);
                PSIZE = (size_t) (sqrt(bytes/12));
        }
        else{
                PSIZE = 2048;
        }
	
	

        //printf("PSIZE: %zu\n", PSIZE);
        printf("Problem size: %.2f GB\n", ((((double)(PSIZE * PSIZE * 3)) * 4)/(1024. * 1024. * 1024.)));
        int cpu = 0;
        if(argc >= 3)
                cpu = atoi(argv[2]);	

	//Set timeout 
	double timeout = -1;
	if(argc>=4){
		timeout = atof(argv[3]);
		printf("Running gramschm with timeout of %.0f seconds\n", timeout);
	}

	double t_start, t_end;

	DATA_TYPE* A = NULL;
	DATA_TYPE* R = NULL;
	DATA_TYPE* Q = NULL;
	DATA_TYPE *A_gpu = NULL;
	DATA_TYPE *R_gpu = NULL;
	DATA_TYPE *Q_gpu = NULL;
	

	CUDA_CHECK(cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * M * N));
	CUDA_CHECK(cudaMallocManaged(&R_gpu, sizeof(DATA_TYPE) * M * N));
	CUDA_CHECK(cudaMallocManaged(&Q_gpu, sizeof(DATA_TYPE) * M * N));
	
	init_array(A_gpu);
	
	GPU_argv_init();

	AccessPolicy acp;	
	acp.setAllocationPolicy((void**)&A_gpu, sizeof(DATA_TYPE) * M * N, 0, argc, argv);
	acp.setAllocationPolicy((void**)&R_gpu, sizeof(DATA_TYPE) * M * N, 1, argc, argv);
	acp.setAllocationPolicy((void**)&Q_gpu, sizeof(DATA_TYPE) * M * N, 2, argc, argv);
	// first kernel's placement happens before timing, like a whole-run policy
	acp.applyKernelPolicy(0);

	
	gramschmidtCuda(A_gpu, R_gpu, Q_gpu, timeout, acp);
	
	if(cpu){
        A = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
        R = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  
        Q = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  
	    init_array(A);
		t_start = rtclock();
		gramschmidt(A, R, Q);
		t_end = rtclock();

		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
		
		compareResults(A, A_gpu);	
	}
	free(A);
	free(R);
	free(Q);  
	cudaFree(A_gpu);
	cudaFree(R_gpu);
	cudaFree(Q_gpu);

    	acp.freeMemPressure();
	
	return 0;
}

