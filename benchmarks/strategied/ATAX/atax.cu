/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>
#include <math.h>
#include "../common/UVMBench/polybenchUtilFuncts.h"
#include "accpol.h"
//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
#define NX PSIZE
#define NY PSIZE

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;
size_t PSIZE;


void init_array(DATA_TYPE *x, DATA_TYPE *A, DATA_TYPE *x_gpu, DATA_TYPE *A_gpu)
{
	int i, j;

	for (i = 0; i < NX; i++)
	{
		x[i] = i * M_PI;
		x_gpu[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
			A_gpu[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
		}
	}
}


void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu)
{
	int i, fail;
	fail = 0;

	for (i=0; i<NY; i++)
	{
		if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp, size_t PSIZE)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX)
	{
		int j;
		for(j=0; j < NY; j++)
		{
			tmp[i] += A[i * NY + j] * x[j];
		}
	}
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp, size_t PSIZE)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		int i;
		for(i=0; i < NX; i++)
		{
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}


void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
	int i,j;
	
	for (i= 0; i < NY; i++)
	{
    	y[i] = 0;
	}
  
	for (i = 0; i < NX; i++)
 	{
      	tmp[i] = 0;

      	for (j = 0; j < NY; j++)
		{
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}
		
      	for (j = 0; j < NY; j++)
		{
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
    }
}


// by reference since per-kernel 'd' swaps the pointers
void ataxGpu(DATA_TYPE*& A_gpu, DATA_TYPE*& x_gpu, DATA_TYPE*& y_gpu, DATA_TYPE*& tmp_gpu, AccessPolicy& acp)
{
	double t_start, t_end;

	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

	t_start = rtclock();
	acp.applyKernelPolicy(0);
	atax_kernel1<<< grid1, block >>>(A_gpu,x_gpu,tmp_gpu, PSIZE);
	cudaDeviceSynchronize();
	acp.applyKernelPolicy(1);
	atax_kernel2<<< grid2, block >>>(A_gpu,y_gpu,tmp_gpu, PSIZE);
	cudaDeviceSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

}


int main(int argc, char** argv)
{

	//Set problem size with argv[1]
        if(argc>=2){
                if(strcmp(argv[1],"-h")==0){
                        printf("Usage: %s <psize (GB)> [1: cpu, 0: no cpu]\n",argv[0]);
                        exit(0);
                }
                double bytes = 1024. * 1024. * 1024. * atof(argv[1]);
                //printf("Bytes: %.2f\n", bytes);
                PSIZE = (size_t) (sqrt(bytes/4));
        }
        else{
                PSIZE = 4096;
        }
        //printf("PSIZE: %zu\n", PSIZE);
        printf("Problem size: %.2f GB\n", ((((double)(PSIZE * PSIZE) + (3 * PSIZE)) * 4)/(1024. * 1024. * 1024.)));
        int cpu = 0;
        if(argc >= 3)
                cpu = atoi(argv[2]);	

	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* x;
	DATA_TYPE* y;
	DATA_TYPE* tmp;

	DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	// DATA_TYPE* tmp;
	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	y = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NX * NY);
	cudaMallocManaged(&x_gpu, sizeof(DATA_TYPE) * NY);
	cudaMallocManaged(&y_gpu, sizeof(DATA_TYPE) * NY);
	cudaMallocManaged(&tmp_gpu, sizeof(DATA_TYPE) * NX);


	init_array(x, A, x_gpu, A_gpu);

	GPU_argv_init();
	AccessPolicy acp;
	
	acp.setAllocationPolicy((void**)&A_gpu, sizeof(DATA_TYPE) * NX * NY, 0, argc, argv);
	acp.setAllocationPolicy((void**)&x_gpu, sizeof(DATA_TYPE) * NY, 1, argc, argv);
	acp.setAllocationPolicy((void**)&y_gpu, sizeof(DATA_TYPE) * NY, 2, argc, argv);
	acp.setAllocationPolicy((void**)&tmp_gpu, sizeof(DATA_TYPE) * NX, 3, argc, argv);
	// place the first kernel before timing
	acp.applyKernelPolicy(0);


	
	ataxGpu(A_gpu, x_gpu, y_gpu, tmp_gpu, acp);
	
	if(cpu){
		t_start = rtclock();
		atax_cpu(A, x, y, tmp);
		t_end = rtclock();
		fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

		compareResults(y, y_gpu);
	}

	free(A);
	free(x);
	free(y);
	free(tmp);
	
	cudaFree(A_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);

  	return 0;
}

