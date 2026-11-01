#ifndef ACCESS_POLICY_H
#define ACCESS_POLICY_H

// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdint.h>
#include <string.h>
#include <chrono>
#include <string>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>
#include <unistd.h>
#include <fcntl.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(status) \
  if (status != cudaSuccess) \
  { \
    printf("%s:%d CudaError: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    assert(0); \
  }
#endif
#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR()                                                    \
{                                                                             \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                   \
    {                                                                         \
        printf("error=%d name=%s at "                                         \
               "ln: %d\n  ",err,cudaGetErrorString(err),__LINE__);            \
        exit(1);\
    }                                                                         \
}
#endif


class AccessPolicy{
private:
	void * mem_pressure = 0x0;
	int aoi = -1; 
	std::size_t aoi_size = 0;
	double perc = -1.0;

	// Per-kernel policies: "-p <kernel 0>:<kernel 1>:..." gives one d/m/h string per kernel, applied by
	// applyKernelPolicy() before each launch. A policy without ':' is the usual whole-run policy.
	struct KernelPolicyAlloc { void **ptr; void *managed; void *device; std::size_t size; char state; };
	std::vector<std::string> kernel_policies;
	std::vector<KernelPolicyAlloc> kernel_allocs;
	double transition_seconds = 0.0;

	bool parseKernelPolicies(int argc, char** argv){
	  if (!kernel_policies.empty()) return true;
	  for (int i = 1; i + 1 < argc; i++){
	    if (strcmp(argv[i], "-p") == 0 && strchr(argv[i+1], ':')){
	      std::string all(argv[i+1]), policy;
	      for (char c : all + ":"){
	        if (c != ':') { policy += c; continue; }
	        if (!policy.empty()) kernel_policies.push_back(policy);
	        policy.clear();
	      }
	      return true;
	    }
	  }
	  return false;
	}

	void setAdvice(void *a, std::size_t size, char policy){
	  if (policy == 'h'){
	    CUDA_CHECK(cudaMemAdvise(a, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
	    CUDA_CHECK(cudaMemAdvise(a, size, cudaMemAdviseSetAccessedBy, 0));
	  } else {
	    CUDA_CHECK(cudaMemAdvise(a, size, cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId));
	    CUDA_CHECK(cudaMemAdvise(a, size, cudaMemAdviseUnsetAccessedBy, 0));
	  }
	}
public:
	~AccessPolicy(){
	  if (!kernel_policies.empty())
	    fprintf(stdout, "Kernel policy transitions: %0.6lfs\n", transition_seconds);
	}

	// Move each allocation to the given kernel's policy. 'd' copies it into a cudaMalloc buffer, frees the
	// managed allocation and swaps the pointer, exactly like the whole-run 'd'; leaving 'd' copies it into a
	// new managed allocation. Call before every launch.
	void applyKernelPolicy(int kernel){
	  if (kernel_policies.empty()) return;
	  if ((std::size_t)kernel >= kernel_policies.size() || kernel_policies[kernel].size() < kernel_allocs.size()){
	    std::cout << "Kernel policy for kernel " << kernel << " is missing or too short.\n";
	    exit(1);
	  }
	  const std::string &policy = kernel_policies[kernel];
	  auto start = std::chrono::steady_clock::now();
	  // leave the device first so its buffers free room for allocations entering it
	  for (std::size_t i = 0; i < kernel_allocs.size(); i++){
	    KernelPolicyAlloc &alloc = kernel_allocs[i];
	    if (alloc.state == 'd' && policy[i] != 'd'){
	      CUDA_CHECK(cudaMallocManaged(&alloc.managed, alloc.size));
	      CUDA_CHECK(cudaMemcpy(alloc.managed, alloc.device, alloc.size, cudaMemcpyDefault));
	      CUDA_CHECK(cudaFree(alloc.device));
	      *alloc.ptr = alloc.managed;
	      alloc.state = 'm';
	    }
	  }
	  for (std::size_t i = 0; i < kernel_allocs.size(); i++){
	    KernelPolicyAlloc &alloc = kernel_allocs[i];
	    char target = policy[i];
	    if (target == alloc.state) continue;
	    if (target == 'd'){
	      if (cudaMalloc(&alloc.device, alloc.size) != cudaSuccess){
	        std::cout << "Kernel policy '" << policy << "' does not fit on the device (allocation " << i << ").\n";
	        exit(2);
	      }
	      // HostToDevice like the whole-run 'd': Default would treat the source as managed memory and could
	      // move its pages onto the GPU first
	      CUDA_CHECK(cudaMemcpy(alloc.device, alloc.managed, alloc.size, cudaMemcpyHostToDevice));
	      // a managed copy left behind can hold GPU memory
	      CUDA_CHECK(cudaFree(alloc.managed));
	      *alloc.ptr = alloc.device;
	    } else if (target == 'h' || target == 'm'){
	      setAdvice(alloc.managed, alloc.size, target);
	    } else {
	      std::cout << "Policy flag '" << target << "' used on allocation " << i << " is not supported per kernel.\n";
	      exit(1);
	    }
	    alloc.state = target;
	  }
	  CUDA_CHECK(cudaDeviceSynchronize());
	  transition_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	}

	void setMemPressure(std::size_t size){
		if(size <= 1000) 
			return;
		cudaMalloc(&mem_pressure, size);
		CHECK_CUDA_ERROR();
		if(mem_pressure == 0x0){
			std::cout << "mem pressure failed\n";
			exit(-1);
		}
	      	double pressureGB = ((double)aoi_size * perc) / (1024. * 1024. * 1024.);
		fprintf(stdout, "Allocated %.2f GB (%.2f%% of allocation %d) of memory pressure\n",
		      pressureGB, perc, aoi);
	}

	void freeMemPressure(){
		cudaDeviceSynchronize();
		if(mem_pressure == 0x0) return;
		cudaFree(mem_pressure);
	}

	void usage(char * argv0){
		std::cout << "Usage: " << argv0 << "<benchmark args> <flag1> <arg1> <flag2...>\n"
			<< "Available flags: \n"
			<< "-p: policy\t| string \t| one character per allocation\n"
			<< "-aoi: alloc of interest\t| int\t| which alloc to apply mem pressure\n"
			<< "-r: pressure\t| float\t | fraction of aoi's size to block in memory as mem pressure\n"; 
	}
	//Checks for two things: 1) The mem policy of a given allocation 2) Checks and
	//implements memory pressure if applicable
	char parseCLA(std::size_t size, int alloc_num, int argc, char** argv){
	  char flag = '-';
	  bool args = false;
	  for (int i = 1; i < argc; i++){
	    if (argv[i][0] == '-' &&  argv[i][1] == 'p'){ // next arg sets policy
	      args = true;
	      flag = argv[++i][alloc_num];
	    }
	    if (strcmp(argv[i], "-aoi") == 0 && atoi(argv[i+1]) == alloc_num){
	      args = true;
	      aoi_size = size;
	      aoi = atoi(argv[i+1]);
	    }
	    if (argv[i][0] == '-' &&  argv[i][1] == 'r'){ // next arg sets pressure
	      args = true;
	      perc = atof(argv[++i]);
	      if(aoi == alloc_num)
		  setMemPressure((std::size_t)((double)aoi_size * perc));
	    }
	  }
	  if (!args) usage(argv[0]);
	  return flag;
	}


	//Set the allocation policy based on cl flags
	void setAllocationPolicy(void **a, std::size_t size, int alloc_num, int argc, char** argv) {
	  if (parseKernelPolicies(argc, argv)){
	    // per-kernel mode: only record the allocation; applyKernelPolicy() places it
	    if (kernel_allocs.size() <= (std::size_t)alloc_num) kernel_allocs.resize(alloc_num + 1);
	    kernel_allocs[alloc_num] = {a, *a, nullptr, size, 'm'};
	    return;
	  }
	  char flag = parseCLA(size, alloc_num, argc, argv);
    //std::cout << "Setting allocation " << alloc_num << " of size " << float(size) / (1024. * 1024. * 1024.) << "GB to '" << flag << "'\n";
	  switch(flag){
	    case 'm': //Policy is migrate; do nothing
	      break;
	    case 'd': //Pin to device; use cudaMemCopy
	      void * devptr;
	      cudaMalloc(&devptr, size);
	      CHECK_CUDA_ERROR();
	      cudaMemcpy(devptr, *a, size, cudaMemcpyHostToDevice);
	      CHECK_CUDA_ERROR();
	      CUDA_CHECK(cudaFree(*a));
	      CHECK_CUDA_ERROR();
	      *a = devptr;
	      break;
	    case 'a': //Pin to device, async variant
	      //First, set preferred location of everything to device
	      cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, 0);
	      //Finally, move the pin device mem to device
	      cudaMemPrefetchAsync(*a, size, 0, 0);
	      break;
	    case 'h': //Pin to host; use cudaMemAdvise
        CUDA_CHECK(cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CUDA_CHECK(cudaMemAdvise(*a, size, cudaMemAdviseSetAccessedBy, 0));
        CHECK_CUDA_ERROR();
        break;
	    case 's': //Split allocation between pin host and pin device
	      //Note that perc will be set already 
	      //First, set preferred location of everything to device
	      cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, 0);
	      printf("Perc is %.2f\n", perc);
	      if(perc <= 0.0) break;
	      //Next, set preferred location of first (perc) bytes to host
	      cudaMemAdvise(*a, (perc*size), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	      //Finally, move the pin device mem to device
	      //Assumes type is <=8 bytes
	      cudaMemPrefetchAsync(static_cast<uint64_t*>(*a)+((size_t)(size*perc)/sizeof(uint64_t)), (1.0-perc) * size, 0, 0);
	      break;
	    case 'x': //Split allocation between pin host and pin device with no prefetch 
	      //Note that perc will be set already 
	      //First, set preferred location of everything to device
	      cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, 0);
	      //Next, set preferred location of first (perc) bytes to host
	      cudaMemAdvise(*a, (perc*size), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	      break;
      default:
        std::cout << "Policy flag '" << flag << "' used on allocation " << alloc_num << " is not supported.\n";
        exit(1);
	  }
	  return;
	}

	
};

#endif
