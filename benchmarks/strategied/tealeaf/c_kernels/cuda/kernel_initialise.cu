#include "c_kernels.h"
#include "cuknl_shared.h"
#include "../../shared.h"
#include "../../accpol.h"
#include "../../chunk.h"
#include <stdlib.h>
#include <map>

#define RELEVANT_SIZE 1024ull * 1024ull * 1024ull * 12ull / 100ull

extern size_t UM_TOT_MEM;
extern size_t UM_MAX_MEM;
extern std::map<void*, size_t> MEMMAP;
void cudaFreeF(void* ptr);

cudaError_t printMallocManaged ( void** devPtr, size_t size){
  //if(size > RELEVANT_SIZE) fprintf(stdout, "Allocated %.3fGB with cudaMallocManaged\n", (float)size / (1024. * 1024. * 1024.));
  if(true) fprintf(stdout, "Allocated %.3fGB with cudaMallocManaged\n", (float)size / (1024. * 1024. * 1024.));
  return cudaMallocManaged(devPtr, size);
}

void set_access_policy(AccessPolicy * acp, void ** ptr, size_t size, int alloc_index, int argc, char** argv){
    void ** old_ptr = ptr;
    fprintf(stdout, "Alloc %d start: %zu\n", alloc_index, *ptr);
    (*acp).setAllocationPolicy(ptr, size, alloc_index, argc, argv);
    UM_TOT_MEM -= MEMMAP[old_ptr];
    MEMMAP.erase(old_ptr);
    MEMMAP[ptr] = size;
    UM_TOT_MEM += MEMMAP[ptr];
}

int alloc_index = 0;
// shared by all allocations so per-phase policies keep their state
AccessPolicy tealeaf_acp;

// phase 0: setup, phase 1: timestep loop
void apply_phase_policy(int phase)
{
    tealeaf_acp.applyKernelPolicy(phase);
}

// Allocates, and zeroes and individual buffer
void allocate_device_buffer(double** a, int x, int y, int argc, char ** argv)
{
    printf("x, y, x*y: %d, %d, %d\n", x, y, x*y);
    printMallocManaged((void**)a, x*y*sizeof(double));
    UM_TOT_MEM += x*y*sizeof(double);
    MEMMAP[a] = x*y*sizeof(double);
    UM_MAX_MEM = UM_TOT_MEM > UM_MAX_MEM ? UM_TOT_MEM : UM_MAX_MEM;

    check_errors(__LINE__, __FILE__);

    int num_blocks = ceil((double)(x*y)/(double)BLOCK_SIZE);
    zero_buffer<<<num_blocks, BLOCK_SIZE>>>(x, y, *a);
    check_errors(__LINE__, __FILE__);
    size_t size = x*y*sizeof(double);
    if(size > RELEVANT_SIZE){
      set_access_policy(&tealeaf_acp, (void**)a, x*y*sizeof(double), alloc_index++, argc, argv);
    }
}

void allocate_host_buffer(double** a, int x, int y)
{
    *a = (double*)malloc(sizeof(double)*x*y);

    if(*a == NULL) 
    {
        die(__LINE__, __FILE__, "Error allocating buffer %s\n");
    }

#pragma omp parallel for
    for(int jj = 0; jj < y; ++jj)
    {
        for(int kk = 0; kk < x; ++kk)
        {
            const int index = kk + jj*x;
            (*a)[index] = 0.0;
        }
    }
}

// Allocates all of the field buffers
void kernel_initialise(
        Settings* settings, int x, int y, double** density0, 
        double** density, double** energy0, double** energy, double** u, 
        double** u0, double** p, double** r, double** mi, 
        double** w, double** kx, double** ky, double** sd, 
        double** volume, double** x_area, double** y_area, double** cell_x, 
        double** cell_y, double** cell_dx, double** cell_dy, double** vertex_dx, 
        double** vertex_dy, double** vertex_x, double** vertex_y,
        double** cg_alphas, double** cg_betas, double** cheby_alphas,
        double** cheby_betas, double** d_comm_buffer, double** d_reduce_buffer, 
        double** d_reduce_buffer2, double** d_reduce_buffer3, double** d_reduce_buffer4)
{
    int argc = settings->argc;
    char ** argv = settings->argv;
    print_and_log(settings,
            "Performing this solve with the CUDA %s solver\n",
            settings->solver_name);    

    // TODO: DOES NOT BELONG HERE!!!
    //
    // Naive assumption that devices are paired even and odd
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    int device_id = settings->rank%num_devices;

    int result = cudaSetDevice(device_id);
    if(result != cudaSuccess)
    {
        die(__LINE__,__FILE__,"Could not allocate CUDA device %d.\n", device_id);
    }

    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device_id);

    print_and_log(settings, "Rank %d using %s device id %d\n", 
            settings->rank, properties.name, device_id);

    const int x_inner = x - 2*settings->halo_depth;
    const int y_inner = y - 2*settings->halo_depth;

    allocate_device_buffer(density0, x, y, argc, argv);
    allocate_device_buffer(density, x, y, argc, argv);
    allocate_device_buffer(energy0, x, y, argc, argv);
    allocate_device_buffer(energy, x, y, argc, argv);
    allocate_device_buffer(u, x, y, argc, argv);
    allocate_device_buffer(u0, x, y, argc, argv);
    allocate_device_buffer(p, x, y, argc, argv);
    allocate_device_buffer(r, x, y, argc, argv);
    allocate_device_buffer(mi, x, y, argc, argv);
    allocate_device_buffer(w, x, y, argc, argv);
    allocate_device_buffer(kx, x, y, argc, argv);
    allocate_device_buffer(ky, x, y, argc, argv);
    allocate_device_buffer(sd, x, y, argc, argv);
    allocate_device_buffer(volume, x, y, argc, argv);
    allocate_device_buffer(x_area, x+1, y, argc, argv);
    allocate_device_buffer(y_area, x, y+1, argc, argv);
    allocate_device_buffer(cell_x, x, 1, argc, argv);
    allocate_device_buffer(cell_y, 1, y, argc, argv);
    allocate_device_buffer(cell_dx, x, 1, argc, argv);
    allocate_device_buffer(cell_dy, 1, y, argc, argv);
    allocate_device_buffer(vertex_dx, x+1, 1, argc, argv);
    allocate_device_buffer(vertex_dy, 1, y+1, argc, argv);
    allocate_device_buffer(vertex_x, x+1, 1, argc, argv);
    allocate_device_buffer(vertex_y, 1, y+1, argc, argv);
    allocate_device_buffer(d_comm_buffer, settings->halo_depth, max(x_inner, y_inner), argc, argv);
    allocate_device_buffer(d_reduce_buffer, x, y, argc, argv);
    allocate_device_buffer(d_reduce_buffer2, x, y, argc, argv);
    allocate_device_buffer(d_reduce_buffer3, x, y, argc, argv);
    allocate_device_buffer(d_reduce_buffer4, x, y, argc, argv);

    allocate_device_buffer(cg_alphas, settings->max_iters, 1, argc, argv);
    allocate_device_buffer(cg_betas, settings->max_iters, 1, argc, argv);
    allocate_device_buffer(cheby_alphas, settings->max_iters, 1, argc, argv);
    allocate_device_buffer(cheby_betas, settings->max_iters, 1, argc, argv);
    // place phase 0 before profiling starts
    apply_phase_policy(0);
}

// Finalises the kernel
void kernel_finalise(
        double* cg_alphas, double* cg_betas, double* cheby_alphas,
        double* cheby_betas)
{
    cudaFreeF(cg_alphas);
    cudaFreeF(cg_betas);
    cudaFreeF(cheby_alphas);
    cudaFreeF(cheby_betas);
}
