CUDA_HOME ?= /usr/local/cuda
CUPTI_HOME ?= $(CUDA_HOME)

# -lcuda is required: static cudart dlopens libcuda with local scope, so cuCtxSynchronize is not
# visible to this library otherwise. libcuda silently skips an injection library that fails to load,
# so if no <bench>_kernels.csv appears, check `ldd libkernel_marker.so` first.
libkernel_marker.so: kernel_marker.cpp ../../drivers/include/hpcs_types.h
	g++ -std=c++11 -O2 -Wall -fPIC -shared -I$(CUDA_HOME)/include -I$(CUPTI_HOME)/include $< -o $@ \
		-L$(CUPTI_HOME)/lib64 -lcupti -Wl,-rpath,$(CUPTI_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcuda

.PHONY: clean
clean:
	rm -f libkernel_marker.so
