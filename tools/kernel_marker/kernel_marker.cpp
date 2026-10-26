#include <cuda.h>
#include <cupti.h>
#include <fcntl.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../../drivers/include/hpcs_types.h"

#define CUPTI_CHECK(call)                                                           \
    do                                                                              \
    {                                                                               \
        CUptiResult status_ = call;                                                 \
        if (status_ != CUPTI_SUCCESS)                                               \
        {                                                                           \
            const char* msg_;                                                       \
            cuptiGetResultString(status_, &msg_);                                   \
            fprintf(stderr, "kernel_marker: %s failed: %s\n", #call, msg_);         \
            return 0;                                                               \
        }                                                                           \
    } while (0)

static const CUpti_CallbackId launch_cbids[] = {
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel,
    CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz,
};

static CUpti_SubscriberHandle subscriber;
static int marker_fd = -1;
static FILE* kernel_map = NULL;
static std::mutex launch_mutex;
static unsigned long next_kernel_id = 0;
static unsigned long current_kernel_id;

static void write_marker(unsigned long kernel_id, unsigned char type)
{
    struct hpcs_kernel_marker marker = {kernel_id, type};
    if (marker_fd >= 0 && write(marker_fd, &marker, sizeof(marker)) != sizeof(marker))
        perror("kernel_marker: write marker");
}

static void CUPTIAPI launch_callback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                                     const void* cbdata)
{
    const CUpti_CallbackData* info = (const CUpti_CallbackData*)cbdata;
    if (info->callbackSite == CUPTI_API_ENTER)
    {
        launch_mutex.lock();
        // finish outstanding async work so its faults are logged before this kernel begins
        cuCtxSynchronize();
        current_kernel_id = next_kernel_id++;
        if (kernel_map)
        {
            // flush every line so the map survives the benchmark being killed by timeout
            fprintf(kernel_map, "%lu,%s\n", current_kernel_id, info->symbolName ? info->symbolName : "unknown");
            fflush(kernel_map);
        }
        write_marker(current_kernel_id, HPCS_KERNEL_BEGIN);
    }
    else
    {
        // wait for the kernel so all of its faults are logged before the end marker
        cuCtxSynchronize();
        write_marker(current_kernel_id, HPCS_KERNEL_END);
        launch_mutex.unlock();
    }
}

extern "C" int InitializeInjection(void)
{
    marker_fd = open("/dev/hpcs_logger", O_WRONLY);
    if (marker_fd < 0)
        perror("kernel_marker: open /dev/hpcs_logger");
    const char* path = getenv("HPCS_KERNEL_MAP");
    kernel_map = fopen(path ? path : "hpcs_kernels.csv", "w");
    if (kernel_map)
        fprintf(kernel_map, "kernel_id,name\n");
    else
        perror("kernel_marker: open kernel map");

    CUPTI_CHECK(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)launch_callback, NULL));
    for (CUpti_CallbackId cbid : launch_cbids)
        CUPTI_CHECK(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, cbid));
    return 1;
}
