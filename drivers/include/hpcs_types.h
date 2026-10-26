#pragma once

typedef enum
{
    HPCS_FAULT = 0,
    HPCS_PREFETCH,
    HPCS_EVICTION,
    HPCS_ADDR_RANGE,
    HPCS_KERNEL_BEGIN,
    HPCS_KERNEL_END
} hpcs_record_types_t;

#define HPCS_DEV_NAME "hpcs_logger"
#define HPCS_CLASS_NAME "hpcs_logger"

#define HPCS_BUFFER_CAP 10000000lu

// in short mode, we will only send fault addresses. for va range, we will have to send two back-to-back messages.
struct hpcs_fault_record_short
{
    unsigned long faddr;
};

struct hpcs_fault_record
{
    unsigned long faddr;
    unsigned long timestamp;
    // this is 2 bytes in source code
    // number of instances of fault with same fault type/mask
    unsigned short num_instances;
    // 16 types means 8 bits should be more than fine
    unsigned char fault_type;
    // these two are 32 bit types for expansion, but
    // in practice have less than 8 states. We can make
    // these fields larger in future versions if needed.
    unsigned char access_type;
    unsigned char access_type_mask;
    // client type can be g, h, or c, 0/1/2
    unsigned char client_type;
    // graphics, host, CE, count
    unsigned char mmu_engine_type;
    // sm id if used with utlb id
    unsigned char client_id;
    // which mmu, presumably there's not 256 mmus.
    unsigned char mmu_engine_id;
    // presumably there's not more than 256 utlbs.
    unsigned char utlb_id;
    // presuambly there's not more than 256 gpcs
    unsigned char gpc_id;
    unsigned char channel_id;
    unsigned char ve_id;
    // valid types: fault/prefetch/eviction/addr range/kernel begin/kernel end; 0/1/2/3/4/5. Assuming all
    // types are present in the data, this is the ordering from most to least likely.
    unsigned char record_type;
};

// written by user space to /dev/hpcs_logger to bracket a kernel launch. The driver logs it as a record
// with faddr = kernel_id; in short mode kernel_id is encoded with HPCS_ENCODE_ID, so it must fit in 56 bits.
struct hpcs_kernel_marker
{
    unsigned long kernel_id;
    // HPCS_KERNEL_BEGIN or HPCS_KERNEL_END
    unsigned char type;
};
