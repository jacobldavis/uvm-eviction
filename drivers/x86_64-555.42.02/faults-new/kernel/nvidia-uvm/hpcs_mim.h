#ifndef __HPCS_MIM_H__
#define __HPCS_MIM_H__

#include "uvm_forward_decl.h"
#include "uvm_processors.h"

// Online MIM: HELM metrics on the live fault stream, m/h placement of large allocations
// (enabled with hpcs_online_mim=1)
int hpcs_mim_init(void);
void hpcs_mim_exit(void);
void hpcs_mim_range_created(uvm_va_space_t *va_space, NvU64 start, NvU64 length);
void hpcs_mim_range_destroyed(uvm_va_space_t *va_space, NvU64 start);
void hpcs_mim_va_space_destroyed(uvm_va_space_t *va_space);
void hpcs_mim_record_batch(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context);

// in uvm_policy.c: 'h' sets preferred location CPU + accessed by gpu_id, 'm' unsets both
NV_STATUS hpcs_mim_apply_policy(uvm_va_space_t *va_space,
                                NvU64 base,
                                NvU64 length,
                                uvm_processor_id_t gpu_id,
                                char label);

#endif
