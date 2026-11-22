#include "uvm_common.h"
#include "uvm_gpu.h"
#include "uvm_va_space.h"
#include "nv-kthread-q.h"
#include "hpcs_mim.h"

static unsigned int hpcs_online_mim = 0;
static unsigned int hpcs_mim_window = 1000;
static unsigned int hpcs_mim_min_mb = 1024;
static unsigned int hpcs_mim_stable = 1;
module_param(hpcs_online_mim, uint, S_IRUGO);
module_param(hpcs_mim_window, uint, S_IRUGO);
module_param(hpcs_mim_min_mb, uint, S_IRUGO);
module_param(hpcs_mim_stable, uint, S_IRUGO);

#define MIM_MAX_ALLOCS 16
#define MIM_MAX_WINDOW 4096
#define MIM_SLOTS (2 * MIM_MAX_WINDOW)
// metrics are fixed point, parts per million
#define MIM_SCALE 1000000ull
// MIM's thresholds (parse_metrics_relative.py)
#define MIM_STREAM_TR_MAX 5000ull
#define MIM_STREAM_D_MIGRATE 130000ull
#define MIM_NONSTREAM_D_MIGRATE 20000ull
#define MIM_DR_MIGRATE MIM_SCALE

typedef struct
{
    // NULL when the slot is free
    uvm_va_space_t *va_space;
    NvU64 start;
    NvU64 length;
    uvm_processor_id_t gpu_id;

    // current window
    NvU32 faults;
    NvU64 instances;
    NvU64 unique;
    NvU64 min_addr;
    NvU64 max_addr;
    // page number + 1 per slot, 0 is empty
    NvU64 *pages;

    // sums over closed windows
    NvU64 windows;
    NvU64 d_sum;
    NvU64 dr_sum;
    NvU64 tr_sum;

    char label;
    char candidate;
    NvU32 candidate_windows;
    bool pending;
    // stop deciding after a failed apply
    bool failed;
    nv_kthread_q_item_t item;
} hpcs_mim_alloc_t;

static hpcs_mim_alloc_t mim_allocs[MIM_MAX_ALLOCS];
static DEFINE_SPINLOCK(mim_lock);
static nv_kthread_q_t mim_q;
static bool mim_ready = false;

static NvU32 window_size(void)
{
    if (hpcs_mim_window == 0)
        return 1;
    return hpcs_mim_window > MIM_MAX_WINDOW ? MIM_MAX_WINDOW : hpcs_mim_window;
}

static void reset_window(hpcs_mim_alloc_t *a)
{
    a->faults = 0;
    a->instances = 0;
    a->unique = 0;
    a->min_addr = ~0ull;
    a->max_addr = 0;
    memset(a->pages, 0, 2 * window_size() * sizeof(a->pages[0]));
}

// returns true if the page wasn't seen in this window
static bool add_page(hpcs_mim_alloc_t *a, NvU64 page)
{
    NvU32 slots = 2 * window_size();
    NvU32 i = (NvU32)((page * 0x9E3779B97F4A7C15ull) >> 32) % slots;

    while (a->pages[i]) {
        if (a->pages[i] == page + 1)
            return false;
        i = (i + 1) % slots;
    }
    a->pages[i] = page + 1;
    return true;
}

// mim_lock held
static void close_window(hpcs_mim_alloc_t *a)
{
    NvU64 span = a->max_addr - a->min_addr;
    NvU64 d_mean, dr_mean, tr_mean;
    bool streaming;
    char label;

    a->windows++;
    a->d_sum += a->unique * MIM_SCALE / (span / 4096 + 1);
    a->dr_sum += (a->instances - a->unique) * MIM_SCALE / a->unique;
    a->tr_sum += span * MIM_SCALE / a->length;

    d_mean = a->d_sum / a->windows;
    dr_mean = a->dr_sum / a->windows;
    tr_mean = a->tr_sum / a->windows;

    // same rules as predict_strat_hardcode, without the knapsack ('d')
    streaming = dr_mean >= MIM_DR_MIGRATE && dr_mean <= 1000 * MIM_SCALE && tr_mean <= MIM_STREAM_TR_MAX;
    if (streaming)
        label = d_mean >= MIM_STREAM_D_MIGRATE ? 'm' : 'h';
    else
        label = (d_mean > MIM_NONSTREAM_D_MIGRATE && dr_mean > MIM_DR_MIGRATE) ? 'm' : 'h';

    if (label == a->candidate) {
        a->candidate_windows++;
    }
    else {
        a->candidate = label;
        a->candidate_windows = 1;
    }

    if (a->candidate != a->label && a->candidate_windows >= hpcs_mim_stable && !a->pending && !a->failed) {
        a->pending = true;
        nv_kthread_q_schedule_q_item(&mim_q, &a->item);
    }

    reset_window(a);
}

void hpcs_mim_record_batch(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i, j;

    if (!hpcs_online_mim || !mim_ready)
        return;

    spin_lock(&mim_lock);
    for (i = 0; i < batch_context->num_coalesced_faults; i++) {
        uvm_fault_buffer_entry_t *entry = batch_context->ordered_fault_cache[i];
        NvU64 addr = entry->fault_address;
        hpcs_mim_alloc_t *a = NULL;

        for (j = 0; j < MIM_MAX_ALLOCS; j++) {
            if (mim_allocs[j].va_space == entry->va_space && entry->va_space &&
                addr >= mim_allocs[j].start && addr - mim_allocs[j].start < mim_allocs[j].length) {
                a = &mim_allocs[j];
                break;
            }
        }
        if (!a)
            continue;

        a->gpu_id = gpu->id;
        a->instances += entry->num_instances;
        if (add_page(a, addr / 4096))
            a->unique++;
        if (addr < a->min_addr)
            a->min_addr = addr;
        if (addr > a->max_addr)
            a->max_addr = addr;
        if (++a->faults >= window_size())
            close_window(a);
    }
    spin_unlock(&mim_lock);
}

static void apply_decision(void *args)
{
    hpcs_mim_alloc_t *a = args;
    uvm_va_space_t *va_space;
    uvm_processor_id_t gpu_id;
    NvU64 start, length, windows, d_mean, dr_mean, tr_mean;
    NV_STATUS status;
    char label;

    spin_lock(&mim_lock);
    if (!a->va_space || !a->pending) {
        spin_unlock(&mim_lock);
        return;
    }
    va_space = a->va_space;
    start = a->start;
    length = a->length;
    gpu_id = a->gpu_id;
    label = a->candidate;
    windows = a->windows;
    d_mean = a->d_sum / windows;
    dr_mean = a->dr_sum / windows;
    tr_mean = a->tr_sum / windows;
    spin_unlock(&mim_lock);

    // va_space stays valid: hpcs_mim_va_space_destroyed flushes this queue first
    status = hpcs_mim_apply_policy(va_space, start, length, gpu_id, label);

    spin_lock(&mim_lock);
    if (a->va_space == va_space && a->start == start) {
        if (status == NV_OK)
            a->label = label;
        else
            a->failed = true;
        a->pending = false;
    }
    spin_unlock(&mim_lock);

    printk(KERN_INFO "hpcs_mim: 0x%llx (%llu MB) -> %c after %llu windows (d %llu dr %llu tr %llu ppm): %s\n",
           start, length >> 20, label, windows, d_mean, dr_mean, tr_mean, nvstatusToString(status));
}

void hpcs_mim_range_created(uvm_va_space_t *va_space, NvU64 start, NvU64 length)
{
    hpcs_mim_alloc_t *free_slot = NULL;
    NvU32 i;

    if (!hpcs_online_mim || !mim_ready || length < ((NvU64)hpcs_mim_min_mb << 20))
        return;

    spin_lock(&mim_lock);
    for (i = 0; i < MIM_MAX_ALLOCS; i++) {
        hpcs_mim_alloc_t *a = &mim_allocs[i];

        // split pieces of an allocation already tracked
        if (a->va_space == va_space && start < a->start + a->length && a->start < start + length) {
            spin_unlock(&mim_lock);
            return;
        }
        if (!a->va_space && !free_slot)
            free_slot = a;
    }

    if (free_slot) {
        free_slot->va_space = va_space;
        free_slot->start = start;
        free_slot->length = length;
        free_slot->windows = 0;
        free_slot->d_sum = 0;
        free_slot->dr_sum = 0;
        free_slot->tr_sum = 0;
        free_slot->label = 'm';
        free_slot->candidate = 'm';
        free_slot->candidate_windows = 0;
        free_slot->pending = false;
        free_slot->failed = false;
        reset_window(free_slot);
    }
    spin_unlock(&mim_lock);

    if (free_slot)
        printk(KERN_INFO "hpcs_mim: tracking 0x%llx (%llu MB)\n", start, length >> 20);
    else
        printk(KERN_INFO "hpcs_mim: no free slot for 0x%llx (%llu MB)\n", start, length >> 20);
}

void hpcs_mim_range_destroyed(uvm_va_space_t *va_space, NvU64 start)
{
    NvU32 i;

    if (!mim_ready)
        return;

    spin_lock(&mim_lock);
    for (i = 0; i < MIM_MAX_ALLOCS; i++) {
        if (mim_allocs[i].va_space == va_space && mim_allocs[i].start == start) {
            mim_allocs[i].va_space = NULL;
            mim_allocs[i].pending = false;
        }
    }
    spin_unlock(&mim_lock);
}

void hpcs_mim_va_space_destroyed(uvm_va_space_t *va_space)
{
    NvU32 i;

    if (!mim_ready)
        return;

    spin_lock(&mim_lock);
    for (i = 0; i < MIM_MAX_ALLOCS; i++) {
        if (mim_allocs[i].va_space == va_space) {
            mim_allocs[i].va_space = NULL;
            mim_allocs[i].pending = false;
        }
    }
    spin_unlock(&mim_lock);

    // wait for a decision already applying to this va_space
    nv_kthread_q_flush(&mim_q);
}

int hpcs_mim_init(void)
{
    NvU32 i;

    if (!hpcs_online_mim)
        return 0;

    for (i = 0; i < MIM_MAX_ALLOCS; i++) {
        mim_allocs[i].pages = kvmalloc(MIM_SLOTS * sizeof(NvU64), GFP_KERNEL);
        if (!mim_allocs[i].pages)
            goto fail;
        mim_allocs[i].va_space = NULL;
        nv_kthread_q_item_init(&mim_allocs[i].item, apply_decision, &mim_allocs[i]);
    }

    if (nv_kthread_q_init(&mim_q, "uvm_hpcs_mim") != 0)
        goto fail;

    mim_ready = true;
    printk(KERN_INFO "hpcs_mim: online MIM enabled (window %u faults, allocations >= %u MB, stable %u)\n",
           window_size(), hpcs_mim_min_mb, hpcs_mim_stable);
    return 0;

fail:
    for (i = 0; i < MIM_MAX_ALLOCS; i++) {
        kvfree(mim_allocs[i].pages);
        mim_allocs[i].pages = NULL;
    }
    printk(KERN_ERR "hpcs_mim: init failed, online MIM disabled\n");
    return -ENOMEM;
}

void hpcs_mim_exit(void)
{
    NvU32 i;

    if (!mim_ready)
        return;

    mim_ready = false;
    nv_kthread_q_stop(&mim_q);
    for (i = 0; i < MIM_MAX_ALLOCS; i++) {
        kvfree(mim_allocs[i].pages);
        mim_allocs[i].pages = NULL;
    }
}
