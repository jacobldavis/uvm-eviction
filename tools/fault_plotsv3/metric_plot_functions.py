import pandas as pd
import numpy as np
import gc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import time 
import psutil
import os

from itertools import product
from functools import wraps
from matplotlib.ticker import FixedLocator

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)
matplotlib.use('Agg')
#matplotlib.use('pdf')
mplstyle.use('fast')

#Plot constants
colors = plt.cm.Dark2.colors
markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
BATCH_SIZE = 256
PAGE_SIZE = 4096
all_scalar=1
markersize=2
other_size=12
text_scalar=1

tic = 0
mem = 0

def printt(string, debug=False):
#Debug wrapper for print, adds time/memory info
    if(debug):
        global tic, mem
        toc = time.perf_counter()
        print(f"Time: {round(toc-tic, 2)}")

        process = psutil.Process(os.getpid())
        # Get the memory usage in bytes and convert it to GB
        memory_usage = process.memory_info().rss / (1024 ** 3)
        print(f"Memory: {memory_usage:.2f}GB")
        print(f"Memory Delta: {(memory_usage - mem):.2f}GB")

        #Print actual string and reset time/mem counters
        print(string)
        tic = time.perf_counter()
        mem=memory_usage

def plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        plt.figure(figsize=(10, 6))
        result = func(*args, **kwargs)
        plt.clf()
        return result
    return wrapper

def prune_allocs(df_address_ranges, psize_percentage):
    #Get total alloc size
    df_address_ranges['size'] = np.round((df_address_ranges['adjusted_end'] - df_address_ranges['adjusted_base']) / (1024 * 1024 * 1024), 2)
    device_memory = int(df_address_ranges['size'].sum() * 100 / psize_percentage)

    #Remove any allocations < 1% of device memory
    df_address_ranges['size'] = np.round((df_address_ranges['adjusted_end'] - df_address_ranges['adjusted_base']) / (1024 * 1024 * 1024) / device_memory * 100, 2)
    df_address_ranges = df_address_ranges.loc[df_address_ranges['size'] > 0.02].copy()
    df_address_ranges.sort_values(by='adjusted_base', inplace=True, ascending=False)
    df_address_ranges = df_address_ranges.reset_index()
    single_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)] #['A', ... 'Z']
    double_letters = ["".join(pair) for pair in product(single_letters, repeat=2)] # ['AA', .. AZ, BA... ZZ]
    alloc_names = single_letters + double_letters
    df_address_ranges['label'] = [
        f"{alloc_names[i % len(alloc_names)]} - {size}%"
        for i, size in enumerate(df_address_ranges['size'])]
    return df_address_ranges

def add_application_absolute_metrics_relative_order(df_faults, df_address_ranges, batch_cap):
    printt("App-wide metrics")
       
    group_data = df_faults.groupby(df_faults.index, observed=False).agg(
    min_adjusted_faddr=('adjusted_faddr', 'min'),
    max_adjusted_faddr=('adjusted_faddr', 'max'), 
    group_nunique_faddrs=('adjusted_faddr', 'nunique'),
    group_num_instances=('num_instances', 'sum'),
    group_total_faddrs=('adjusted_faddr', 'size')
    ).fillna(-1)
    
    num_batch_groups = df_faults.index.nunique()
    
    printt("Adding new columns")

    #touchspan ratio
    allocation_length = 12 * (1024**3)
    group_data["touchspan_ratio"] = ((group_data['max_adjusted_faddr'] - group_data['min_adjusted_faddr']) / allocation_length).astype(np.float32)
    
    #faults
    group_data["faults"] = group_data['group_total_faddrs'].astype(np.float32)

    #access density
    group_data["access_density"] = (group_data['group_nunique_faddrs'] / ((group_data['max_adjusted_faddr'] - group_data['min_adjusted_faddr']) // PAGE_SIZE + 1)).astype(np.float32)

    #duplication rate
    group_data["duplicates_inter"] = ((group_data['group_total_faddrs'] - group_data['group_nunique_faddrs'])/group_data['group_num_instances']).astype(np.float32)
    group_data["duplicates_intra"] = ((group_data['group_num_instances'] - group_data['group_nunique_faddrs'])/group_data['group_num_instances']).astype(np.float32)
    group_data["duplicates_all"] = (group_data['duplicates_intra'] + group_data['duplicates_inter']).astype(np.float32)

    #For relative metrics, we need the totals for each batch group
    printt("Getting prop view")
    prop_view = group_data.groupby('fault_group', observed=False)[['faults', 'duplicates_inter', 'duplicates_intra', 'duplicates_all', 'group_nunique_faddrs','access_density','touchspan_ratio']].sum()




    #Calculate stats
    printt("Calculating Stats")
    new_row = {'label': 'all'}
    for metric, metric_str in [('f', 'faults'),('dr_all','duplicates_all'),('dr_inter', 'duplicates_inter'),('dr_intra','duplicates_intra'), ('ws', 'group_nunique_faddrs'), ('d', 'access_density'), ('tr', 'touchspan_ratio')]:
        group_data[f'{metric_str}_prop'] = group_data[metric_str] / prop_view[metric_str]
        label = 'all'
        printt(f"Processing allocation {label}")

        new_row[f'{metric}_mean_{batch_cap}']     = float(group_data[metric_str].mean())
        new_row[f'{metric}_median_{batch_cap}']   = float(group_data[metric_str].median())
        new_row[f'{metric}_max_{batch_cap}']      = float(group_data[metric_str].max())
        new_row[f'{metric}_prop_{batch_cap}']     = float(group_data[f'{metric_str}_prop'].mean())
        new_row[f'{metric}_prop_0s_{batch_cap}']  = float(group_data[f'{metric_str}_prop'].sum() / num_batch_groups)
        if batch_cap > 1: #1 is trivial for this one
            new_row[f'{metric}_prop_max_{batch_cap}'] = group_data[f'{metric_str}_prop'].max()

    # Append or update rows based on condition
    if 'all' in df_address_ranges['label'].values:
    # Update row where 'label' == 'all'
        for key, value in new_row.items():
            df_address_ranges.loc[df_address_ranges['label'] == 'all', key] = value
    else:
    # Append a new row
        new_row_df = pd.DataFrame([new_row])  # Convert dictionary to a DataFrame
        df_address_ranges = pd.concat([df_address_ranges, new_row_df], ignore_index=True)


    #Continue to keep memory footprint in check 
    del prop_view
    del group_data
    gc.collect()
    return df_address_ranges

def prepare_relative_faults(df_faults, df_address_ranges, psize):
    #Remove any allocations < 512MB
    df_address_ranges = prune_allocs(df_address_ranges, psize)

    #Add allocation for each faddr
    printt("Adding allocation column")
    possible_categories = list(df_address_ranges.index.astype(int))
    possible_categories.append(-1)
    print(possible_categories)
    df_faults['allocation'] = pd.Series(-1, dtype=pd.CategoricalDtype(categories=possible_categories))
    for idx, row in df_address_ranges.iterrows():
        printt(f"Adding allocation for faults in allocation {row['label']}")
        df_faults.loc[(df_faults['adjusted_faddr'] >= row['adjusted_base']) & 
                     (df_faults['adjusted_faddr'] < row['adjusted_end']), 'allocation'] = idx
    df_faults['allocation'] = df_faults['allocation'].astype('category')
    

    #Group faults into 'pseudo-batches'; sequential fault groups of size BATCH_SIZE faults
    printt("Adding fault_group")
    df_faults.index = (df_faults.index // BATCH_SIZE).astype(int)
    df_faults.index.name = 'fault_group'
    #df_faults['fault_group'] = df_faults.index // BATCH_SIZE
    printt("Done adding columns")
    return df_faults, df_address_ranges


def compute_relative_metrics(df_faults, df_address_ranges):
    printt("Absolute metrics")
   
    #Init stat columns
    #for metric in ['f', 'dr_all', 'dr_inter', 'dr_intra', 'ws', 'd', 'tr']:
    #    for agg_method in ['mean', 'median', 'max']:
    #        for grouping in [1, 1000]:
    #            if grouping != 1:
    #                df_address_ranges[f'{metric}_{agg_method}_{grouping}']= np.float32(0.0)
                    #df_address_ranges[f'{metric}_{agg_method}_{grouping}'] = df_address_ranges[f'{metric}_{agg_method}_{grouping}'].astype(np.float32)
    #Calculate stats
    old_batchcap = 1
    printt("Calculate Stats")
    for batch_cap in [1, 50, 100, 1000]:
        printt(f"Grouping by fault group with batch_cap {batch_cap}")
        if batch_cap > 1:
            df_faults.index = df_faults.index * old_batchcap // batch_cap
            old_batchcap = batch_cap
        
        group_data = df_faults.groupby(['fault_group', 'allocation'], observed=False).agg(
        allocation=('allocation', 'first'),
        min_adjusted_faddr=('adjusted_faddr', 'min'),
        max_adjusted_faddr=('adjusted_faddr', 'max'), 
        group_nunique_faddrs=('adjusted_faddr', 'nunique'),
        group_total_faddrs=('adjusted_faddr', 'size'), 
        group_num_instances=('num_instances', 'sum')
        ).fillna(-1)
        group_data = group_data[group_data['allocation'] != -1]
        
        num_batch_groups = df_faults.index.nunique()
        
        printt("Adding new columns")

        #touchspan ratio
        for idx, row in df_address_ranges.iterrows():
            allocation_length = row['adjusted_end'] - row['adjusted_base']
            group_data.loc[group_data['allocation']==idx, "touchspan"] = (group_data['max_adjusted_faddr'] - group_data['min_adjusted_faddr']) 
            group_data.loc[group_data['allocation']==idx, "touchspan_ratio"] = ((group_data['max_adjusted_faddr'] - group_data['min_adjusted_faddr']) / allocation_length).astype(np.float32)
        
        #faults
        group_data["faults"] = group_data['group_total_faddrs'].astype(np.float32)

        #access density
        group_data["access_density"] = (group_data['group_nunique_faddrs'] / ((group_data['max_adjusted_faddr'] - group_data['min_adjusted_faddr']) // PAGE_SIZE + 1)).astype(np.float32)

        #duplicates
        group_data["duplicates_inter"] = (group_data['group_total_faddrs'] - group_data['group_nunique_faddrs'])
        group_data["duplicates_intra"] = (group_data['group_num_instances'] - group_data['group_nunique_faddrs'])
        group_data["duplicates_all"] = (group_data['duplicates_intra'] + group_data['duplicates_inter']).astype(np.float32)
        #duplicates per addr
        group_data["dup_ratio_inter"] = ((group_data['group_total_faddrs'] - group_data['group_nunique_faddrs']) / group_data['group_nunique_faddrs'])
        group_data["dup_ratio_intra"] = (group_data['group_num_instances'] - group_data['group_nunique_faddrs']) / group_data['group_nunique_faddrs']
        group_data["dup_ratio_all"] = (group_data['duplicates_intra'] + group_data['duplicates_inter']).astype(np.float32)
        

        #For relative metrics, we need the totals for each batch group
        printt("Getting rel view")
        rel_view = group_data.groupby('fault_group', observed=False)[['faults', 'duplicates_inter', 'duplicates_intra', 'duplicates_all', 'dup_ratio_intra', 'dup_ratio_inter', 'dup_ratio_all', 'group_nunique_faddrs','touchspan', 'touchspan_ratio', 'access_density']].sum()


        #Calculate stats
        printt("Calculating Stats")
        for metric, metric_str in [('f', 'faults'),('dc_all','duplicates_all'),('dc_inter', 'duplicates_inter'),('dc_intra','duplicates_intra'), ('dr_all','dup_ratio_all'),('dr_inter', 'dup_ratio_inter'),('dr_intra','dup_ratio_intra'), ('ws', 'group_nunique_faddrs'), ('d', 'access_density'), ('ts', 'touchspan'), ('tr', 'touchspan_ratio')]:
            group_data[f'{metric_str}_rel'] = group_data[metric_str] / rel_view[metric_str]
            group_data[f'{metric_str}_sum'] = rel_view[metric_str]
            for idx, row in df_address_ranges.iterrows():
                group_data_view = group_data[group_data['allocation']==idx]
                label = row['label']
                printt(f"Processing allocation {label}")
                group_data_view[metric_str] = group_data_view[metric_str].astype(np.float32)
                group_data_view[f'{metric_str}_rel'] = group_data_view[f'{metric_str}_rel'].astype(np.float32)

                if metric_str in ['faults', 'duplicates_inter', 'duplicates_intra', 'duplicates_all', 'group_nunique_faddrs','touchspan']:
                    df_address_ranges.loc[idx, f'{metric}_rel_mean_{batch_cap}']  = group_data_view[f'{metric_str}_rel'].mean()
                    df_address_ranges.loc[idx, f'{metric}_rel_median_{batch_cap}']  = group_data_view[f'{metric_str}_rel'].median()
                    df_address_ranges.loc[idx, f'{metric}_rel_max_{batch_cap}'] = group_data_view[f'{metric_str}_rel'].max()
                    df_address_ranges.loc[idx, f'{metric}_rel_mean_0s_{batch_cap}']  = group_data_view[f'{metric_str}_rel'].sum() / num_batch_groups
                    if batch_cap > 1: #1 is trivial for this one
                        df_address_ranges.loc[idx, f'{metric}_rel_max_{batch_cap}'] = group_data_view[f'{metric_str}_rel'].max() 
                else: 
                    df_address_ranges.loc[idx, f'{metric}_mean_{batch_cap}']     = group_data_view[metric_str].mean()
                    df_address_ranges.loc[idx, f'{metric}_median_{batch_cap}']   = group_data_view[metric_str].median()
                    df_address_ranges.loc[idx, f'{metric}_max_{batch_cap}']      = group_data_view[metric_str].max()
    

        #Continue to keep memory footprint in check 
        del rel_view
        del group_data
        gc.collect()
    return df_address_ranges


@plotter
def print_relative_metrics_relative_order(df_faults, df_prefetch, df_evictions, df_address_ranges, output_state, plot_end, output_dir=None):
    printt("configuring print")
    output_path = ""
    if output_dir is None:
        output_path = output_state.get_output_path("metrics_stats_relative")
    else:
        output_path = output_state.get_output_path(output_dir)

    df_faults, df_address_ranges = prepare_relative_faults(df_faults, df_address_ranges, output_state.psize)
    df_address_ranges = compute_relative_metrics(df_faults, df_address_ranges)

    # Write the entire DataFrame to a text file
    df_address_ranges.to_csv(output_path, index=False)

    printt(df_address_ranges)
    printt(f"Stats saved to {output_path}")


@plotter
def print_relative_metrics_relative_order_per_kernel(df_faults, df_prefetch, df_evictions, df_address_ranges, output_state, plot_end, output_dir=None):
    printt("configuring print")
    output_path = ""
    if output_dir is None:
        output_path = output_state.get_output_path("metrics_stats_relative_kernel")
    else:
        output_path = output_state.get_output_path(f"{output_dir}_kernel")

    if 'kernel' not in df_faults.columns:
        raise ValueError("log has no kernel markers; collect it with KERNEL_MARKERS = True")

    df_faults, df_address_ranges = prepare_relative_faults(df_faults, df_address_ranges, output_state.psize)

    kernel_metrics = []
    for kernel, df_kernel_faults in df_faults.groupby('kernel', observed=True):
        printt(f"Processing kernel {kernel}")
        df_kernel_ranges = compute_relative_metrics(df_kernel_faults.copy(), df_address_ranges.copy())
        df_kernel_ranges.insert(0, 'kernel', kernel)
        df_kernel_ranges.insert(1, 'launches', df_kernel_faults['kernel_id'].nunique())
        df_kernel_ranges.insert(2, 'kernel_faults', len(df_kernel_faults))
        kernel_metrics.append(df_kernel_ranges)
        del df_kernel_faults
        gc.collect()

    df_kernel_metrics = pd.concat(kernel_metrics, ignore_index=True)
    df_kernel_metrics.to_csv(output_path, index=False)

    printt(df_kernel_metrics)
    printt(f"Stats saved to {output_path}")



@plotter
def print_absolute_metrics_relative_order(df_faults, df_prefetch, df_evictions, df_address_ranges, output_state, plot_end, output_dir=None):
    printt("configuring print")
    output_path = ""
    if output_dir is None:
        output_path = output_state.get_output_path("metrics_stats")
    else:
        output_path = output_state.get_output_path(output_dir)


    #Remove any allocations < 512MB
    df_address_ranges = prune_allocs(df_address_ranges, output_state.psize)

    #Add allocation for each faddr
    printt("Adding allocation column")
    possible_categories = list(df_address_ranges.index.astype(int))
    possible_categories.append(-1)
    print(possible_categories)
    df_faults['allocation'] = pd.Series(-1, dtype=pd.CategoricalDtype(categories=possible_categories))
    for idx, row in df_address_ranges.iterrows():
        printt(f"Adding allocation for faults in allocation {row['label']}")
        df_faults.loc[(df_faults['adjusted_faddr'] >= row['adjusted_base']) & 
                     (df_faults['adjusted_faddr'] < row['adjusted_end']), 'allocation'] = idx
    df_faults['allocation'] = df_faults['allocation'].astype('category')
    

    #Group faults into 'pseudo-batches'; sequential fault groups of size BATCH_SIZE faults
    printt("Adding fault_group")
    df_faults.index = (df_faults.index // BATCH_SIZE).astype(int)
    df_faults.index.name = 'fault_group'
    #df_faults['fault_group'] = df_faults.index // BATCH_SIZE
    printt("Done adding columns")

    
    printt("Absolute metrics")
   
    #Init stat columns
    for metric in ['f', 'dr_all', 'dr_inter', 'dr_intra', 'ws', 'd', 'tr']:
        for agg_method in ['mean', 'median', 'max', 'prop', 'prop_max']:
            for grouping in [1, 1000]:
                if grouping != 1 or agg_method != 'prop_max':
                    df_address_ranges[f'{metric}_{agg_method}_{grouping}']= np.float32(0.0)
                    #df_address_ranges[f'{metric}_{agg_method}_{grouping}'] = df_address_ranges[f'{metric}_{agg_method}_{grouping}'].astype(np.float32)
    #Calculate stats
    printt("Calculate Stats")
    for batch_cap in [1, 1000]:
        printt(f"Grouping by fault group with batch_cap {batch_cap}")
        if batch_cap > 1:
            df_faults.index = df_faults.index // batch_cap
        
        group_data = df_faults.groupby(['fault_group', 'allocation'], observed=False).agg(
        allocation=('allocation', 'first'),
        min_adjusted_faddr=('adjusted_faddr', 'min'),
        max_adjusted_faddr=('adjusted_faddr', 'max'), 
        group_nunique_faddrs=('adjusted_faddr', 'nunique'),
        group_total_faddrs=('adjusted_faddr', 'size'), 
        group_num_instances=('num_instances', 'sum')
        ).fillna(-1)
        group_data = group_data[group_data['allocation'] != -1]
        
        num_batch_groups = df_faults.index.nunique()
        
        printt("Adding new columns")
        #fix by batch_cap
        #group_data['group_nunique_faddrs'] = group_data['group_nunique_faddrs'] / batch_cap
        #group_data['group_total_faddrs'] = group_data['group_total_faddrs'] / batch_cap

        #touchspan ratio
        for idx, row in df_address_ranges.iterrows():
            allocation_length = row['adjusted_end'] - row['adjusted_base']
            group_data.loc[group_data['allocation']==idx, "touchspan_ratio"] = ((group_data['max_adjusted_faddr'] - group_data['min_adjusted_faddr']) / allocation_length).astype(np.float32)
        
        #faults
        group_data["faults"] = group_data['group_total_faddrs'].astype(np.float32)

        #access density
        group_data["access_density"] = (group_data['group_nunique_faddrs'] / ((group_data['max_adjusted_faddr'] - group_data['min_adjusted_faddr']) // PAGE_SIZE + 1)).astype(np.float32)

        #duplication rate
        group_data["duplicates_inter"] = ((group_data['group_total_faddrs'] - group_data['group_nunique_faddrs'])/group_data['group_num_instances']).astype(np.float32)
        group_data["duplicates_intra"] = ((group_data['group_num_instances'] - group_data['group_nunique_faddrs'])/group_data['group_num_instances']).astype(np.float32)
        group_data["duplicates_all"] = (group_data['duplicates_intra'] + group_data['duplicates_inter']).astype(np.float32)

        #For relative metrics, we need the totals for each batch group
        printt("Getting prop view")
        prop_view = group_data.groupby('fault_group', observed=False)[['faults', 'duplicates_inter', 'duplicates_intra', 'duplicates_all', 'group_nunique_faddrs','access_density','touchspan_ratio']].sum()


        #Calculate stats
        printt("Calculating Stats")
        for metric, metric_str in [('f', 'faults'),('dr_all','duplicates_all'),('dr_inter', 'duplicates_inter'),('dr_intra','duplicates_intra'), ('ws', 'group_nunique_faddrs'), ('d', 'access_density'), ('tr', 'touchspan_ratio')]:
            group_data[f'{metric_str}_prop'] = group_data[metric_str] / prop_view[metric_str]
            for idx, row in df_address_ranges.iterrows():
                group_data_view = group_data[group_data['allocation']==idx]
                label = row['label']
                printt(f"Processing allocation {label}")
                group_data_view[metric_str] = group_data_view[metric_str].astype(np.float32)
                group_data_view[f'{metric_str}_prop'] = group_data_view[f'{metric_str}_prop'].astype(np.float32)

                df_address_ranges.loc[idx, f'{metric}_mean_{batch_cap}']     = group_data_view[metric_str].mean()
                df_address_ranges.loc[idx, f'{metric}_median_{batch_cap}']   = group_data_view[metric_str].median()
                df_address_ranges.loc[idx, f'{metric}_max_{batch_cap}']      = group_data_view[metric_str].max()
                df_address_ranges.loc[idx, f'{metric}_prop_{batch_cap}']     = group_data_view[f'{metric_str}_prop'].mean()
                df_address_ranges.loc[idx, f'{metric}_prop_0s_{batch_cap}']  = group_data_view[f'{metric_str}_prop'].sum() / num_batch_groups
                if batch_cap > 1: #1 is trivial for this one
                    df_address_ranges.loc[idx, f'{metric}_prop_max_{batch_cap}'] = group_data_view[f'{metric_str}_prop'].max()  

        
        #Add metrics for 'all'
        df_address_ranges = add_application_absolute_metrics_relative_order(df_faults, df_address_ranges, batch_cap)
           
        #Continue to keep memory footprint in check 
        del prop_view
        del group_data
        gc.collect()

    
    # Write the entire DataFrame to a text file
    df_address_ranges.to_csv(output_path, index=False)

    printt(df_address_ranges)
    printt(f"Stats saved to {output_path}")


@plotter
def plot_absolute_metrics_relative_order(df_faults, df_prefetch, df_evictions, df_address_ranges, output_state, plot_end, output_dir=None):
    printt("configuring plot")
    output_path = ""
    if output_dir is None:
        output_path = output_state.get_output_path("metrics")
    else:
        output_path = output_state.get_output_path(output_dir)

    #Set axes
    fig, (ax1, ax2) = plt.subplots(2, 2)

    #Remove any allocations < 512MB
    df_address_ranges = prune_allocs(df_address_ranges, output_state.psize)

    #Add allocation for each faddr
    printt("Adding allocation column")
    df_faults['allocation'] = 0
    for idx, row in df_address_ranges.iterrows():
        printt(f"Adding allocation for faults in allocation {row['label']}")
        df_faults.loc[(df_faults['adjusted_faddr'] >= row['adjusted_base']) & 
                     (df_faults['adjusted_faddr'] < row['adjusted_end']),'allocation'] = idx
    df_faults['allocation'] = df_faults['allocation'].astype('category') #Reduces memory footprint

    #Group faults into 'pseudo-batches'; sequential fault groups of size BATCH_SIZE faults
    printt("Adding fault_group")
    df_faults['fault_group'] = df_faults['fault_group'] // BATCH_SIZE
    printt("Done adding columns")

    for idx, row in df_address_ranges.iterrows():
        label = row['label']
        printt(f"Processing allocation {label}")
        
        
        printt("Grouping by fault group")
        alloc_faults_df = df_faults[df_faults['allocation']==idx]
        fault_groups = alloc_faults_df['fault_group'].unique()

        alloc_faults_df['group_nunique'] = alloc_faults_df.groupby(['fault_group'])['adjusted_faddr'].transform('nunique').astype('int16')
        alloc_faults_df['group_total'] = alloc_faults_df.groupby(['fault_group'])['adjusted_faddr'].transform('size').astype('int16')
        group_data = alloc_faults_df.groupby(['fault_group']).agg(
        min_adjusted_faddr=('adjusted_faddr', 'min'),
        max_adjusted_faddr=('adjusted_faddr', 'max'), 
        group_nunique_faddrs=('group_nunique', 'first'),
        group_total_faddrs=('group_total', 'first')
        )
        #Alloc faults no longer needed, reduce memory footprint
        del alloc_faults_df

        allocation_length = row["adjusted_end"] - row["adjusted_base"]

        #touchspan ratio
        group_data["touchspan_ratio"] = ((group_data['max_adjusted_faddr'] - group_data['min_adjusted_faddr']) / allocation_length).astype(np.float32)
        #access density
        group_data["access_density"] = (group_data['group_nunique_faddrs'] / ((group_data['max_adjusted_faddr'] - group_data['min_adjusted_faddr']) // PAGE_SIZE + 1)).astype(np.float32)

        #duplication rate
        group_data["duplicates"] = (group_data['group_total_faddrs'] - group_data['group_nunique_faddrs']).astype('int16')

        current_color = colors[idx % len(colors)]
        current_marker = markers[idx % len(markers)]
        
        printt(f"Plotting allocation {label}")

        #Plot metrics for given allocation
        ax1[0].scatter(fault_groups, group_data['duplicates'], marker=current_marker, color=current_color, s=markersize,  label=label)
        ax2[0].scatter(fault_groups, group_data['group_nunique_faddrs'], marker=current_marker, color=current_color, s=markersize,  label=label)
        ax1[1].scatter(fault_groups, group_data['access_density'], marker=current_marker, color=current_color, s=markersize,  label=label)
        ax2[1].scatter(fault_groups, group_data['touchspan_ratio'], marker=current_marker, color=current_color, s=markersize,  label=label)
        
        #Continue to keep memory footprint in check 
        del group_data
        gc.collect()

    #General axes settings
    for ax, density_name in [(ax1[0], "Fault Duplication Rate"), (ax2[0], "Working Set Delta"), (ax1[1], "Fault Density"), (ax2[1], "Fault Span Ratio")]:
        ax.tick_params(axis='both', labelsize=all_scalar*other_size)
        ax.set_xlabel('Time (Batch Group)', fontsize=all_scalar*other_size)
        ax.set_ylabel(density_name, fontsize=all_scalar*other_size)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    printt("Finishing plot")

    plt.xlabel('Event Order'
    )
    # Shrink current axis's height by 10% on the bottom
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    printt("plotting and saving")
    plt.savefig(output_path, format='png')
    printt(f"Plot saved to {output_path}")

@plotter
def plot_relative_metrics_relative_order(df_faults, df_prefetch, df_evictions, df_address_ranges, output_state, plot_end, output_dir=None):
    printt("configuring plot")
    output_path = ""
    if output_dir is None:
        output_path = output_state.get_output_path("access_frequency")
    else:
        output_path = output_state.get_output_path(output_dir)
    
    #Remove any allocations < 512MB
    df_address_ranges = prune_allocs(df_address_ranges, output_state.psize)

    #Add allocation for each faddr
    printt("Adding allocation column")
    df_faults['allocation'] = ''
    for idx, row in df_address_ranges.iterrows():
        printt(f"Adding allocation for faults in allocation {row['label']}")
        df_faults.loc[(df_faults['adjusted_faddr'] >= row['adjusted_base']) & 
                     (df_faults['adjusted_faddr'] < row['adjusted_end']),'allocation'] = row['label']
    df_faults['allocation'] = df_faults['allocation'].astype('category') #Reduces memory footprint

    #Group faults into 'pseudo-batches'; sequential fault groups of size BATCH_SIZE faults
    printt("Adding fault_group")
    df_faults['fault_group'] = (df_faults.index // (BATCH_SIZE * 1000)).astype('category')
    printt("Done adding columns")
    
    # Assume df is your DataFrame
    # Step 1: Group by fault_group and alloc, and count unique adjusted_faddr
    grouped = df_faults.groupby(['fault_group', 'allocation'], observed=False)['adjusted_faddr'].nunique().reset_index()
    
    # Step 2: Normalize by fault_group to get the proportions
    grouped['proportion'] = grouped.groupby('fault_group', observed=False)['adjusted_faddr'].transform(lambda x: x / x.sum())

    # Step 3: Pivot the DataFrame so that alloc categories become columns
    pivot_df = grouped.pivot(index='fault_group', columns='allocation', values='proportion').fillna(0)

    # Get stat values
    df_address_ranges['num_unique_faults'] = 0
    df_address_ranges['avg_proportion'] = 0.0
    df_address_ranges['max_proportion'] = 0.0
    for idx, row in df_address_ranges.iterrows():
        alloc_group = grouped[grouped['allocation'] == row['label']]
        df_address_ranges.loc[idx,'num_unique_faults'] = alloc_group['adjusted_faddr'].sum()
        df_address_ranges.loc[idx,'avg_proportion']    = alloc_group['proportion'].mean()
        df_address_ranges.loc[idx,'max_proportion'] = alloc_group['proportion'].max()
    df_address_ranges['perc_unique_faults'] = df_address_ranges['num_unique_faults'] / df_address_ranges['num_unique_faults'].sum() * 100.0
    df_address_ranges.sort_values(by='num_unique_faults', inplace=True)
    print(df_address_ranges[['label','num_unique_faults', 'perc_unique_faults', 'avg_proportion', 'max_proportion']])

    #Free up memory
    del grouped
    del df_faults

    # Step 4: Prepare for stackplot
    x = pivot_df.index  # This represents the fault_group
    y = pivot_df.values.T  # Stackplot expects categories as rows
    
    # Plot the stackplot
    plt.stackplot(x, *y, labels=pivot_df.columns)
    del pivot_df

    # Add legend and labels
    #plt.legend(loc='upper left', title='Alloc')
    plt.ylabel('Unique Faults Distribution')
    #plt.title('Unique Faults Proportion per 1000 Batches')

    printt("Finishing plot")

    plt.xlabel('1000 Batches')
    # Shrink current axis's height by 10% on the bottom
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.1, -0.05), fancybox=True, shadow=True, ncol=5)

    printt("plotting and saving")
    plt.savefig(output_path, format='png')
    printt(f"Plot saved to {output_path}")




if __name__ == "__main__":
    printt("This is a utility class; you are probably looking for metric_plot.py")
