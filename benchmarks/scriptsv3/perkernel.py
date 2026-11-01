#!/usr/bin/python3
"""
perkernel.py - brute-force per-kernel placement policies next to the usual whole-run policies.

A per-kernel policy is one d/m/h string per kernel joined with ':' (e.g. "dmm:mdd"); accpol.h applies each
kernel's string before its launch. Results go to <app>_perkernel2_numa_pref.data in each strategied app
directory (PERF_TAG), separate from the paper's <app>_numa_pref.data; tools/fault_plotsv3/perkernel_compare.py
reads them. Whole-run results in the older <app>_perkernel_numa_pref.data are reused.

Search space per app (allocation indices follow the setAllocationPolicy() order in the app):
  - whole-run policies, the app-level optimum to beat
  - per kernel, d/m/h choices for the allocations that kernel touches; untouched allocations are
    unpinned ('m') during that kernel
  - GRAMSCHM relaunches its kernels ~N times, so 'd' stays fixed across kernels and only m/h switch
  - sgemm is one cuBLAS call, so it only checks the per-kernel path: the baseline and probes ("xyz") are
    rerun as "xyz:" and must time the same (control)

To keep it tractable, per app and psize:
  1. time the all-m baseline; every later run is cut at TIMEOUT_FACTOR x the baseline's wall time
  2. probe each allocation with h and with d (m everywhere else); a letter whose probe does not finish
     is dropped for that allocation from the whole search
  3. run the whole-run and per-kernel policies built from the remaining letters, minus per-kernel policies
     that give every kernel the same string (the whole-run policy again)
Policies already in the .data file are not rerun, so an interrupted search resumes where it stopped.

Usage: TIMEOUT=12000 python3 -u perkernel.py [--apps 2MM GRAMSCHM sgemm] [--psizes 100 125] [--policies P ...] [--plan]
"""
import argparse
import itertools
import os
import re

import config
from experiments_nolog import Experiment

# allocations each kernel touches, by allocation index, in launch order
APPS = {
    '2MM':      {'allocs': 5, 'kernels': [{0, 1, 2}, {2, 3, 4}], 'switch_d': True},
    'GRAMSCHM': {'allocs': 3, 'kernels': [{0, 1}, {0, 1, 2}, {0, 1, 2}], 'switch_d': False},
    'sgemm':    {'allocs': 3, 'kernels': [{0, 1, 2}], 'switch_d': True, 'control': True},
}
TIMEOUT_FACTOR = 1.25
PERF_TAG = 'perkernel2'


def whole_run_policies(app, letters):
    return [''.join(p) for p in itertools.product(*letters)]


def kernel_choices(app, kernel, letters):
    # every allowed choice for the touched allocations; untouched ones unpinned
    spec = APPS[app]
    slots = [letters[i] if i in spec['kernels'][kernel] else 'm' for i in range(spec['allocs'])]
    return [''.join(p) for p in itertools.product(*slots)]


def per_kernel_policies(app, letters):
    spec = APPS[app]
    if spec['switch_d']:
        per_kernel = itertools.product(*(kernel_choices(app, k, letters) for k in range(len(spec['kernels']))))
        # trailing ':' keeps a one-kernel policy on the per-kernel path
        return [':'.join(p) + (':' if len(p) == 1 else '') for p in per_kernel]
    # d fixed: each allocation is 'd' for every kernel, or m/h per kernel where it is touched
    per_alloc = []
    for i in range(spec['allocs']):
        touched = [k for k, s in enumerate(spec['kernels']) if i in s]
        options = [['d'] * len(spec['kernels'])] if 'd' in letters[i] else []
        for chosen in itertools.product(letters[i].replace('d', ''), repeat=len(touched)):
            column = ['m'] * len(spec['kernels'])
            for k, letter in zip(touched, chosen):
                column[k] = letter
            options.append(column)
        per_alloc.append(options)
    return [':'.join(''.join(column[k] for column in columns) for k in range(len(spec['kernels'])))
            for columns in itertools.product(*per_alloc)]


def search_space(app, letters, probes):
    whole = whole_run_policies(app, letters)
    if APPS[app].get('control'):
        return whole + [p + ':' for p in probes]
    # the same string for every kernel is the whole-run policy again
    return whole + [p for p in per_kernel_policies(app, letters)
                    if len(set(p.rstrip(':').split(':'))) > 1 or p.rstrip(':').split(':')[0] not in whole]


def data_blocks(app):
    """One text block per run, from the app's .data files. The first per-kernel 'd' kept its managed copy
    alive and ran 3-6% slow, so per-kernel runs count only from the 'perkernel2' file written since the
    fix; whole-run runs count from both."""
    for tag, keep_per_kernel in [('perkernel', False), (PERF_TAG, True)]:
        path = f'{config.BENCHDIR}/strategied/{app}/{app.lower()}_{tag}_numa_pref.data'
        if not os.path.exists(path):
            continue
        for block in re.split(r'^(?=\./)', open(path).read(), flags=re.M)[1:]:
            words = block.split('\n', 1)[0].split()
            if '-p' in words and (keep_per_kernel or ':' not in words[words.index('-p') + 1]):
                yield block, words


def load_results(app):
    """{(psize_gb, policy): seconds, or None if the run did not finish}."""
    results = {}
    for block, words in data_blocks(app):
        if not re.search(r'timed out|did not time out', block):
            continue  # interrupted (e.g. Ctrl-C): no outcome, so it reruns
        runtime = re.search(r'GPU Runtime: ([\d.]+)s', block)
        limit = re.search(r'within ([\d.]+) seconds', block)
        # GRAMSCHM stops itself at TIMEOUT and still prints a runtime: count that as not finished
        finished = runtime and not (limit and float(runtime.group(1)) >= 0.95 * float(limit.group(1)))
        results[(words[1], words[words.index('-p') + 1])] = float(runtime.group(1)) if finished else None
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apps', nargs='+', default=list(APPS))
    parser.add_argument('--psizes', nargs='+', default=['100', '125'])
    parser.add_argument('--policies', nargs='+', help="run only these policies (e.g. MIM labels), plus the all-m baseline")
    parser.add_argument('--plan', action='store_true',
                        help="print letters and runs left from results on disk, without running anything")
    args = parser.parse_args()

    os.environ['PERF_TAG'] = PERF_TAG
    limit = float(os.environ['TIMEOUT'])
    for app in args.apps:
        n = APPS[app]['allocs']
        baseline = 'm' * n
        exp = Experiment(app, 'run_numa.sh', f'{config.BENCHDIR}/strategied/{app}/', args.psizes, [], app,
                         config.KERNEL_VERSION, 'vanilla', {})
        for psize in args.psizes:
            psize_gb = str(int(int(psize) * config.VRAM_SIZE / 100))
            results = load_results(app)

            def run(policy):
                if (psize_gb, policy) not in results:
                    exp._run(False, psize, policy, min(TIMEOUT_FACTOR * baseline_s, limit))

            if args.plan:
                # probes not run yet keep all three letters
                letters = ['m' + ''.join(l for l in 'hd' if (psize_gb, baseline[:i] + l + baseline[i + 1:]) not in results
                                         or results[(psize_gb, baseline[:i] + l + baseline[i + 1:])] is not None)
                           for i in range(n)]
                probe_list = [baseline] + [baseline[:i] + l + baseline[i + 1:] for i in range(n) for l in 'hd']
                policies = search_space(app, letters, probe_list)
                print(f"{app} {psize}%: letters {letters}; {len(policies)} policies, "
                      f"{sum((psize_gb, p) not in results for p in policies)} not yet run")
                continue
            # the baseline sets the cut-off, so it always runs
            baseline_s = exp._run(False, psize, baseline, limit)
            if args.policies:
                # asked for by name: run even if already recorded (repeats are combined by their median)
                for policy in args.policies:
                    exp._run(False, psize, policy, min(TIMEOUT_FACTOR * baseline_s, limit))
                continue
            probes = {(i, letter): baseline[:i] + letter + baseline[i + 1:] for i in range(n) for letter in 'hd'}
            for policy in probes.values():
                run(policy)
            results = load_results(app)
            letters = ['m' + ''.join(l for l in 'hd' if results.get((psize_gb, probes[(i, l)])) is not None)
                       for i in range(n)]
            policies = search_space(app, letters, [baseline] + list(probes.values()))
            todo = [p for p in policies if (psize_gb, p) not in results and p != baseline]
            print(f"{app} {psize}%: letters per allocation {letters}; {len(policies)} policies, {len(todo)} to run")
            for policy in todo:
                run(policy)


if __name__ == "__main__":
    main()
