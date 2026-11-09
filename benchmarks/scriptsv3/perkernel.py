#!/usr/bin/python3
# Times per-kernel placement policies ("-p dmm:mdd", one string per kernel) next to whole-run policies
# usage: TIMEOUT=12000 python3 -u perkernel.py [--apps ...] [--psizes ...] [--policies P ...] [--plan]
import argparse
import itertools
import os
import re

import config
from experiments_nolog import Experiment

# allocations touched by each kernel, in launch order
APPS = {
    '2MM':      {'allocs': 5, 'kernels': [{0, 1, 2}, {2, 3, 4}], 'switch_d': True},
    'GRAMSCHM': {'allocs': 3, 'kernels': [{0, 1}, {0, 1, 2}, {0, 1, 2}], 'switch_d': False},
    'sgemm':    {'allocs': 3, 'kernels': [{0, 1, 2}], 'switch_d': True, 'control': True},
    # setup and timestep phases, --policies only; 16-19 are reduction buffers read on the host, can't be 'd'
    'tealeaf':  {'allocs': 20, 'policies_only': True, 'host_read': {16, 17, 18, 19}},
}
TIMEOUT_FACTOR = 1.25
PERF_TAG = 'perkernel2'


def whole_run_policies(app, letters):
    return [''.join(p) for p in itertools.product(*letters)]


def kernel_choices(app, kernel, letters):
    # untouched allocations stay 'm'
    spec = APPS[app]
    slots = [letters[i] if i in spec['kernels'][kernel] else 'm' for i in range(spec['allocs'])]
    return [''.join(p) for p in itertools.product(*slots)]


def per_kernel_policies(app, letters):
    spec = APPS[app]
    if spec['switch_d']:
        per_kernel = itertools.product(*(kernel_choices(app, k, letters) for k in range(len(spec['kernels']))))
        # trailing ':' forces the per-kernel path
        return [':'.join(p) + (':' if len(p) == 1 else '') for p in per_kernel]
    # 'd' is fixed across kernels, only m/h change
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
    # skip policies with the same string for every kernel
    return whole + [p for p in per_kernel_policies(app, letters)
                    if len(set(p.rstrip(':').split(':'))) > 1 or p.rstrip(':').split(':')[0] not in whole]


def data_blocks(app):
    # per-kernel runs in the old file predate the 'd' fix, so only its whole-run runs are used
    for tag, keep_per_kernel in [('perkernel', False), (PERF_TAG, True)]:
        path = f'{config.BENCHDIR}/strategied/{app}/{app.lower()}_{tag}_numa_pref.data'
        if not os.path.exists(path):
            continue
        for block in re.split(r'^(?=\./)', open(path).read(), flags=re.M)[1:]:
            words = block.split('\n', 1)[0].split()
            if '-p' in words and (keep_per_kernel or ':' not in words[words.index('-p') + 1]):
                yield block, words


def load_results(app):
    results = {}
    for block, words in data_blocks(app):
        if not re.search(r'timed out|did not time out', block):
            continue  # interrupted run
        runtime = re.search(r'GPU Runtime: ([\d.]+)s', block)
        limit = re.search(r'within ([\d.]+) seconds', block)
        # GRAMSCHM prints a runtime even when it hits its own timeout
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
        if APPS[app].get('policies_only') and not args.policies:
            print(f"{app}: no search space; pass the policies to time with --policies")
            continue
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
                # letters without a probe result stay in
                letters = ['m' + ''.join(l for l in 'hd' if (psize_gb, baseline[:i] + l + baseline[i + 1:]) not in results
                                         or results[(psize_gb, baseline[:i] + l + baseline[i + 1:])] is not None)
                           for i in range(n)]
                probe_list = [baseline] + [baseline[:i] + l + baseline[i + 1:] for i in range(n) for l in 'hd']
                policies = search_space(app, letters, probe_list)
                print(f"{app} {psize}%: letters {letters}; {len(policies)} policies, "
                      f"{sum((psize_gb, p) not in results for p in policies)} not yet run")
                continue
            # baseline sets the timeout for the rest
            baseline_s = exp._run(False, psize, baseline, limit)
            if args.policies:
                # always rerun named policies
                for policy in args.policies:
                    pinned_host_read = [i for phase in policy.rstrip(':').split(':') for i in APPS[app].get('host_read', ())
                                        if i < len(phase) and phase[i] == 'd']
                    if pinned_host_read:
                        print(f"skipping {policy}: allocations {sorted(set(pinned_host_read))} are read by the CPU and can't be 'd'")
                        continue
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
