#!/bin/bash
logbase=sgemm${PERF_TAG:+_$PERF_TAG}
exe=./matrixMul2
matsizes=$(python3 genSizes.py $1)
cmd="$exe $matsizes ${@:2}"
out="$exe $1 $matsizes ${@:2}"

../common/numa_exp.sh $logbase "$cmd" "$out"
