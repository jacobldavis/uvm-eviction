#!/bin/bash -xe

logbase=tealeaf${PERF_TAG:+_$PERF_TAG}
exe=./tealeaf
cmd="$exe ${@:2}"
out="$exe $1 ${@:2}"
python3 gentea.py $1
../common/numa_exp.sh $logbase "$cmd" "$out"
