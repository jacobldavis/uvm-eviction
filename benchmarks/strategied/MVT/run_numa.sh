#!/bin/bash
logbase=mvt${PERF_TAG:+_$PERF_TAG}
exe=./mvt.exe
cmd="$exe $1 0 ${@:2}"
out="$exe $1 0 ${@:2}"

../common/numa_exp.sh $logbase "$cmd" "$out"
