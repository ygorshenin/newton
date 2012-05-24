#!/bin/bash

set -e

MAX_NUM_THREADS=12

echo Binary: $1
echo Test: $2

if [ $# -ne 2 ]; then
  echo Usage: $0 path-to-binary path-to-test
  exit 1
fi

for (( num_threads=1 ; num_threads <= $MAX_NUM_THREADS ; ++num_threads ))
do
  echo
  echo Running on $num_threads threads...
  time env OMP_NUM_THREADS=$num_threads $1 < $2 > /dev/null 2>&1
done
