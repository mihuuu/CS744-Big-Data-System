#!/bin/sh

# run with different partitions
for i in 10 50 100 200 300; do
    docker exec master python3 /src/task2.py /task1/ "task2-${i}" "${i}" "task2-${i}"
done