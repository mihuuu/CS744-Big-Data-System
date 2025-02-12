#!/bin/bash

# Remember to set machine rank using 
# $ export RANK=0   for node 0
# $ export RANK=1   for node 1
# $ export RANK=2   for node 2
# $ export RANK=3   for node 3
python main.py --master-ip 10.10.1.1 --num-nodes 4 --rank $RANK
