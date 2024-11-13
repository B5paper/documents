#!/bin/bash

/home/hlc/Documents/Projects/openmpi-5.0.5/bin/bin/oshrun -np 4 ./main
if [ $? -ne 0 ]; then
    exit
fi

echo ""

./main_cpu
