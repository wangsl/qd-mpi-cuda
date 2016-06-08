#!/bin/bash

module purge
module load mvapich2-gdr/2.2
module load matlab/2015b

RANK=
if [ "$PMI_RANK" != "" ]; then
    RANK=$PMI_RANK
elif [ "$OMPI_COMM_WORLD_RANK" != "" ]; then
    RANK=$OMPI_COMM_WORLD_RANK
fi

if [ "$RANK" == "" ]; then
    matlab "$*" #> /dev/null 2>&1
else
    if [ $RANK -eq 0 ]; then
	matlab "$*" 2>&1 | tee ${RANK}.log
    else
	matlab "$*" > ${RANK}.log 2>&1
    fi
fi
