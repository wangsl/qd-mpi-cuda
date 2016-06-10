#!/bin/bash

module purge
module load mvapich2-gdr/2.2
#module load mvapich2/intel/cuda/2.2rc1

export MV2_USE_CUDA=1 
export MV2_USE_GPUDIRECT_GDRCOPY=1 

export LD_PRELOAD=$MVAPICH2_GDR_LIB/libmpi.so:$LD_PRELOAD

export LD_PRELOAD=$LD_PRELOAD:$MKL_LIB/libmkl_intel_ilp64.so:$MKL_LIB/libmkl_core.so:$MKL_LIB/libmkl_intel_thread.so:$INTEL_LIB/libiomp5.so

#export MV2_CUDA_BLOCK_SIZE=$(echo 262144*10 | bc)
#export MV2_GPUDIRECT_LIMIT=16384
#export MV2_CUDA_IPC_THRESHOLD=524288
#export MV2_USE_GPUDIRECT_GDRCOPY_NAIVE_LIMIT=16384
#export MV2_USE_CORE_DIRECT=2

mpiexec -np 8 sh matlab-mpi-wraper.sh -nodisplay -r "\"main1;\""

exit

command="mpiexec"

command="$command -np 1 $env mymatlab.sh -nodisplay -r \"main1;\""

np=3;
for((i=1; i<$np; i++)); do
    command="$command : -np 1 $env mymatlab.sh -nodesktop -r \"main1;\" "
done

echo $command

eval "$command"

exit

# mpiexec -np 8 env MV2_USE_CUDA=1 ./a.out

# http://cudamusing.blogspot.com/2011/11/mpi-communications-from-gpu-memory.html

# https://www.olcf.ornl.gov/tutorials/gpudirect-mpich-enabled-cuda/

