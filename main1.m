
format long

global myMPI
global FH2Data

[ myMPI.rank, myMPI.size ] = MPI_Init();

setenv('CUDA_VISIBLE_DEVICES', int2str(myMPI.rank));
%setenv('OMP_NUM_THREADS', int2str(20));
%setenv('MKL_NUM_THREADS', int2str(20));

FH2main();

for i = 1 : 100
  %FH2main()
  tic
  cudaMPIEvolution(FH2Data);
  toc
end

% PiMPI()
% cudaDirect()
% cudaMPITest();

MPI_Finalize();

exit


J = 6;
p = 0;
M = 2;
lmax = 100;

for i = 1 : J
  for l = 1 : lmax
    dstevTest(int32([J, M, p, l]))
  end
end

cudaMPITest();

MPI_Finalize();

exit

system('ldd -r /home/wang/matlab/MPIMex/dstevTest.mexa64')


J = 6;
p = 0;
M = 2;
lmax = 100;

for i = 1 : J
  for l = 1 : lmax
    dstevTest(int32([J, M, p, l]))
  end
end

matDirect()

PiMPI()

main(myMPI.rank, 0)

matMPIFinalize()

exit
