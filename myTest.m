

function [] = myTest()

global myMPI

fprintf(' Rank = %d, size = %d\n', myMPI.rank, myMPI.size);


