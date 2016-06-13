
close all
clear all
clc
format long;

addpath(genpath('/home/wang/matlab/quantum-dynamics/common'))
setenv('GAUSS_LEGENDRE_DIR', '/home/wang/matlab/quantum-dynamics/GaussLegendre')

J = 10;
M = 1; 
p = 1;

LMax = 100;

n = LMax + J + 60;
[ x, w ] = GaussLegendre2(n);

Omega = OmegaList(J, M, p, LMax);

n_Omega = Omega(end);

P = zeros(n, LMax+1, n_Omega+1);

for l = 1 : LMax+1
  lp = legendre(l-1, x, 'norm');
  n_dim = min(l-1, n_Omega);
  P(:, l, 1:n_dim+1) = lp(1:n_dim+1, :)';
end
