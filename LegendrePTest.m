
close all
clear all
clc
format long;

addpath(genpath('/home/wang/matlab/quantum-dynamics/common'))
setenv('GAUSS_LEGENDRE_DIR', '/home/wang/matlab/quantum-dynamics/GaussLegendre')

n = 500;
[ x, w ] = GaussLegendre2(n);

omega = 12;
LMax = 180;

L = LMax - omega + 1;
P = zeros(L, n);

for l = omega : LMax
  p = legendre(l, x, 'norm');
  P(l-omega+1,:) = p(omega+1, :);
end

s = P*diag(w)*P';

max(max(abs(s)))
