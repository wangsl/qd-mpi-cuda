
% Refercens:
% J. Phys. Chem. A, 102, 9372-9379 (1998)
% John Zhang  Theory and Application of Quantum Molecular Dynamics P343

function CoriolisTerm3

format long

for l = 0 : 10
  A = CoriolisTerm(l)
  %A*A'
end

function [ A ] = CoriolisTerm(l)

format long

J = 1;
p = 1;
M = 1;
OmegaMax = min(J, l);

if J == 0
  OmegaMin = 0;
  p = 0;
elseif rem(J+p, 2) == 0
  OmegaMin = 0;
else
  OmegaMin = 1;
end

n = OmegaMax - OmegaMin + 1;

H = zeros(n, n);
for Omega = OmegaMin : OmegaMax
  for Omega1 = OmegaMin : OmegaMax
    H(Omega-OmegaMin+1, Omega1-OmegaMin+1) = T(J, Omega, M, l, p, Omega1);
  end
end

[ vecs, e ] = eig(H);

A = vecs*diag(exp(-j*diag(e)))*vecs';

%%%

function [ d ] = KroneckerDelta(a, b)
if a == b
  d = 1;
else
  d = 0;
end

%%% 

function [ t ] = T(J, Omega, M, l, p, Omega1)

t = (J*(J+1) + l*(l+1) - 2*Omega^2)*KroneckerDelta(Omega1,Omega)  ...
    -KroneckerDelta(Omega1,Omega+1)*sqrt(1+KroneckerDelta(Omega, 0))*...
    C(J, Omega, 0)*C(l, Omega, 0) - ...
    KroneckerDelta(Omega1,Omega-1)*sqrt(1+KroneckerDelta(Omega, 1))*...
    C(J, Omega, 1)*C(l, Omega, 1);
    
%%%

function [ c ]  = C(J, Omega, p)
if J < Omega
  c = 0;
  return
end

if rem(p, 2) == 0
  c = sqrt(J*(J+1) - Omega*(Omega+1));
else
  c = sqrt(J*(J+1) - Omega*(Omega-1));
end

%%% 
