
function [] = FH2main()

global myMPI

run_evolution = 1;
if(isempty(getenv('PMI_RANK')))
  run_evolution = 0;
  myMPI.rank = 0;
  myMPI.size = 1;
end

warning off MATLAB:maxNumCompThreads:Deprecated
maxNumCompThreads(20);

format long

%addpath(genpath('/home/wang/matlab/quantum-dynamics/build'))
addpath(genpath('/home/wang/matlab/quantum-dynamics/build/common-cuda'));
addpath(genpath('/home/wang/matlab/quantum-dynamics/common'));
addpath(genpath('/home/wang/matlab/quantum-dynamics/GaussLegendre'));
addpath(genpath('/home/wang/matlab/quantum-dynamics/FH2/HSW'));
addpath(genpath('/home/wang/matlab/quantum-dynamics/build/FH2'));

setenv('HSW_DATA_DIR', ...
       '/home/wang/matlab/quantum-dynamics/FH2/HSW')

global H2eV 
global FH2Data
%global myMPI

theta.n = int32(180);
[ theta.x, theta.w ] = GaussLegendre2(theta.n);

J = 15;
M = 1;
p = 1;
LMax = 120;

Omegas = OmegaList(J, p, LMax)

if (myMPI.size == 1) 
  n = numel(Omegas);
else
  n = 4;
end
index_start = n*myMPI.rank+1;
index_end = n*(myMPI.rank+1);
if index_end > numel(Omegas)
  index_end = numel(Omegas)
end

Omegas = Omegas(index_start:index_end)
nMax = LMax - min(Omegas) + 1;

P = zeros(numel(theta.x), nMax, numel(Omegas));

for i = 1 : numel(Omegas) 
  n = LMax - Omegas(i) + 1;
  P(:,1:n,i) = AssLegendreP(Omegas(i), LMax, theta.x);
end

OmegaStates.J = J;
OmegaStates.parity = 1;
OmegaStates.lmax = int32(LMax);
OmegaStates.omegas = int32(Omegas);
OmegaStates.associated_legendres = P;

jRot = 0;
nVib = 0;

H2eV = 27.21138505;

MassAU = 1.822888484929367e+03;

mH = 1.0079;
mF = 18.998403;

masses = [ mF, mH, mH ];

masses = masses*MassAU;

% time

time.total_steps = int32(10000);
time.time_step = 0;
time.steps = int32(0);

% r1: R

%r1.n = int32(1024);
r1.n = int32(512);
r1.r = linspace(0.2, 14.0, r1.n);
r1.left = r1.r(1);
r1.dr = r1.r(2) - r1.r(1);
r1.mass = masses(1)*(masses(2)+masses(3))/(masses(1)+masses(2)+ ...
					   masses(3));
r1.dump_Cd = 4.0;
r1.dump_xd = 12.0;

r1.r0 = 10.0;
r1.k0 = 2.0;8
r1.delta = 0.2;

eGT = 1/(2*r1.mass)*(r1.k0^2 + 1/(2*r1.delta^2))*H2eV;
fprintf(' Gaussian wavepacket energy: %.15f\n', eGT);

dump1.Cd = 4.0;
dump1.xd = 12.0;
%dump1.dump = WoodsSaxon(dump1.Cd, dump1.xd, r1.r);

% r2: r

%r2.n = int32(1024);
r2.n = int32(512);
r2.r = linspace(0.3, 12.0, r2.n);
r2.left = r2.r(1);
r2.dr = r2.r(2) - r2.r(1);
r2.mass = masses(2)*masses(3)/(masses(2)+masses(3));

r2.dump_Cd = 4.0;
r2.dump_xd = 10.0;

dump2.Cd = 4.0;
dump2.xd = 10.0;
%dump2.dump = WoodsSaxon(dump2.Cd, dump2.xd, r2.r);

% dividing surface

rd = 6.5;
nDivdSurf = int32((rd - min(r2.r))/r2.dr);
r2Div = double(nDivdSurf)*r2.dr + min(r2.r);
fprintf(' Dviding surface: %.8f\n', r2Div);

% theta

dimensions = 3;

if dimensions == 2 
  %theta.n = int32(1);
  theta.m = int32(0);
  %theta.x = 1.0;
  %theta.w = 2.0;
else 
  %theta.n = int32(180);
  theta.m = int32(LMax);
  %[ theta.x, theta.w ] = GaussLegendre2(theta.n);
end

theta.associated_legendre = LegendreP2(double(theta.m), theta.x);
% transpose Legendre polynomials in order to do 
% matrix multiplication in C++ and Fortran LegTransform.F
theta.associated_legendre = theta.associated_legendre';

theta.legendre = LegendreP2(double(theta.m), theta.x);
% transpose Legendre polynomials in order to do 
% matrix multiplication in C++ and Fortran LegTransform.F
theta.legendre = theta.legendre';

% options

options.wave_to_matlab = 'FH2Matlab.m';
options.CRPMatFile = sprintf('CRPMat-j%d-v%d.mat', jRot, nVib);
options.steps_to_copy_psi_from_device_to_host = int32(100);
options.plot = false;

% setup potential energy surface and initial wavepacket
pot = FH2PESJacobi(r1.r, r2.r, acos(theta.x), masses);

[ psi, eH2, psiH2 ] = InitWavePacket(r1, r2, theta, jRot, nVib);

nOmegas = numel(OmegaStates.omegas);
OmegaStates.wave_packets = zeros([size(psi), nOmegas]);
for i = 1 : nOmegas
  OmegaStates.wave_packets(:,:,:,i) = psi;
end

size(OmegaStates.wave_packets)

if options.plot 
  PlotPotWave(r1, r2, pot, psi);
end

% cummulative reaction probabilities

CRP.eDiatomic = eH2;
CRP.n_dividing_surface = nDivdSurf;
CRP.n_gradient_points = int32(21);
CRP.n_energies = int32(400);
eLeft = 0.001/H2eV + eH2;
eRight = 0.25/H2eV + eH2;
CRP.energies = linspace(eLeft, eRight, CRP.n_energies);
CRP.eta_sq = EtaSq(r1, CRP.energies-eH2);
CRP.CRP = zeros(size(CRP.energies));
CRP.calculate_CRP = int32(1);

% pack data to one structure

FH2Data.r1 = r1;
FH2Data.r2 = r2;
FH2Data.theta = theta;
FH2Data.pot = pot;
FH2Data.psi = psi;
FH2Data.time = time;
FH2Data.options = options;
FH2Data.dump1 = dump1;
FH2Data.dump2 = dump2;
FH2Data.CRP = CRP;

FH2Data.OmegaStates = OmegaStates

% time evoluation

if run_evolution == 0
  TimeEvolutionMexCUDA(FH2Data);
  return
end
  
tic
cudaMPIEvolution(FH2Data);
toc

FH2Data.time.steps

return

