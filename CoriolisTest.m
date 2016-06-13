
close all
clear all
clc
format long;

J = 10;
M = 2;
p = 1;

LMax = 100;

OmegaMax = min(J, LMax);

if J == 0
  OmegaMin = 0;
  OmegaMax = 0;
  p = 0;
elseif rem(J+p, 2) == 0
  OmegaMin = 0;
else
  OmegaMin = 1;
end



