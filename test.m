
close all
clear all
clc

global FH2Data

FH2main

psi = FH2Data.psi;
pot = FH2Data.pot;

w = FH2Data.theta.w;

n1 = FH2Data.r1.n;
n2 = FH2Data.r2.n;
nTheta = numel(w);

psi = psi(1:2:end, :, :) + j*psi(2:2:end, :, :);

dr1 = FH2Data.r1.dr;
dr2 = FH2Data.r2.dr;

s = sum(sum(abs(psi).^2, 1), 2);
sum(reshape(s, [numel(s), 1]).*w)*dr1*dr2

s = sum(sum(abs(psi).^2.*pot, 1), 2);
sum(reshape(s, [numel(s), 1]).*w)*dr1*dr2
