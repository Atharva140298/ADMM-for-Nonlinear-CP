close all; clear all; clc;
%addpath('./utils');

%% Model and Error Measure
model         = 'ReLU';      %Choose the model among: 'ReLU', 'CSF', 'MinMax', or 'Modulus' or 'Exponential
error_measure = 'Frobenius'; %Choose the error measure among: 'Frobenius','L1' (Componentwise L1 norm), or 'KL divergence'

%% Define parameters of the algorithm
param.maxiter = 3000;
param.maxtime =30;
param.a=0; param.b = 1000;  %For the MinMax model, the user has to define the desired interval [a,b] to generate data in that interval
%param.display = 0;        %If the user does not want any display of the plots
%param.factors='nonneg';
%% Data
m = 10;
n = 10;
p = 8;
r = 3;
[truefactors,X]= generate_synthetic(m,n,p,r,model,param);

%% Run of the algorithm
results = admm_ntd(X,r,model,error_measure,param);

[FacXn, ~, outXn] = cp_opt(tensor(X), r, 'init','rand', 'lower',0, 'printitn', 10);