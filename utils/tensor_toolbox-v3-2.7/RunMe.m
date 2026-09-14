%% ============================================================
%  run_gcopt.m
%  Minimal example for generalized CP decomposition using GCPOPT
%  Tammy Kolda et al., SIAM Review
%% ============================================================

clear; clc; close all;

%% -------------------------------
% Parameters
%% -------------------------------
m = 145;
n = 145;
p = 200;
R = 10;

rng(0);   % reproducibility

%% -------------------------------
% Generate synthetic CP factors
%% -------------------------------
A = rand(m, R);
B = rand(n, R);
C = rand(p, R);

% Normalize columns (recommended)
A = A ./ vecnorm(A);
B = B ./ vecnorm(B);
C = C ./ vecnorm(C);

%% -------------------------------
% Construct tensor X
%% -------------------------------
X = ktensor({A, B, C});
X = tensor(X);   % convert to dense tensor
S = load("Indian_pines_corrected.mat");
A = S.indian_pines_corrected;
X = tensor(A);
sz = size(X);
fprintf('Tensor size: %d x %d x %d\n', sz(1), sz(2), sz(3));
%% ============================================================
% Choose loss type
%% ============================================================
% 'gaussian'  -> Frobenius norm
% 'poisson'   -> KL divergence
% 'bernoulli' -> binary data
% 'gamma'

loss_type = 'gaussian';

%% -------------------------------
% Initialization
%% -------------------------------
A0 = randn(m, R);
B0 = randn(n, R);
C0 = randn(p, R);

M0 = ktensor({A0, B0, C0});

%% -------------------------------
% Run GCPOPT
%% -------------------------------
fprintf('Running GCP-OPT (%s loss)...\n', loss_type);
M = gcp_opt(X, R, ...
            'type', loss_type, ...
            'init', M0, ...
            'maxiters', 200, ...
            'printitn', 10);

%% -------------------------------
% Reconstruction
%% -------------------------------
Xhat = tensor(M);

relerr = norm(X(:) - Xhat(:)) / norm(X(:));

fprintf('Relative reconstruction error = %.3e\n', relerr);

%% -------------------------------
% Optional: plot convergence
%% -------------------------------
% figure;
% semilogy(info.relerr_trace, 'LineWidth', 2);
% xlabel('Iteration');
% ylabel('Objective value');
% title(['GCPOPT convergence (', loss_type, ')']);
% grid on;
