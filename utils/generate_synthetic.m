function [truefactors, X] = generate_synthetic(m,n,p,r, model,param)
%Generating random low-rank factors
A0=randn(m,r);
B0=randn(n,r);
C0=randn(p,r);
%sparsity=0.7;
% Impose sparsity by zeroing out entries
% A0(rand(m,r) < sparsity) = 0;
% B0(rand(n,r) < sparsity) = 0;
% C0(rand(p,r) < sparsity) = 0;

U0 = {A0,B0,C0};
Xhat0 = cpdgen(U0);
%% Add noise
% N = randn(m,n,p);
% Xnoisy = Xhat0 + 0.2*N*(norm(Xhat0,'fro')/norm(N,'fro'));
% Xhat0=Xnoisy;
%% Generating synthetic data matrix X
switch model
    case 'ReLU'
        X = max(0,Xhat0);
    case 'CSF'
        X = (Xhat0).^2;
    case 'MinMax'
        X = min(param.b, max(param.a,Xhat0));
    case 'Modulus'
        X = abs(Xhat0);
    case 'Exponential'
        X = exp(Xhat0);
end

for i=1:3
    truefactors{i} = U0{i};
end
