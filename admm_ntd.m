function results = admm_ntd(X,r,model,error_measure, param)
%addpath('./utils');
% Computes an approximate solution of the following non linear tensor
% decomposition problem (NTD)
%
%       min_{A,B,C,T} d(X , f(T))  s.t. T = [A,B,C]
%
% using Alternating Direction method of Multipliers. With this flexible
% framework, the user can choose the nonlinear model and the loss function
% as per their requirements.

%****** Input ******
%   X          : m-by-n-by-p matrix.
%   r          : scalar, desired approximation rank
%   model      : Select nonlinearity among: ReLU, CSF, MinMax, Modulus, Exponential
% error_measure: Select the loss function among: Frobenius, L1, KL divergence
%   param   : structure, containing the parameters of the algorithm .
%       .T0     = initialization of the variable T (default: X).
%       .A0     = initialization of the variable A (default: randn).
%       .B0     = initialization of the variable B (default: randn).
%       .C0     = initialization of the variable C (default: randn).
%       .lambda = initialization of the Lagrange Multiplier (default: zeroes).
%       .maxiter= maximum number of iterations (default: 1000).
%       .tol    = tolerance on the relative error (default: 1.e-9).
%       .tolerr = tolerance on 10 successive errors (err(i+1)-err(i-10))(default: 1.e-10)
%       .maxtime= time limit (default: 10 seconds).
%       For using MinMax model, specify the interval [a,b] (default: [0,1].
%       .a      = initial point of the interval.
%       .b      = end point of the interval.
%       .display= displays plots of relative error and residual
%                  (default = 1 means plots are drawn, 0 = no plots)
%       .factors= you can set the factors to be nonnegative, A,B,C >=0
%                  (default=none means no sign restriction,
%                   param.factors='nonneg' means Nonnegative NTD)
%       .mask   = mask for missing entries, if any.
%                  param.mask has the same size as X, and
%                  param.mask(i,j)=0 if (i,j) is missing,
%                  param.mask(i,j)=1 if (i,j) is observed.
%       .validmask  = must be a subset of .mask,
%                     and X(param.validmask) are actually known.
%                     This allows to validate the factorization. The user
%                     can put entries in the mask that are actually known
%                     to validate that the model can recover them. This
%                     is computed via the RMSE.
%                     By default, param.validmask=param.mask
%       .p_obs  = percentage of observed entries in matrix completion.
%       .B      = corrupted matrix with noise and missing entries.

% ****** Output ******
%   T       : m-by-n-by-p matrix, approximate solution of
%            min_{A,B,C,T} d(X , f(T))  s.t. T = [A,B,C]
%   A       : low-rank factor matrix A of dimension m x r.
%   B       : low-rank factor matrix B of dimension n x r.
%   C       : low-rank factor matrix C of dimension p x r.
%           such that X ~ f([A,B,C])
% objective : vector containing evolution of relative error along
%             iterations d(X, f([A,B,C])) / norm(X).
% residual : vector containing evolution of residual error along
%             iterations || T - [A,B,C]||_F / ||T||_F.
% If mask is provided, the task is matrix completion, we compute and display:
%rmse_test : RMSE on test data
%rmse_train: RMSE on train data


%% How to add other models with a new nonlinearity and loss function
% In this framework, we have added 14 different combinations of nonlinear
% functions and loss-functions. However, to add new nonlinearity or loss
% function, solve the one-dimensional subproblems for each entry of T and
% add it to update_T.m file.

[m,n,p]=size(X);
if nargin < 5
    param = [];
end
if ~isfield(param,'T0')
    if strcmpi(model,'CSF')
        param.T0 = sqrt(X);
    else
        param.T0 = X;
    end
end
if ~isfield(param,'A0') || ~isfield(param,'B0') || ~isfield(param,'C0')
    param.A0 = randn(m,r);
    param.B0 = randn(n,r);
    param.C0 = randn(p,r);
end
if ~isfield(param,'maxiter')
    param.maxiter = 100;
end
if ~isfield(param,'tol')
    param.tol = 1.e-9;
end
if ~isfield(param,'tolerr')
    param.tolerr = 1.e-10;
end
if ~isfield(param,'maxtime')
    param.maxtime = 10;
end
if ~isfield(param,'lambda')
    param.lambda = zeros(m,n,p);
end
if ~isfield(param,'a')
    param.a = eps;
end
if ~isfield(param,'b')
    param.b = 1;
end
if ~isfield(param,'Xtrue')
    param.Xtrue = X;
end
if ~isfield(param,'display')
    param.display = 1;
end
if ~isfield(param,'factors')
    param.factors=[];
end

% Synthetic vs real data flag (used by mc_metrics)
if ~isfield(param,'synthetic')
    param.synthetic = false; % set true from the caller for synthetic experiments
end

cpu0=tic;
timedisplay = 0.1; cntdisp = 0;
compute_obj=0;
%Define the initial variables
A=param.A0;
B=param.B0;
C=param.C0;
T = param.T0;
Xtrue = param.Xtrue;
lambda = param.lambda;
maxiter= param.maxiter;
maxtime = param.maxtime;
eps_reg=1e-6;

if isfield(param, 'mask') && ~isempty(param.mask)
    if param.display==1
        disp('This is a matrix completion task');
        disp('RMSE computation for missing/validation entries will be performed.');
    end
    Mask = logical(param.mask);

    % Build observed matrix O from X and M (zeros where missing)
    O = zeros(m,n,p);
    O(Mask) = X(Mask);

    % Decide which reference to give mc_metrics:
    % - Synthetic: compare to full ground truth X
    % - Real data: no ground truth -> pass [] and compare to B on observed entries
    if isfield(param,'synthetic') && param.synthetic
        X_true_metrics = X;
    else
        X_true_metrics = []; % real/incomplete data
    end
    O_observed = O;
else
    if param.display==1
        disp('Full data factorization mode detected.');
        disp('Display of iteration number and relative error in %:');
    end
end

k = 1;
start_obj=tic;
compute_obj=compute_obj+toc(start_obj);
runtime=toc(cpu0)-compute_obj;
savetime(k)=runtime;
rho(k)=1; % Setting initial value for penalty parameter rho


U = {A,B,C};
Xhat = cpdgen(U);
% Computing initial RMSE/Relative error
if isfield(param, 'mask') && ~isempty(param.mask)
    metrics = mc_metrics(X_true_metrics, O_observed, Xhat, model, param);
    rmse_train(k)   = metrics.rmse_train;
    rmse_test(k)    = metrics.rmse_validation;
    if isfield(metrics,'rmse_missing') && ~isempty(metrics.rmse_missing)
        rmse_missing(k) = metrics.rmse_missing;
    else
        rmse_missing(k) = NaN;
    end
else
    [objective(k),residual(k)] = compute_objective_residual(Xtrue,Xhat,T,model,error_measure,param);
end

while k < maxiter &&  runtime < maxtime
    if strcmpi(param.factors,'nonneg')
        n_inner = 20;                           % HALS inner sweeps

        %% ===== Update A (HALS) =====
        KR_A     = kr(C,B);
        T_A      = tens2mat(T,1);
        lambda_A = tens2mat(lambda,1);
        M        = (T_A + (1/rho(k))*lambda_A) * KR_A;
        G_A      = (C'*C) .* (B'*B) + eps_reg*eye(r);
        A        = hals_nnls(M, G_A, A, n_inner);

        %% ===== Update B (HALS) =====
        KR_B     = kr(C,A);
        T_B      = tens2mat(T,2);
        lambda_B = tens2mat(lambda,2);
        M        = (T_B + (1/rho(k))*lambda_B) * KR_B;
        G_B      = (C'*C) .* (A'*A) + eps_reg*eye(r);
        B        = hals_nnls(M, G_B, B, n_inner);

        %% ===== Update C (HALS) =====
        KR_C     = kr(B,A);
        T_C      = tens2mat(T,3);
        lambda_C = tens2mat(lambda,3);
        M        = (T_C + (1/rho(k))*lambda_C) * KR_C;
        G_C      = (B'*B) .* (A'*A) + eps_reg*eye(r);
        C        = hals_nnls(M, G_C, C, n_inner);
    else
        %% ===== Update A =====
        KR_A = kr(C,B);   % Khatri-Rao product

        T_A = tens2mat(T,1);
        lambda_A = tens2mat(lambda,1);
        M = (T_A + (1/rho(k))*lambda_A) * KR_A;

        G_A = (C'*C) .* (B'*B);
        G_A = G_A + eps_reg*eye(r);

        A = M / G_A;

        %% ===== Update B =====
        KR_B = kr(C,A);
        T_B = tens2mat(T,2);
        lambda_B = tens2mat(lambda,2);
        M = (T_B + (1/rho(k))*lambda_B) * KR_B;

        G_B = (C'*C) .* (A'*A);
        G_B = G_B + eps_reg*eye(r);

        B = M / G_B;

        %% ===== Update C =====
        KR_C = kr(B,A);
        T_C = tens2mat(T,3);
        lambda_C = tens2mat(lambda,3);
        M = (T_C + (1/rho(k))*lambda_C) * KR_C;

        G_C = (B'*B) .* (A'*A);
        G_C = G_C + eps_reg*eye(r);

        C = M / G_C;
    end

    %% ===== Reconstruct Tensor =====
    U = {A,B,C};
    Xhat = cpdgen(U);
    T_prev=T;

    %% ===== Update T (user function) =====
    % Updating T, depending on the task of full factorization or matrix completion
    if isfield(param, 'mask') && ~isempty(param.mask)
        [T] = update_T(O,rho(k),T,Xhat,lambda, model, error_measure, param);
        T(Mask==0) = Xhat(Mask==0) - (lambda(Mask==0)./rho(k));
    else
        [T] = update_T(X,rho(k),T,Xhat,lambda./rho(k),model,error_measure,param);
    end

    %% ===== Dual Update =====
    lambda = lambda + rho(k)*(T - Xhat);
    %% Adaptive update for the penalty paramater rho
    mu = 10;
    t_incr = 2;
    t_decr = 2;
    p_residual = norm(T - Xhat, 'fro');
    S = rho(k)*(tens2mat(T,3) - tens2mat(T_prev,3))*KR_C;
    d_residual = norm(S, 'fro');
    if p_residual > mu*d_residual
        rho(k+1) = t_incr*rho(k);
    elseif d_residual > mu*p_residual
        rho(k+1) = rho(k)/t_decr;
    else
        rho(k+1) = rho(k);
    end

    %% ===== Bookkeeping =====
    k = k + 1;
    start_obj=tic;
    compute_obj=compute_obj+toc(start_obj);
    runtime=toc(cpu0)-compute_obj;
    savetime(k)=runtime;


    %% Computing objective and residual or RMSE for matrix completion
    if isfield(param, 'mask') && ~isempty(param.mask)
        metrics = mc_metrics(X_true_metrics, O_observed,Xhat, model, param);
        rmse_train(k)   = metrics.rmse_train;
        rmse_test(k)    = metrics.rmse_validation;
        if isfield(metrics,'rmse_missing') && ~isempty(metrics.rmse_missing)
            rmse_missing(k) = metrics.rmse_missing;
        else
            rmse_missing(k) = NaN;
        end
    else
        [objective(k),residual(k)] = compute_objective_residual(Xtrue, Xhat , T, model, error_measure,param);
    end

    %% ===== Stopping criteria =====

    % 1) ADMM objecive error
    if ~isfield(param, 'mask')
        rel_residual = objective(k);


        if rel_residual < param.tol
            status_msg = 'Stopped: ADMM objective below tolerance';
            break;
        end
    end
    % 2) Successive error change stopping
    window = 10;

    if k > window
        if isfield(param, 'mask') && ~isempty(param.mask)
            err_now  = rmse_test(k);
            err_prev = rmse_test(k-window);
        else
            err_now  = objective(k);
            err_prev = objective(k-window);
        end

        if abs(err_now - err_prev) < param.tolerr
            status_msg = 'Stopped: successive errors changed less than tolerr';
            break;
        end
    end
    %% Displaying iteration number and errors on the command screen
    if param.display == 1
        if toc(cpu0) >= timedisplay
            if isfield(param, 'mask') && ~isempty(param.mask)
                fprintf('Iteration:%2.0f Error:%2.3f \n',k,rmse_test(k));
            else
                fprintf('Iteration:%2.0f Error:%2.3f \n',k,100*objective(k));
            end
            timedisplay = min(2*timedisplay,timedisplay+5);
            cntdisp = cntdisp+1;
            % if mod(cntdisp,5) == 0
            %     fprintf('\n');
            % end
        end
    end

end

%% Build results struct
% Making sure optional variables exist
if ~exist('rmse_train','var'), rmse_train = []; end
if ~exist('rmse_test','var'),  rmse_test  = []; end
if ~exist('rmse_missing','var'),  rmse_missing  = []; end
if ~exist('residual','var'), residual = []; end
if ~exist('objective','var'),  objective  = []; end
if ~exist('rho','var'),        rho        = []; end
if ~exist('iter','var'),       iter       = k; end
%if ~exist('status_msg','var'), status_msg = 'unknown'; end
if ~exist('status_msg','var')
    status_msg = 'Stopped: maxiter or maxtime reached';
end
results = struct();
results.T           = T;
results.A           = A;
results.B           = B;
results.C           = C;
results.lambda      = lambda;
results.residual    = residual;
results.objective   = objective;
results.rmse_train  = rmse_train;
results.rmse_test   = rmse_test;
results.rmse_missing= rmse_missing;
results.time        = savetime;
results.rho         = rho;
results.status_msg = status_msg;
results.iter       = k;
% Keep the parameters actually used
results.param = param;


%% ===== Plot =====
if param.display==1
    if isfield(param, 'mask') && ~isempty(param.mask)
        figure('Name','RMSE Plots','NumberTitle','off');
        % --- RMSE on Test Data ---
        subplot(1,2,1);
        if param.synthetic == true
            plot(rmse_missing, 'b-', 'LineWidth', 2); grid on;
        else
            plot(rmse_test, 'b-', 'LineWidth', 2); grid on;
        end
        xlabel('Iterations','Interpreter','latex','FontSize',16);
        ylabel('RMSE on Missing/Validation Data','Interpreter','latex','FontSize',16);
        title('Test RMSE','Interpreter','latex','FontSize',18);
        set(gca,'FontSize',14,'LineWidth',1);
        set(gca,'TickLabelInterpreter','latex');
        % --- RMSE on Train Data ---
        subplot(1,2,2);
        plot(rmse_train, 'r-', 'LineWidth', 2); grid on;
        xlabel('Iterations','Interpreter','latex','FontSize',16);
        ylabel('RMSE on Train Data','Interpreter','latex','FontSize',16);
        title('Train RMSE','Interpreter','latex','FontSize',18);
        set(gca,'FontSize',14,'LineWidth',1);
        set(gca,'TickLabelInterpreter','latex');
        % --- Overall figure title ---
        sgtitle('RMSE Evolution During ADMM Iterations','Interpreter','latex','FontSize',18);
    else
        figure('Name','ADMM evolution of errors','NumberTitle','off');
        subplot(1,2,1);
        semilogy(objective,'b-','LineWidth',2); grid on;
        xlabel('Iterations','Interpreter','latex','FontSize',16);
        if strcmpi(error_measure, 'KL divergence')
            ylabel('Normalized KL:$KL(X,\hat{X})/KL(X,\bar{X})$', 'Interpreter','latex', 'FontSize',16,'FontWeight','bold');
        else
            ylabel('Relative error: $d(X,f( [[A,B,C]])/d(X,0)$','Interpreter','latex','FontSize',16, 'FontWeight','bold');
        end
        title('ADMM Objective','Interpreter','latex','FontSize',18);
        set(gca,'FontSize',14,'LineWidth',1);
        set(gca,'TickLabelInterpreter','latex');
        % --- Residual plot ---
        subplot(1,2,2);
        semilogy(residual,'r-','LineWidth',2); grid on;
        xlabel('Iterations','Interpreter','latex','FontSize',16);
        ylabel('Residual: $\|T- [[A,B,C]] \|_F/\|T\|_F$','Interpreter','latex','FontSize',16);
        title('ADMM Residual','Interpreter','latex','FontSize',18);
        set(gca,'FontSize',14,'LineWidth',1);
        set(gca,'TickLabelInterpreter','latex');
        sgtitle('ADMM evolution of error','Interpreter','latex','FontSize',18);
    end


end