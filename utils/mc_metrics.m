function metrics = mc_metrics(X_true, O_observed, A, model, param)
% mc_metrics  Compute RMSE metrics for matrix completion (synthetic vs real).
%
% Inputs
%   X_true      : (optional) full ground-truth matrix. If [], treated as real data.
%   O_observed  : observed data matrix (use B for real data; for synthetic you can pass X_true or noisy B)
%   Xhat           : pre-activation estimate (e.g., [A,B,C])
%   model       : 'ReLU' | 'CSF' | 'MinMax' | 'Modulus' | Exponential
%   param       : struct with fields:
%                   .mask        -> 1 for observed, 0 for missing (required)
%                   .validmask   -> subset of mask for validation (optional)
%                   .a,.b        -> for 'MinMax' (optional)
%
% Output
%   metrics: struct with fields (some may be [])
%       .rmse_train
%       .rmse_validation
%       .rmse_missing        (only if X_true provided)
%       .is_synthetic        (true if X_true provided)
%
% Example:
%   % Synthetic
%   metrics = mc_metrics(X_true, X_noisy, [A,B,C], 'ReLU', param);
%   % Real data (MovieLens)
%   metrics = mc_metrics([], B, [A,B,C], 'ReLU', param);

if nargin < 1 || isempty(X_true)
    X_true = [];
end
if nargin < 4
    error('Usage: mc_metrics(X_true, B_observed, A, model, param)');
end
valid_models = {'ReLU','CSF','MinMax','Modulus','WLRA','Exponential'};
if ~ismember(model, valid_models)
    error('Unknown model "%s". Must be one of: %s', model, strjoin(valid_models,', '));
end
if ~isstruct(param)
    error('param must be a struct.');
end

    % --- Masks (robust defaults) ---
    if ~isfield(param,'mask') || isempty(param.mask)
        error('param.mask is required: 1=observed, 0=missing.');
    end
    M_mask  = logical(param.mask);

    if ~isfield(param,'validmask') || isempty(param.validmask)
    M_valid = M_mask;                         % default
    else
    M_valid = logical(param.validmask);
    % Allow disjoint validation masks (common in real datasets).
    % For synthetic runs (X_true provided), you may enforce subset if desired:
    % if ~is_synth, keep as-is; if is_synth and you want subset: M_valid = M_valid & M_mask;
    end

    M_train   = M_mask & ~M_valid;  % observed-but-not-in-validation
    M_missing = ~M_mask;            % truly missing

    % Special case: if user didn’t separate validation, make train = mask
    if ~any(M_train(:))
        M_train = M_mask;
    end

    % --- Apply model nonlinearity to A -> Xhat ---
    switch model
        case 'ReLU'
            Xhat = max(0, A);
        case 'CSF'
            Xhat = A.^2;
        case 'MinMax'
            if ~isfield(param,'a') || ~isfield(param,'b')
                error('MinMax model requires param.a and param.b.');
            end
            Xhat = min(param.b, max(param.a, A));
        case 'Modulus'
            Xhat = abs(A);
        case 'WLRA'
            Xhat = A;
        case 'Exponential'
            Xhat = exp(A);
    end

    % --- RMSE helper (MATLAB-legal) ---
    rmse_on = @(Xref, M) sqrt( mean( (Xref(M) - Xhat(M)).^2 ));

    % --- Decide synthetic vs real ---
    is_synth = ~isempty(X_true);

    % --- Compute metrics ---
    metrics = struct;
    metrics.is_synthetic = is_synth;

    % Train RMSE: always computed
    %   - For synthetic: compare to X_true on train entries
    %   - For real: compare to B_observed on train entries
    if is_synth
        metrics.rmse_train = rmse_on(X_true, M_train);
    else
        metrics.rmse_train = rmse_on(O_observed, M_train);
    end

    % Validation RMSE: only meaningful if M_valid provided separately; still computed
    if is_synth
        metrics.rmse_validation = rmse_on(X_true, M_valid);
    else
        metrics.rmse_validation = rmse_on(O_observed, M_valid);
    end

    % Missing RMSE: only when you have ground truth (synthetic)
    if is_synth
        if any(M_missing(:))
            metrics.rmse_missing = rmse_on(X_true, M_missing);
        else
            metrics.rmse_missing = [];
        end
    else
        metrics.rmse_missing = []; % cannot compute for real data
    end
end

