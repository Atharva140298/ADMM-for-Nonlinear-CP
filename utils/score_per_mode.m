function fms_modes = score_per_mode(A, B, mode_names)
% SCORE_PER_MODE  Computes FMS along each mode separately.
%
%  Uses the best permutation found by score() to align components,
%  then computes per-mode cosine similarity for each matched pair,
%  averaged across components.
%
%  A, B        : ktensor models (same size, same R)
%  mode_names  : cell array of mode labels, e.g. {'Subjects','Time','Metabolites'}
%
%  fms_modes   : 1 x N vector of per-mode FMS values
%
%  Example:
%    fms_modes = score_per_mode(model_CPWOPT, model_CPNN, ...
%                   {'Subjects','Time','Metabolites'});

    if nargin < 3
        mode_names = arrayfun(@(n) sprintf('Mode %d', n), ...
            1:ndims(A), 'UniformOutput', false);
    end

    % --- Step 1: get best permutation from overall score ---
    % lambda_penalty=false so per-mode values are the clean factors
    % of the overall score without weight distortion
    [fms_overall, ~, ~, best_perm] = score(A, B, 'lambda_penalty', false);

    % --- Step 2: normalise and align A to B ---
    A = normalize(A);
    B = normalize(B);
    A = arrange(A, best_perm);   % reorder components to match B

    % --- Step 3: per-mode cosine similarity ---
    N  = ndims(A);
    R  = ncomponents(A);
    fms_per_comp = zeros(R, N);   % rows = components, cols = modes

    for n = 1:N
        for r = 1:R
            fms_per_comp(r, n) = abs(A.u{n}(:,r)' * B.u{n}(:,r));
        end
    end

    % --- Step 4: average across components ---
    fms_modes = mean(fms_per_comp, 1);   % 1 x N

    % --- Step 5: print summary ---
    fprintf('  Overall FMS (no lambda penalty): %.4f\n', fms_overall);
    fprintf('  Per-mode FMS:\n');
    for n = 1:N
        fprintf('    %s: %.4f\n', mode_names{n}, fms_modes(n));
    end
    fprintf('  Per-component breakdown:\n');
    for r = 1:R
        fprintf('    Component %d: %s\n', r, ...
            strjoin(arrayfun(@(n) sprintf('%s=%.3f', mode_names{n}, ...
            fms_per_comp(r,n)), 1:N, 'UniformOutput', false), '  '));
    end
end