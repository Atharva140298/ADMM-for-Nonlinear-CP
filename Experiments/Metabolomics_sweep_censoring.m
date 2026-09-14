%% Setup environment
clearvars; close all; clc;

%% Model specification
R          = 3;
nb_starts  = 20;
lambda_reg = 1e-1; 

% ADMM NTD specifications
error_measure = 'Frobenius';
param.maxiter = 1000;
param.maxtime = Inf;
param.tol     = 1e-9;
param.tolerr  = 1e-4;
param.display = 0;

%% Load and prepare data
S      = load('Simu_6meta_8time_beta02_set2.mat');
X_orig = S.X_orig;

% Subject selection
subj_idx = find(X_orig.class{1,1} == 1);

% Extract: subjects x metabolites x time, reorder to subjects x time x metabolites
Xraw = X_orig.data(subj_idx, :, :);
Xp   = permute(Xraw, [1 3 2]);

% Metadata
ages = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 4.0];
vars = {"Ins", "Glc", "Pyr", "Lac", "Ala", "Bhb"};

% =========================================================================
%  STEP 1 — Fit CP-NN once on the fully observed data
%
%  CP-NN is the reference model throughout the sweep. It is fitted once
%  here on complete Xp and its factors serve as the gold standard against
%  which CP-imp, CP-WOPT and MinMax are evaluated at every censoring level.
% =========================================================================

X_cpnn = preprocess_centerscale_new(Xp, 1, 0);

ff  = nan(1, nb_starts);
Fac = cell(1, nb_starts);
for i = 1:nb_starts
    rng(i, 'twister');
    [Fac{i}, ~, out{i}] = cp_opt(tensor(X_cpnn), R, ...
        'init','rand','lower',0,'printitn',0,'maxiters',5000);
    ff(i) = out{i}.f;
end
[~, index] = min(ff);
model_CPNN = normalize(Fac{index});
fprintf('CP-NN fitted. Exit: %s\n', out{index}.optout.exit_condition);

% =========================================================================
%  STEP 2 — Sweep over censoring levels
%
%  observed_pct: proportion of data kept (left tail censored)
%  100% -> no censoring, all data observed
%  50%  -> bottom 50% of values treated as sub-LOD, undetected
%
%  At each level:
%  - getInterval(..., p, 'left') sets a_raw = p-th quantile of Xp
%  - X_censored is built by setting sub-LOD entries to NaN
%  - CP-imp  imputes sub-LOD entries at a_raw and fits standard CP
%  - CP-WOPT ignores sub-LOD entries via mask W
%  - MinMax  clips sub-LOD entries to a_raw and fits nonlinear CP
%  - FMS vs CP-NN is recorded for all three methods
% =========================================================================

observed_pct = 100:-5:50;          % [100, 95, 90, ..., 50]
missing_pct  = 100 - observed_pct; % [0,   5,  10, ..., 50] — for x-axis
n_levels     = numel(observed_pct);

fms_CPWOPT = nan(1, n_levels);
fms_MINMAX = nan(1, n_levels);
fms_CPIMP  = nan(1, n_levels);

outDir = fullfile('.', 'results_sweep');
if ~exist(outDir, 'dir'), mkdir(outDir); end

for lv = 1:n_levels

    p_keep = observed_pct(lv) / 100;   % proportion to keep, e.g. 0.80

    fprintf('\n=== Level %d/%d: %.0f%% observed (%.0f%% censored) ===\n', ...
        lv, n_levels, observed_pct(lv), missing_pct(lv));

    % --- Build censored tensor for this level ---
    if p_keep == 1
        % 100% observed — no censoring at all
        a_raw      = min(Xp(:));
        b_raw      = max(Xp(:));
        X_censored = Xp;
    else
        interval   = getInterval(Xp, p_keep, 'left');
        a_raw      = interval(1);
        b_raw      = interval(2);
        X_censored = Xp;
        X_censored(Xp < a_raw) = NaN;
        X_censored(Xp > b_raw) = NaN;
    end

    W = ~isnan(X_censored);
    fprintf('  Censored entries: %.1f%%\n', 100*mean(~W(:)));
    plot_missing_slices(W, lv, missing_pct, vars, ages, outDir)

    % -----------------------------------------------------------------------
    %  CP-WOPT input
    %  Scale X_censored with preprocess_centerscale_new — nanmean inside
    %  handles NaN entries. Then zero sub-LOD entries before passing to solver.
    % -----------------------------------------------------------------------
    X_cpwopt      = X_censored;
    X_cpwopt      = preprocess_centerscale_new(X_cpwopt, 1, 0);
    X_cpwopt(~W)  = 0;

    % -----------------------------------------------------------------------
    %  MinMax / CP-imp input
    %  Clip X_censored to [a_raw, b_raw] — NaN entries become a_raw via
    %  max(a_raw, NaN) = a_raw in MATLAB. Then scale. param.a/b derived
    %  from the scaled tensor so bounds match what admm_ntd receives.
    %  CP-imp receives the SAME tensor, so the two differ only in the model.
    % -----------------------------------------------------------------------
    X_MinMax      = min(b_raw, max(a_raw, X_censored));
    X_MinMax      = preprocess_centerscale_new(X_MinMax, 1, 0);
    param.a       = min(X_MinMax(:));
    param.b       = max(X_MinMax(:));
    param.factors = 'nonneg';

    % --- Fit CP-WOPT ---
    ff  = nan(1, nb_starts);
    Fac = cell(1, nb_starts);
    for i = 1:nb_starts
        rng(i, 'twister');
        [Fac{i}, ~, out{i}] = cp_wopt_wridge(tensor(X_cpwopt), tensor(double(W)), R, ...
            'init','rand','lower',0,'verbosity',0,'skip_zeroing',true, 'lambda', lambda_reg);
        ff(i) = out{i}.f;
    end
    [~, index]        = min(ff);
    model_CPWOPT      = normalize(Fac{index});
    fms_CPWOPT(lv)    = score(model_CPWOPT, model_CPNN,'lambda_penalty',false);
    fprintf('  CP-WOPT FMS: %.4f\n', fms_CPWOPT(lv));

    % --- Fit CP-imp (standard CP on the threshold-imputed tensor) ---
    %  Same input as MinMax: sub-LOD entries imputed at a_raw, then scaled.
    %  Differs only in the model: the threshold is treated as a value,
    %  not as a bound.
    ff  = nan(1, nb_starts);
    Fac = cell(1, nb_starts);
    for i = 1:nb_starts
        rng(i, 'twister');
        [Fac{i}, ~, out{i}] = cp_opt(tensor(X_MinMax), R, ...
            'init','rand','printitn',0,'maxiters',5000);
        ff(i) = out{i}.f;
    end
    [~, index]     = min(ff);
    model_CPIMP    = normalize(Fac{index});
    fms_CPIMP(lv)  = score(model_CPIMP, model_CPNN,'lambda_penalty',false);
    fprintf('  CP-imp  FMS: %.4f\n', fms_CPIMP(lv));

    % --- Fit MinMax-ADMM ---
    ff       = nan(1, nb_starts);
    results1 = cell(1, nb_starts);
    parfor n = 1:nb_starts
        RandStream.setGlobalStream(RandStream('twister', 'Seed', n));
        param2    = param;
        param2.A0 = rand(size(X_censored,1), R);
        param2.B0 = rand(size(X_censored,2), R);
        param2.C0 = rand(size(X_censored,3), R);
        results1{n} = admm_ntd(X_MinMax, R, 'MinMax', error_measure, param2);
        ff(n)       = results1{n}.objective(end);
    end
    [~, index]     = min(ff);
    U{1} = results1{index}.A;
    U{2} = results1{index}.B;
    U{3} = results1{index}.C;
    model_MINMAX   = normalize(ktensor(U));
    fms_MINMAX(lv) = score(model_MINMAX, model_CPNN,'lambda_penalty',false);
    fprintf('  MinMax  FMS: %.4f\n', fms_MINMAX(lv));

end

%% Save sweep results

save(fullfile(outDir, 'sweep_results.mat'), ...
    'observed_pct', 'missing_pct', ...
    'fms_CPWOPT', 'fms_MINMAX', 'fms_CPIMP', ...
    'model_CPNN', 'R', 'ages', 'vars', 'Xp');

%% Plot FMS vs missing percentage
fig_sweep = figure('Name', 'FMS sweep — censoring level');

plot(missing_pct, fms_CPWOPT, 'b-o', ...
    'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', 'b', ...
    'DisplayName', 'CP-WOPT');
hold on;
plot(missing_pct, fms_CPIMP, '-^', ...
    'Color', [0 0.6 0], 'LineWidth', 2, 'MarkerSize', 7, ...
    'MarkerFaceColor', [0 0.6 0], 'DisplayName', 'CP-imp');
plot(missing_pct, fms_MINMAX, 'r-s', ...
    'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', 'r', ...
    'DisplayName', 'NN-MinMax');

% Reference line at FMS = 1
yline(1, 'k--', 'LineWidth', 1, 'DisplayName', 'Perfect recovery');

xlabel('Censored entries (%)','FontSize',13);
ylabel('FMS vs CP-NN reference','FontSize',13);
title('Factor recovery under increasing left-censoring','FontSize',14);
xlim([min(missing_pct)-2, max(missing_pct)+2]);
ylim([0 1.05]);
xticks(missing_pct);
legend('Location','southwest','FontSize',11);
grid on;

% Annotate each point with its FMS value
for lv = 1:n_levels
    text(missing_pct(lv), fms_CPWOPT(lv)+0.03, sprintf('%.2f', fms_CPWOPT(lv)), ...
        'HorizontalAlignment','center','FontSize',8,'Color','b');
    text(missing_pct(lv), fms_CPIMP(lv)+0.07, sprintf('%.2f', fms_CPIMP(lv)), ...
        'HorizontalAlignment','center','FontSize',8,'Color',[0 0.5 0]);
    text(missing_pct(lv), fms_MINMAX(lv)-0.04, sprintf('%.2f', fms_MINMAX(lv)), ...
        'HorizontalAlignment','center','FontSize',8,'Color','r');
end

savefig(fig_sweep, fullfile(outDir,'fig_fms_sweep.fig'));
exportgraphics(fig_sweep, fullfile(outDir,'fig_fms_sweep.png'),'Resolution',150);

fprintf('\nSweep complete. Results saved to %s\n', outDir);

%% Print a compact summary table
fprintf('\n%-10s %10s %10s %10s\n', 'Censored%', 'CP-imp', 'CP-WOPT', 'NN-MinMax');
fprintf('---------------------------------------------------\n');
for lv = 1:n_levels
    fprintf('%-10d %10.4f %10.4f %10.4f\n', missing_pct(lv), ...
        fms_CPIMP(lv), fms_CPWOPT(lv), fms_MINMAX(lv));
end

%% Visualizing the missingness in the tensor

function plot_missing_slices(W, lv, missing_pct, vars, ages, outDir)
% Plots missingness pattern slice by slice along the metabolite mode.
% Each subplot is one metabolite — a subjects x time grid.
% White = missing (sub-LOD), coloured = observed.
%
% W           : logical mask (subjects x time x metabolites), TRUE = observed
% lv          : sweep level index (for filename)
% missing_pct : scalar, e.g. 20 for 20% missing
% vars        : cell array of metabolite names
% ages        : time point labels
% outDir      : output directory

    [I, J, K] = size(W);   % subjects x time x metabolites

    fig = figure('Name', sprintf('Missingness slices — %d%% censored', missing_pct), ...
    'Position', [100 100 1200 600], 'Visible', 'off');
    clf(fig);
    for k = 1:K
        subplot(2, 3, k);

        % Slice along metabolite mode — subjects x time
        slice_W = W(:,:,k);   % I x J logical matrix

        % Display: 1 = observed (coloured), 0 = missing (white)
        imagesc(double(slice_W));
        colormap(gca, [1 1 1; 0.20 0.60 0.50]);   % white=missing, teal=observed
        clim([0 1]);

        % Axes labels
        xticks(1:J);
        xticklabels(arrayfun(@(a) sprintf('%.2g', a), ages, 'UniformOutput', false));
        xlabel('Time (years)', 'FontSize', 9);
        ylabel('Subject index', 'FontSize', 9);
        

        % Add text annotations showing O/X per cell
        for i = 1:I
            for j = 1:J
                if slice_W(i,j)
                    txt = 'O';   col = [0.1 0.4 0.3];
                else
                    txt = 'X';   col = [0.8 0.1 0.1];
                end
                text(j, i, txt, 'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', 'FontSize', 6, 'Color', col);
            end
        end

        % Show percentage missing for this metabolite
        pct_miss = 100 * mean(~slice_W(:));
        title(sprintf('Metabolite: %s  (%.1f%% censored)', vars{k}, pct_miss), 'FontSize', 11);
    end

    % Isolated title via text on hidden axes
    ax_top = axes(fig, 'Position', [0 0.95 1 0.04], 'Visible', 'off');
    text(ax_top, 0.5, 0.5, ...
        sprintf('Missingness pattern per metabolite — %d%% entries censored', missing_pct), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Units', 'normalized', ...
        'Parent', ax_top);


    exportgraphics(fig, fullfile(outDir, sprintf('fig_slices_missing_lv%02d.png', lv)), ...
        'Resolution', 150);
    close(fig);
end