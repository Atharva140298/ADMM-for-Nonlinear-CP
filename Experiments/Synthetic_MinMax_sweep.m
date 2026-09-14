close all; clear; clc;

%% =========================================================================
%  Synthetic MinMax experiment: sweep over clipping level
%
%  Same tensors as Synthetic_MinMax.m, but the observation window shrinks
%  symmetrically from both sides as the clipping level increases:
%
%      a = quantile(Y, p/2),   b = quantile(Y, 1 - p/2)
%
%  so that p% of entries are clipped, half at each end. The sweep runs to
%  70%, matching the total clipped fraction of the fixed [0,1] experiment.
%
%  Methods : CP-missing, CP-imputation, NCP (MinMax)
%  Reported: FMS of the best-objective run vs the TRUE factors,
%            mean +/- std over n_tensors independently generated tensors.
% =========================================================================

%% Model
model         = 'MinMax';
error_measure = 'Frobenius';

param.maxiter = 1000;
param.maxtime = Inf;
param.tol     = 1e-5;
param.display = 0;

%% Experiment settings
R  = 3;
I  = 50;  J = 40;  K = 30;
n_tensors  = 1;             % lower to 3 for a quick first pass
nb_starts  = 20;
lambda_reg = 1e-1;

clip_pct = 0:10:90;          % total % clipped, half at each end
n_lv     = numel(clip_pct);

fms_mis = nan(n_tensors, n_lv);
fms_imp = nan(n_tensors, n_lv);
fms_ncp = nan(n_tensors, n_lv);
obs_pct = nan(n_tensors, n_lv);

outDir = fullfile('.','results_synth_minmax_sweep');
if ~exist(outDir,'dir'), mkdir(outDir); end

%% =========================================================================
for t = 1:n_tensors

    rng(100*t, 'twister');

    A = cell(3,1);
    A{1} = randn(I,R);
    A{2} = randn(J,R);
    A{3} = randn(K,R);
    Y    = full(ktensor(A));

    TT = cell(3,1);
    for i = 1:3, TT{i} = A{i} ./ vecnorm(A{i}); end
    TrueModel = ktensor(TT);

    fprintf('\n########## Tensor %d/%d ##########\n', t, n_tensors);

    for lv = 1:n_lv

        p = clip_pct(lv)/100;

        % --- symmetric window: shrinks from both sides ---
        p_keep = 1 - clip_pct(lv)/100;              % proportion KEPT
        if p_keep == 1
            a = min(Y.data(:));  b = max(Y.data(:));
        else
            interval = getInterval(Y.data, p_keep, 'symmetric');
            a = interval(1);  b = interval(2);
        end

        X   = min(b, max(a, Y.data));
        obs = (Y.data >= a) & (Y.data <= b);
        obs_pct(t,lv) = 100*mean(obs(:));

        param.a = a;
        param.b = b;

        fprintf(' %2d%% clipped  (a=%.3f, b=%.3f, %.1f%% observed)\n', ...
            clip_pct(lv), a, b, obs_pct(t,lv));

        % ---------------- NCP ----------------
        o = nan(nb_starts,1);  s = nan(nb_starts,1);
        for n = 1:nb_starts
            res = admm_ntd(X, R, model, error_measure, param);
            bad = any(isnan([res.A(:);res.B(:);res.C(:)])) ...
                || any(vecnorm(res.A)==0) || any(vecnorm(res.B)==0) ...
                || any(vecnorm(res.C)==0);
            if bad
                o(n) = Inf;  s(n) = NaN;
            else
                U = {res.A./vecnorm(res.A), res.B./vecnorm(res.B), ...
                    res.C./vecnorm(res.C)};
                o(n) = res.objective(end);
                s(n) = score(TrueModel, ktensor(U), 'lambda_penalty', false);
            end
        end
        [~, ib] = min(o);   fms_ncp(t,lv) = s(ib);

        % ------------- CP-imputation -------------
        o = nan(nb_starts,1);  s = nan(nb_starts,1);
        for n = 1:nb_starts
            [Fac, ~, out] = cp_opt(tensor(X), R, 'init','randn', 'printitn',0);
            o(n) = out.f;
            s(n) = score(TrueModel, Fac, 'lambda_penalty', false);
        end
        [~, ib] = min(o);   fms_imp(t,lv) = s(ib);

        % -------------- CP-missing --------------
        W  = tensor(double(obs));
        YY = tensor(Y.data .* obs);
        o = nan(nb_starts,1);  s = nan(nb_starts,1);
        for n = 1:nb_starts
            [FacW, ~, outW] = cp_wopt_wridge(YY, W, R, ...
                'init','randn','verbosity',0,'skip_zeroing',true, ...
                'lambda',lambda_reg);
            o(n) = outW.f;
            s(n) = score(TrueModel, FacW, 'lambda_penalty', false);
        end
        [~, ib] = min(o);   fms_mis(t,lv) = s(ib);

        fprintf('        CP-missing %.4f | CP-imputation %.4f | NCP %.4f\n', ...
            fms_mis(t,lv), fms_imp(t,lv), fms_ncp(t,lv));
    end
end

%% =========================================================================
%  Summary
%% =========================================================================
m_mis = mean(fms_mis,1);  sd_mis = std(fms_mis,0,1);
m_imp = mean(fms_imp,1);  sd_imp = std(fms_imp,0,1);
m_ncp = mean(fms_ncp,1);  sd_ncp = std(fms_ncp,0,1);

fprintf('\n=================================================================\n');
fprintf(' MinMax sweep: mean FMS over %d tensors (best of %d starts)\n', ...
    n_tensors, nb_starts);
fprintf('=================================================================\n');
fprintf('%-10s %7s %16s %16s %16s\n', ...
    'Clipped%','Obs%','CP-missing','CP-imputation','NCP');
for lv = 1:n_lv
    fprintf('%-10d %7.1f %9.3f+/-%.2f %9.3f+/-%.2f %9.3f+/-%.2f\n', ...
        clip_pct(lv), mean(obs_pct(:,lv)), ...
        m_mis(lv), sd_mis(lv), m_imp(lv), sd_imp(lv), m_ncp(lv), sd_ncp(lv));
end

%% Line plot, same style as the metabolomics sweep
fig = figure('Name','Synthetic MinMax sweep');

plot(clip_pct, m_mis, 'b-o', 'LineWidth',2, 'MarkerSize',7, ...
    'MarkerFaceColor','b', 'DisplayName','CP-missing'); hold on;
plot(clip_pct, m_imp, '-^', 'Color',[0 0.6 0], 'LineWidth',2, ...
    'MarkerSize',7, 'MarkerFaceColor',[0 0.6 0], 'DisplayName','CP-imputation');
plot(clip_pct, m_ncp, 'r-s', 'LineWidth',2, 'MarkerSize',7, ...
    'MarkerFaceColor','r', 'DisplayName','NCP');
yline(1, 'k--', 'LineWidth',1, 'DisplayName','Perfect recovery');

xlabel('Clipped entries (%)','FontSize',13);
ylabel('FMS vs ground truth','FontSize',13);
title('Synthetic MinMax: factor recovery under increasing clipping','FontSize',13);
xlim([-3, max(clip_pct)+3]);  ylim([0 1.08]);
xticks(clip_pct);
legend('Location','southwest','FontSize',11);
grid on;

for lv = 1:n_lv
    text(clip_pct(lv), m_mis(lv)+0.03, sprintf('%.2f',m_mis(lv)), ...
        'HorizontalAlignment','center','FontSize',8,'Color','b');
    text(clip_pct(lv), m_imp(lv)-0.05, sprintf('%.2f',m_imp(lv)), ...
        'HorizontalAlignment','center','FontSize',8,'Color',[0 0.5 0]);
    text(clip_pct(lv), m_ncp(lv)+0.03, sprintf('%.2f',m_ncp(lv)), ...
        'HorizontalAlignment','center','FontSize',8,'Color','r');
end

savefig(fig, fullfile(outDir,'fig_synth_minmax_sweep.fig'));
exportgraphics(fig, fullfile(outDir,'fig_synth_minmax_sweep.png'),'Resolution',150);

save(fullfile(outDir,'synth_minmax_sweep.mat'), ...
    'clip_pct','fms_mis','fms_imp','fms_ncp','obs_pct', ...
    'R','I','J','K','n_tensors','nb_starts','lambda_reg');
