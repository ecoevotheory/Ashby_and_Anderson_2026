%% guarding_vs_self_guarding.m
% Reproduces deterministic and stochastic simulations for the manuscript
% "Guarding versus self-guarding in innate immunity".
%
% The script:
%   1. Solves the deterministic guarding and self-guarding models.
%   2. Simulates false-positive immune activation with v = 0 and additive
%      noise applied only to the regulatory factor x. Uses the same x-noise
%      trajectory for guarded and self-guarding SDE runs so that 
%      differences are due to model structure.
%
% Notes:
%   - In stochastic runs, x is clamped to [0,1], y to y >= 0, z to [0,1]
%     for guarded runs, z = 0 for self-guarding, and v is fixed at 0.
%   - This file is self-contained and requires no other .m files.

clear; close all; clc;

%% ------------------- Parameters (non-dimensional) -------------------
p.rho         = 5.0;   % r/s (unused in SDE when v is clamped to 0)
p.alpha       = 0.6;   % dR*R0/r (unused in SDE when v is clamped to 0)
p.theta       = 0.95;  % Ra/R0
p.gamma_on    = 0.5;   % g_on/(s*Gmax)
p.gamma_off   = 2.5;   % g_off/s
p.mu_on_guard = 2.0;   % (dI*m_on*Gmax)/(r*s)
p.mu_on_self  = 2.0;   % (dI*m_on)/(r*s)
p.mu_off      = 0.1;   % m_off/s

% Trigger smoothing parameter for the sqrt-smoothed hinge.
p.deltaD      = 1e-3;

% Additive noise strength for x in false-positive SDE runs.
p.sigma_x     = 0.05;

%% ------------------- Simulation settings -------------------
tauSpan = [0, 50];

% Plot colours
col_guard = [0 0.4470 0.7410];
col_self  = [0.8500 0.3250 0.0980];

% Guard parameter sets to compare
%   fast, gain < 1  : gamma_off = 2.5, gamma_on = 0.5
%   fast, no gain   : gamma_off = 2.5, gamma_on = 2.5
%   slow, no gain   : gamma_off = 0.5, gamma_on = 0.5
guardSets = [ ...
    struct('gamma_off',2.5,'gamma_on',0.5,'ls','-'); ...
    struct('gamma_off',2.5,'gamma_on',2.5,'ls',':'); ...
    struct('gamma_off',0.5,'gamma_on',0.5,'ls','--') ...
];

% Viral clearance threshold for deterministic runs.
% Once v crosses below v_thresh, we treat the virus as cleared and enforce
% an absorbing state v \equiv 0 for the remainder of the simulation.
v_thresh = 1e-6;

% Initial conditions
% State vector is Y = [x; v; y; z].
y0_ode = [1; 1e-3; 0; 0];

y0_sde = y0_ode;
y0_sde(2) = 0;  % false-positive runs start with v = 0

% ODE solver tolerances
odeOpts = odeset('RelTol',1e-8,'AbsTol',1e-10);

% SDE step and RNG (Euler-Maruyama)
dt = 1e-3;
rng(1);  % reproducible shared noise trajectory

% Clamp stochastic trajectories to biologically meaningful bounds.
doClamp = true;

%% ===================== 1) Deterministic ODE =========================
% Deterministic infection trajectories with an absorbing cleared-virus rule
% to prevent rebounds once v falls below v_thresh.

% Self-guarding
[tS_ode, YS_ode] = solve_ode_absorbing_clearance( ...
    @guard_models_nd_sqrt, tauSpan, y0_ode, odeOpts, p, "self", v_thresh);

% Guarding for three parameter sets
ntypes = numel(guardSets);
tG_ode = cell(ntypes,1);
YG_ode = cell(ntypes,1);
for k = 1:ntypes
    pk = p;
    pk.gamma_off = guardSets(k).gamma_off;
    pk.gamma_on  = guardSets(k).gamma_on;
    [tG_ode{k}, YG_ode{k}] = solve_ode_absorbing_clearance( ...
        @guard_models_nd_sqrt, tauSpan, y0_ode, odeOpts, pk, "guarded", v_thresh);
end

plot_figure1_ode_multiguard( ...
    tS_ode, YS_ode, tG_ode, YG_ode, guardSets, v_thresh, col_guard, col_self);

%% ===================== 2) Stochastic SDE (false positives) ==========
% Pre-generate a common x-noise trajectory (Wiener increments) and reuse it
% in both architectures so that differences are structural.
tau0 = tauSpan(1);
tauF = tauSpan(2);
N = floor((tauF - tau0)/dt) + 1;
dW_x = sqrt(dt) * randn(N-1, 1);

% Self-guarding
[tS_sde, YS_sde] = simulate_em_common_xnoise( ...
    @guard_models_nd_sde_falsepos_xnoise, tauSpan, y0_sde, dt, p, "self", doClamp, dW_x);

% Guarding for three parameter sets
ntypes = numel(guardSets);
tG_sde = cell(ntypes,1);
YG_sde = cell(ntypes,1);
for k = 1:ntypes
    pk = p;
    pk.gamma_off = guardSets(k).gamma_off;
    pk.gamma_on  = guardSets(k).gamma_on;
    [tG_sde{k}, YG_sde{k}] = simulate_em_common_xnoise( ...
        @guard_models_nd_sde_falsepos_xnoise, tauSpan, y0_sde, dt, pk, "guarded", doClamp, dW_x);
end

plot_figure2_sde_xy( ...
    tS_sde, YS_sde, tG_sde, YG_sde, guardSets, col_guard, col_self, p.theta);

%% ====================================================================
%% Local functions used by the deterministic and stochastic simulations
%% ====================================================================

function dydtau = guard_models_nd_sqrt(~, Y, p, modelFlag)
% Right-hand side for the deterministic non-dimensional model with a
% sqrt-smoothed trigger:
%   u = 1 - x/theta
%   D = 0.5*(u + sqrt(u^2 + deltaD^2))

flag = parseFlag(modelFlag);

x = Y(1);
v = Y(2);
y = Y(3);
z = Y(4);

u = 1 - x / p.theta;
D = 0.5 * (u + sqrt(u*u + p.deltaD*p.deltaD));

dx = (1 - x) - x * v;
dv = p.rho * v * (1 - p.alpha * x - y);

switch flag
    case 1
        dz = p.gamma_on * D * (1 - z) - p.gamma_off * z;
        dy = p.mu_on_guard * z - p.mu_off * y;
    case 2
        dz = 0;
        dy = p.mu_on_self * D - p.mu_off * y;
    otherwise
        error('modelFlag must be 1/''guarded'' or 2/''self''.');
end

dydtau = [dx; dv; dy; dz];
end

function [tOut, YOut] = solve_ode_absorbing_clearance(rhsFun, tauSpan, y0, odeOpts, p, modelFlag, v_thresh)
% Solve the deterministic model until v falls below v_thresh, then continue
% with v held at 0 as an absorbing cleared-virus state.

odeOpts1 = odeset(odeOpts, 'NonNegative', 1:4, ...
    'Events', @(t,Y) evt_v_clear(t, Y, v_thresh));
[t1, Y1, te, Ye] = ode45(@(t,Y) rhsFun(t, Y, p, modelFlag), tauSpan, y0, odeOpts1);

if isempty(te)
    tOut = t1;
    YOut = Y1;
    return;
end

t_clear = te(end);
Y_clear = Ye(end, :)';
Y_clear(2) = 0;

if t_clear >= tauSpan(2)
    tOut = t1;
    YOut = Y1;
    YOut(end,2) = 0;
    return;
end

odeOpts2 = odeset(odeOpts, 'NonNegative', 1:4);
[t2, Y2] = ode45(@(t,Y) rhs_cleared(t, Y, p, modelFlag), [t_clear, tauSpan(2)], Y_clear, odeOpts2);

if numel(t2) > 1
    t2 = t2(2:end);
    Y2 = Y2(2:end,:);
end

tOut = [t1; t2];
YOut = [Y1; Y2];

idx_after = find(tOut >= t_clear);
YOut(idx_after,2) = 0;
end

function dY = rhs_cleared(~, Y, p, modelFlag)
% Deterministic dynamics after viral clearance, with v fixed at 0.

Yc = Y;
Yc(2) = 0;
dY = guard_models_nd_sqrt([], Yc, p, modelFlag);
dY(2) = 0;
end

function [value, isterminal, direction] = evt_v_clear(~, Y, v_thresh)
% Event function for first downward crossing of the viral clearance threshold.

v = Y(2);
value = v - v_thresh;
isterminal = 1;
direction = -1;
end

function [f, G] = guard_models_nd_sde_falsepos_xnoise(~, Y, p, modelFlag)
% Drift and diffusion for the false-positive SDE experiment.
% Here v is fixed at 0 and only x has additive noise.

flag = parseFlag(modelFlag);

x = Y(1);
y = Y(3);
z = Y(4);

u = 1 - x / p.theta;
D = 0.5 * (u + sqrt(u*u + p.deltaD*p.deltaD));

dx = (1 - x);
dv = 0;

switch flag
    case 1
        dz = p.gamma_on * D * (1 - z) - p.gamma_off * z;
        dy = p.mu_on_guard * z - p.mu_off * y;
    case 2
        dz = 0;
        dy = p.mu_on_self * D - p.mu_off * y;
    otherwise
        error('modelFlag must be 1/''guarded'' or 2/''self''.');
end

f = [dx; dv; dy; dz];
G = diag([p.sigma_x, 0, 0, 0]);
end

function [tau, Y] = simulate_em_common_xnoise(sdeFun, tauSpan, y0, dt, p, modelFlag, doClamp, dW_x)
% Euler-Maruyama simulation using a supplied x-noise trajectory.

	tau0 = tauSpan(1);
	tauF = tauSpan(2);
	N = floor((tauF - tau0)/dt) + 1;

if numel(dW_x) ~= (N-1)
    error('dW_x must have length N-1 (%d). Got %d.', N-1, numel(dW_x));
end

tau = tau0 + (0:N-1)' * dt;
Y = zeros(N, numel(y0));
Y(1,:) = y0(:)';

flag = parseFlag(modelFlag);

[~, G0] = sdeFun(tau(1), y0(:), p, modelFlag);
m = size(G0, 2);

for n = 1:N-1
    yn = Y(n,:)';
    tn = tau(n);

    [f, G] = sdeFun(tn, yn, p, modelFlag);

    dW = zeros(m,1);
    dW(1) = dW_x(n);
    yn1 = yn + f*dt + G*dW;

    yn1(2) = 0;  % enforce v \equiv 0

    if doClamp
        yn1(1) = min(max(yn1(1), 0), 1);  % x in [0,1]
        yn1(3) = max(yn1(3), 0);          % y >= 0

        if flag == 1
            yn1(4) = min(max(yn1(4), 0), 1);  % z in [0,1]
        else
            yn1(4) = 0;                       % z = 0 for self-guarding
        end
    end

    Y(n+1,:) = yn1';
end
end

function flag = parseFlag(modelFlag)
% Convert model labels to an internal numeric flag.
%   1 = guarded
%   2 = self-guarding

if ischar(modelFlag) || isstring(modelFlag)
    mf = lower(string(modelFlag));

    if mf == "guarded"
        flag = 1;
    elseif mf == "self" || mf == "self-guarding" || mf == "selfguarding"
        flag = 2;
    else
        error('Unknown modelFlag: %s. Use ''guarded'' or ''self''.', string(modelFlag));
    end

elseif isnumeric(modelFlag) && isscalar(modelFlag) && any(modelFlag == [1, 2])
    flag = modelFlag;

else
    error('modelFlag must be 1, 2, ''guarded'', or ''self''.');
end
end

function plot_figure1_ode_multiguard(tS, YS, tG_cell, YG_cell, guardSets, v_thresh, col_guard, col_self)
% Plot deterministic time series for self-guarding and three guarded
% parameter sets.

figure('Color','w');
ax = gobjects(4,1);

allT = tS(:);
for k = 1:numel(tG_cell)
    allT = [allT; tG_cell{k}(:)]; %#ok<AGROW>
end
xlims = [min(allT), max(allT)];

    function vplot = truncate_v(v)
        vplot = v;
        idx = find(v < v_thresh, 1, 'first');
        if ~isempty(idx)
            vplot(idx:end) = NaN;
        end
    end

ax(1) = subplot(2,2,1); hold on;
plot(tS, YS(:,1), '-', 'LineWidth', 2, 'Color', col_self);
for k = 1:numel(guardSets)
    plot(tG_cell{k}, YG_cell{k}(:,1), ...
        'LineStyle', guardSets(k).ls, 'LineWidth', 2, 'Color', col_guard);
end
set(gca,'FontSize',12,'Box','on');
ylabel('regulatory factor, $x$','Interpreter','latex','FontSize',20);

ax(2) = subplot(2,2,2); hold on;
hSv = plot(tS, truncate_v(YS(:,2)), '-', 'LineWidth', 2, 'Color', col_self);
hGv = gobjects(numel(guardSets),1);
for k = 1:numel(guardSets)
    hGv(k) = plot(tG_cell{k}, truncate_v(YG_cell{k}(:,2)), ...
        'LineStyle', guardSets(k).ls, 'LineWidth', 2, 'Color', col_guard);
end
set(gca,'YScale','log','FontSize',12,'Box','on');
ylabel('viral load, $v$','Interpreter','latex','FontSize',20);

ax(3) = subplot(2,2,3); hold on;
plot(tS, YS(:,3), '-', 'LineWidth', 2, 'Color', col_self);
for k = 1:numel(guardSets)
    plot(tG_cell{k}, YG_cell{k}(:,3), ...
        'LineStyle', guardSets(k).ls, 'LineWidth', 2, 'Color', col_guard);
end
set(gca,'FontSize',12,'Box','on');
ylabel('immune response, $y$','Interpreter','latex','FontSize',20);

ax(4) = subplot(2,2,4); hold on;
for k = 1:numel(guardSets)
    plot(tG_cell{k}, YG_cell{k}(:,4), ...
        'LineStyle', guardSets(k).ls, 'LineWidth', 2, 'Color', col_guard);
end
set(gca,'FontSize',12,'Box','on');
ylabel('guard state, $z$','Interpreter','latex','FontSize',20);

axes(ax(2));
legend([hSv, hGv(1), hGv(2), hGv(3)], ...
    {'Self-guard', ...
     'Guard: fast, gain < 1', ...
     'Guard: fast, no gain', ...
     'Guard: slow, no gain'}, ...
    'FontSize',12,'Location','best','Box','on');

set(ax,'XLim',xlims);
linkaxes(ax,'x');

labels = {'A','B','C','D'};
for i = 1:4
    axes(ax(i)); %#ok<LAXES>
    text(0, 1.01, labels{i}, 'Units','normalized', ...
        'FontWeight','bold', 'FontSize',16, ...
        'HorizontalAlignment','left', 'VerticalAlignment','bottom');
end

annotation(gcf,'textbox',[0 0.01 1 0.05], ...
    'String','time, $\tau$', ...
    'Interpreter','latex', ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', ...
    'EdgeColor','none', ...
    'FontSize',20);
end

function plot_figure2_sde_xy(tS, YS, tG_cell, YG_cell, guardSets, col_guard, col_self, theta)
% Plot stochastic false-positive trajectories for x and y.

figure('Color','w');
ax = gobjects(2,1);

ax(1) = subplot(1,2,1); hold on;
plot(tS, YS(:,1), 'k-', 'LineWidth', 1.5);
plot([0, tS(end)], theta*[1, 1], '--', 'LineWidth', 1.5, 'Color', 'r');
set(gca,'FontSize',12,'Box','on');
ylabel('regulatory factor, $x$','Interpreter','latex','FontSize',20);

ax(2) = subplot(1,2,2); hold on;
hS = plot(tS, YS(:,3), '-', 'LineWidth', 2, 'Color', col_self);
hG = gobjects(numel(guardSets),1);
for k = 1:numel(guardSets)
    hG(k) = plot(tG_cell{k}, YG_cell{k}(:,3), ...
        'LineStyle', guardSets(k).ls, ...
        'LineWidth', 2, ...
        'Color', col_guard);
end
set(gca,'FontSize',12,'Box','on');
ylabel('immune response, $y$','Interpreter','latex','FontSize',20);

legend([hS, hG(1), hG(2), hG(3)], ...
    {'Self-guard', ...
     'Guard: fast, gain < 1', ...
     'Guard: fast, no gain', ...
     'Guard: slow, no gain'}, ...
    'FontSize',12,'Location','best','Box','on');

allT = tS(:);
for k = 1:numel(tG_cell)
    allT = [allT; tG_cell{k}(:)]; %#ok<AGROW>
end
xlims = [min(allT), max(allT)];
set(ax,'XLim',xlims);
linkaxes(ax,'x');

for i = 1:2
    pos = ax(i).Position;
    pos(2) = pos(2) + 0.06;
    pos(4) = pos(4) - 0.06;
    ax(i).Position = pos;
end

labels = {'A','B'};
for i = 1:2
    axes(ax(i)); %#ok<LAXES>
    text(0, 1.01, labels{i}, 'Units','normalized', ...
        'FontWeight','bold', 'FontSize',16, ...
        'HorizontalAlignment','left', 'VerticalAlignment','bottom');
end

annotation(gcf,'textbox',[0 0.01 1 0.04], ...
    'String','time, $\tau$', ...
    'Interpreter','latex', ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', ...
    'EdgeColor','none', ...
    'FontSize',20);
end
