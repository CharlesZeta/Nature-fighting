%% 第二轮实验绘图：European IV / American Premium / Premium by H / Early Exercise
% 使用前请修改 csv_path 为你的 price_results_latest.csv 路径
% 第一优先级：含 M0，无 colorbar，区分 European IV 与 American IV proxy
% 第二优先级：图 1–4 见下方
clear; clc; close all

% ========== 1. 读取数据 ==========
csv_path = 'results第二轮/price_results_latest.csv';
% 或完整路径，例如: csv_path = 'C:\Users\Administrator\Desktop\论文\results第二轮\price_results_latest.csv';
T = readtable(csv_path, 'Encoding', 'UTF-8');

% 若列名带 BOM，可先查看: disp(T.Properties.VariableNames)
% 兼容旧 CSV：若无 iv_european，用 european_price 在后文不画 European IV 或需先重跑实验

% 过滤掉带 error 的行
if ismember('error', T.Properties.VariableNames)
    hasErr = cellfun(@(x) ~isempty(x) && ischar(x) && ~isempty(strtrim(x)), T.error);
    T = T(~hasErr, :);
end

% 统一模型顺序 M0–M5，确保 M0 一定被画
model_order = {'M0','M1','M2','M3','M4','M5'};
model = T.model;
if iscategorical(model)
    model = cellstr(model);
end
if iscell(model) && ~isempty(model) && ischar(model{1})
    % 已是 cellstr
else
    model = cellstr(string(model));
end

% 提取列（列名需与 CSV 一致；若为 iv_american_proxy 则用该列）
moneyness   = T.moneyness;
T_mat       = T.T;
H           = T.H;
american_price  = T.american_price;
european_price  = T.european_price;
american_premium = T.american_premium;
early_exercise_ratio = T.early_exercise_ratio;
% 新增：提前行权相关拆分
if ismember('early_exercise_ratio_before_maturity', T.Properties.VariableNames)
    early_exercise_ratio_before_maturity = T.early_exercise_ratio_before_maturity;
else
    early_exercise_ratio_before_maturity = nan(size(early_exercise_ratio));
end
if ismember('exercise_at_maturity_ratio', T.Properties.VariableNames)
    exercise_at_maturity_ratio = T.exercise_at_maturity_ratio;
else
    exercise_at_maturity_ratio = nan(size(early_exercise_ratio));
end
if ismember('itm_ratio', T.Properties.VariableNames)
    itm_ratio = T.itm_ratio;
else
    itm_ratio = nan(size(early_exercise_ratio));
end

% European IV（新实验才有）
if ismember('iv_european', T.Properties.VariableNames)
    iv_european = T.iv_european;
else
    iv_european = nan(size(american_price));
end
% American IV proxy
if ismember('iv_american_proxy', T.Properties.VariableNames)
    iv_american_proxy = T.iv_american_proxy;
elseif ismember('implied_vol_proxy', T.Properties.VariableNames)
    iv_american_proxy = T.implied_vol_proxy;
else
    iv_american_proxy = nan(size(american_price));
end

% 行权边界（分位数定义，样本不足为 NaN）
if ismember('exercise_boundary_T05', T.Properties.VariableNames)
    eb_T05 = T.exercise_boundary_T05;
else
    eb_T05 = nan(size(american_price));
end
% 行权样本数：仅当 exercise_count_T05 >= 此值时才画 boundary 点（与 Python MIN_BOUNDARY_SAMPLES 一致）
if ismember('exercise_count_T05', T.Properties.VariableNames)
    exc_count_T05 = T.exercise_count_T05;
else
    exc_count_T05 = nan(size(american_price));
end
MIN_BOUNDARY_DISPLAY = 30;

% 固定一个 T 用于图 1、图 2
T_vals = unique(T_mat);
T_fix = T_vals(1);
if length(T_vals) > 1
    T_fix = T_vals(round(length(T_vals)/2));  % 或选 T_vals(1)
end

% 颜色：M0–M5 固定
colors = lines(6);

%% 图 1：European IV Smile（European put，x: moneyness K/S0，y: BS implied vol from European price）
% 模型 M0–M5，固定 T。图注/正文请写明：European put，x 轴为 moneyness K/S0，BS IV 由 European 价格反推。
figure('Name', 'Fig1 European IV Smile');
hold on
for m = 1:6
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1')
        idx = idx & (H == 0.5);
    else
        h_vals = unique(H(idx));
        if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end
    end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = iv_european(idx);
    if all(isnan(y)), continue; end
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('European Implied Vol (BS, from European put price)');
title(['European IV Smile (put), T = ' num2str(T_fix) ', x = K/S_0']);
legend('Location', 'best'); grid on; hold off

% 轻微平滑仅用于展示（窗口 3，保留原始数据用于统计）
smooth_win = 3;  % 设为 1 则不平滑

%% 图 2a：American Premium — 经典基准 M0, M1, M2
figure('Name', 'Fig2a American Premium (M0,M1,M2)');
hold on
for m = 1:3
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1'), idx = idx & (H == 0.5); else, h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end; end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = american_premium(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    if smooth_win > 1, y_s = movmean(y_s, smooth_win, 'Endpoints', 'shrink'); end
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('American - European');
title(['American Premium (benchmark), T = ' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

%% 图 2b：American Premium — Rough + jump M3, M4, M5
figure('Name', 'Fig2b American Premium (M3,M4,M5)');
hold on
for m = 4:6
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = american_premium(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    if smooth_win > 1, y_s = movmean(y_s, smooth_win, 'Endpoints', 'shrink'); end
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('American - European');
title(['American Premium (rough+jump), T = ' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

%% 图 3：不同 H 下的 Premium（Rough 模型；H=0.5 标为 Markovian benchmark）
figure('Name', 'Fig3 Premium by Hurst');
hold on
mod_rough = 'M2';
idx_base = strcmp(model, mod_rough) & (T_mat == T_fix);
h_vals = unique(H(idx_base));
for h = 1:length(h_vals)
    idx = idx_base & (H == h_vals(h));
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = american_premium(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    if abs(h_vals(h) - 0.5) < 0.01
        leg = 'Markovian benchmark (H=0.5)';
    else
        leg = ['H = ' num2str(h_vals(h))];
    end
    plot(x_s, y_s, '-o', 'DisplayName', leg, 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('American - European');
title(['Premium by H, ' mod_rough ', T = ' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

%% 图 4：Early exercise（新定义：严格区分 ITM / 到期前 / 到期日行权）
figure('Name', 'Fig4 Early Exercise Ratios');
% 子图 1：ITM ratio、提前行权比例、到期行权比例 按模型
subplot(1,3,1); hold on
for m = 1:6
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1'), idx = idx & (H == 0.5); else, h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end; end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = itm_ratio(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', [mod ' (ITM)'], 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('ITM Ratio (S_T < K)');
title('ITM Ratio at Maturity'); legend('Location', 'best'); grid on; hold off

subplot(1,3,2); hold on
for m = 1:6
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1'), idx = idx & (H == 0.5); else, h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end; end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = early_exercise_ratio_before_maturity(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('Early Exercise Ratio (before maturity)');
title('Early Exercise (strict, t < T)'); legend('Location', 'best'); grid on; hold off

subplot(1,3,3); hold on
for m = 1:6
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1'), idx = idx & (H == 0.5); else, h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end; end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = exercise_at_maturity_ratio(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('Exercise at Maturity Ratio');
title('Exercise at Maturity (t = T)'); legend('Location', 'best'); grid on; hold off

sgtitle(['Early Exercise Breakdown, T = ' num2str(T_fix)]);

%% 图 5：American IV proxy（与 European 区分，对照用）
figure('Name', 'Fig5 American IV Proxy');
hold on
for m = 1:6
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1'), idx = idx & (H == 0.5); else, h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end; end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = iv_american_proxy(idx);
    if all(isnan(y)), continue; end
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-s', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('American Implied Vol (proxy, BS from American price)');
title(['American IV Proxy, T = ' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

%% 图 6：ΔIV = American IV proxy − European IV（美式提前行权价值映射到 IV 的差异）
% 分组展示：经典基准 vs rough+jump
delta_iv = iv_american_proxy - iv_european;

figure('Name', 'Fig6a Delta IV (M0,M1,M2)');
hold on
for m = 1:3
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1'), idx = idx & (H == 0.5); else, h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end; end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = delta_iv(idx);
    if all(isnan(y)), continue; end
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('\Delta IV (American proxy - European)');
title(['\Delta IV (benchmark), T = ' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

figure('Name', 'Fig6b Delta IV (M3,M4,M5)');
hold on
for m = 4:6
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = delta_iv(idx);
    if all(isnan(y)), continue; end
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('\Delta IV (American proxy - European)');
title(['\Delta IV (rough+jump), T = ' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

disp('Done. Fig1: European IV Smile (put). Fig2a/2b: American Premium (benchmark / rough+jump). Fig3: Premium by H. Fig4: Early exercise (ITM/early/maturity). Fig5: American IV proxy. Fig6a/6b: Delta IV (benchmark / rough+jump).');
