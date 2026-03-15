%% 第三轮实验绘图：分母定义 + 一致性校验 + Premium vs DeltaIV
% 使用前请修改 csv_path 为你的 price_results_latest.csv 路径
% 
% 关键修改:
%   1. 明确 strict early exercise ratio 分母 = all paths
%   2. 添加 decomposition gap 校验图
%   3. 添加 premium vs DeltaIV 辅助图
%   4. 添加 ATM 峰值逻辑解释注释

clear; clc; close all

%% ========== 1. 读取数据 ==========
csv_path = 'results第二轮/price_results_latest.csv';
T = readtable(csv_path, 'Encoding', 'UTF-8');

% 过滤掉带 error 的行
if ismember('error', T.Properties.VariableNames)
    hasErr = cellfun(@(x) ~isempty(x) && ischar(x) && ~isempty(strtrim(x)), T.error);
    T = T(~hasErr, :);
end

% 统一模型顺序
model_order = {'M0','M1','M2','M3','M4','M5'};
model = T.model;
if iscategorical(model), model = cellstr(model); end
if iscell(model) && ~isempty(model) && ischar(model{1}), model = model(:); end

% 提取数据
moneyness   = T.moneyness;
T_mat       = T.T;
H           = T.H;
american_price  = T.american_price;
european_price  = T.european_price;
american_premium = T.american_premium;

% 三个关键比例（分母 = all paths）
% 注意：新字段名
if ismember('strict_early_exercise_ratio', T.Properties.VariableNames)
    strict_early_exercise_ratio = T.strict_early_exercise_ratio;
elseif ismember('early_exercise_ratio_before_maturity', T.Properties.VariableNames)
    strict_early_exercise_ratio = T.early_exercise_ratio_before_maturity;
else
    strict_early_exercise_ratio = nan(size(american_price));
end

if ismember('exercise_at_maturity_ratio', T.Properties.VariableNames)
    exercise_at_maturity_ratio = T.exercise_at_maturity_ratio;
else
    exercise_at_maturity_ratio = nan(size(american_price));
end

if ismember('total_exercise_ratio', T.Properties.VariableNames)
    total_exercise_ratio = T.total_exercise_ratio;
else
    total_exercise_ratio = nan(size(american_price));
end

if ismember('itm_ratio', T.Properties.VariableNames)
    itm_ratio = T.itm_ratio;
else
    itm_ratio = nan(size(american_price));
end

% 新增: decomposition gap
if ismember('decomposition_gap', T.Properties.VariableNames)
    decomposition_gap = T.decomposition_gap;
else
    decomposition_gap = nan(size(american_price));
end

% IV
if ismember('iv_european', T.Properties.VariableNames)
    iv_european = T.iv_european;
else
    iv_european = nan(size(american_price));
end
if ismember('iv_american_proxy', T.Properties.VariableNames)
    iv_american_proxy = T.iv_american_proxy;
else
    iv_american_proxy = nan(size(american_price));
end
delta_iv = iv_american_proxy - iv_european;

% 固定 T
T_vals = unique(T_mat);
T_fix = T_vals(round(length(T_vals)/2));

colors = lines(6);
smooth_win = 3;

%% ========== 图 1：分母定义明确说明 ==========
% 关键说明：所有比例的分母 = all simulated paths
figure('Name', 'Fig1 Early Exercise Ratios with Clear Denominator');
sgtitle(['All ratios use "all paths" as denominator', ', T = ' num2str(T_fix)]);

% 子图1: ITM ratio
subplot(1,3,1); hold on
for m = 1:6
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1'), idx = idx & (H == 0.5); else, h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end; end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = itm_ratio(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', [mod ' (ITM/All)'], 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('Ratio');
title('ITM Ratio at Maturity\n(denominator = all paths)'); legend('Location', 'best'); grid on; hold off

% 子图2: Strict early exercise ratio (t < T)
subplot(1,3,2); hold on
for m = 1:6
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1'), idx = idx & (H == 0.5); else, h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end; end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = strict_early_exercise_ratio(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('Ratio');
title('Strict Early Exercise Ratio (t < T)\n(denominator = all paths)'); legend('Location', 'best'); grid on; hold off

% 子图3: Exercise at maturity ratio
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
xlabel('Moneyness K/S_0'); ylabel('Ratio');
title('Exercise at Maturity Ratio (t = T)\n(denominator = all paths)'); legend('Location', 'best'); grid on; hold off

%% ========== 图 2：Decomposition Gap 校验 ==========
% 验证: total ≈ strict_early + maturity
figure('Name', 'Fig2 Decomposition Gap Validation');
sgtitle(['Decomposition Consistency Check: total = strict + maturity', ', T = ' num2str(T_fix)]);

% 选择几个代表性模型
model_examples = {'M0', 'M2', 'M5'};

for i = 1:length(model_examples)
    mod = model_examples{i};
    subplot(1,3,i); hold on
    
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1')
        idx = idx & (H == 0.5);
    else
        h_vals = unique(H(idx));
        if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end
    end
    
    if sum(idx) == 0, continue; end
    
    x = moneyness(idx);
    y_total = total_exercise_ratio(idx);
    y_sum = strict_early_exercise_ratio(idx) + exercise_at_maturity_ratio(idx);
    y_gap = decomposition_gap(idx);
    
    [x_s, ord] = sort(x);
    plot(x_s, y_total(ord), '-o', 'Color', [0.8 0 0], 'DisplayName', 'total', 'LineWidth', 2);
    plot(x_s, y_sum(ord), '--s', 'Color', [0 0.8 0], 'DisplayName', 'strict+maturity', 'LineWidth', 2);
    plot(x_s, y_gap(ord), '-^', 'Color', [0 0 0.8], 'DisplayName', 'gap', 'LineWidth', 2);
    
    xlabel('Moneyness K/S_0'); ylabel('Ratio');
    title(sprintf('%s: gap should be ~0', mod));
    legend('Location', 'best'); grid on; hold off
    
    % 输出 gap 统计
    gap_valid = y_gap(~isnan(y_gap));
    if ~isempty(gap_valid)
        fprintf('%s: Mean gap = %.2e, Max |gap| = %.2e\n', mod, mean(gap_valid), max(abs(gap_valid)));
    end
end

%% ========== 图 3：Premium vs DeltaIV 辅助分析 ==========
% 用于解释为什么某些模型 DeltaIV 更高
figure('Name', 'Fig3 Premium vs DeltaIV Analysis');
subplot(1,2,1); hold on
for m = 1:3
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1'), idx = idx & (H == 0.5); else, h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end; end
    if sum(idx) == 0, continue; end
    x = american_premium(idx); y = delta_iv(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
end
xlabel('American Premium'); ylabel('\Delta IV');
title(['Premium vs \Delta IV (benchmark), T=' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

subplot(1,2,2); hold on
for m = 4:6
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end
    if sum(idx) == 0, continue; end
    x = american_premium(idx); y = delta_iv(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
end
xlabel('American Premium'); ylabel('\Delta IV');
title(['Premium vs \Delta IV (rough+jump), T=' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

%% ========== 图 4：American Premium ==========
figure('Name', 'Fig4 American Premium');
subplot(1,2,1); hold on
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

subplot(1,2,2); hold on
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

%% ========== 图 5：Delta IV ==========
figure('Name', 'Fig5 Delta IV');
subplot(1,2,1); hold on
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
xlabel('Moneyness K/S_0'); ylabel('\Delta IV');
title(['\Delta IV (benchmark), T = ' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

subplot(1,2,2); hold on
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
xlabel('Moneyness K/S_0'); ylabel('\Delta IV');
title(['\Delta IV (rough+jump), T = ' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

%% ========== 图 6：European IV Smile ==========
figure('Name', 'Fig6 European IV Smile');
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
xlabel('Moneyness K/S_0'); ylabel('European Implied Vol (BS)');
title(['European IV Smile (put), T = ' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

%% ========== 重要说明输出 ==========
fprintf('\n========================================\n');
fprintf('绘图完成。关键说明:\n');
fprintf('========================================\n');
fprintf('1. 所有 early exercise ratio 的分母 = all simulated paths\n');
fprintf('2. strict early exercise + exercise at maturity ≈ total exercise\n');
fprintf('3. decomposition gap (total - sum) 应接近 0\n');
fprintf('4. Premium vs DeltaIV 图用于解释 DeltaIV 差异来源\n');
fprintf('\n=== ATM 峰值解释 ===\n');
fprintf('strict early exercise ratio 在 ATM 附近达到峰值，原因如下:\n');
fprintf('  - 太深虚值 (K>>S0): 行权价值低，无必要提前行权\n');
fprintf('  - 太深实值 (K<<S0): 到期执行占比高，严格提前行权未必最高\n');
fprintf('  - ATM/略 ITM (K≈S0): continuation value 与 intrinsic value\n');
fprintf('    差距最敏感，博弈最激烈，提前行权概率最高\n');
fprintf('========================================\n');