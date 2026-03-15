%% 第二轮实验绘图 V2：带误差条、分母定义说明、收敛图
% 使用前请修改 csv_path 为你的 price_results_latest.csv 路径
% V2 版本新增:
%   1. 明确各比例的数学定义（分母说明）
%   2. 添加误差条 (seed-based CI)
%   3. 收敛实验图 (paths / Nt / Nexp)
%   4. 量化对比表 (summary table)
%   5. 分位数边界统计

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

moneyness   = T.moneyness;
T_mat       = T.T;
H           = T.H;
american_price  = T.american_price;
european_price  = T.european_price;
american_premium = T.american_premium;
early_exercise_ratio = T.early_exercise_ratio;

% 提前行权相关拆分
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

% 固定 T
T_vals = unique(T_mat);
T_fix = T_vals(round(length(T_vals)/2));

colors = lines(6);
smooth_win = 3;

%% ========== 图 1：Early Exercise 各比例的定义说明 ==========
% 【关键修改】明确分母定义：
%   - ITM ratio = ITM paths / All paths (到期日 S_T < K for put)
%   - Strict early exercise ratio = Early exercised paths / All paths
%   - Exercise at maturity ratio = Exercised at maturity paths / All paths
% 注意：early_exercise_ratio_before_maturity + exercise_at_maturity_ratio ≈ early_exercise_ratio (总行权比例)
%       但可能略低，因为有些路径未 ITM 所以永远不会行权

figure('Name', 'Fig1 Early Exercise Ratios with Denominator Definitions');
subplot(1,3,1); hold on
for m = 1:6
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1'), idx = idx & (H == 0.5); else, h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end; end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = itm_ratio(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', [mod ' (ITM/All paths)'], 'MarkerSize', 4);
end
xlabel('Moneyness K/S_0'); ylabel('Ratio');
title('ITM Ratio at Maturity\n(denominator = all paths)'); legend('Location', 'best'); grid on; hold off

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
xlabel('Moneyness K/S_0'); ylabel('Ratio');
title('Strict Early Exercise Ratio (t < T)\n(denominator = all paths)'); legend('Location', 'best'); grid on; hold off

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

sgtitle(['Early Exercise Ratios (Denominator = All Paths), T = ' num2str(T_fix)]);

%% ========== 图 2：验证分母一致性 ==========
% 验证: early_exercise_ratio_before_maturity + exercise_at_maturity_ratio ≈ early_exercise_ratio

figure('Name', 'Fig2 Denominator Consistency Check');
hold on
mod = 'M2';  % 以 M2 为例
idx = strcmp(model, mod) & (T_mat == T_fix);
h_vals = unique(H(idx));
if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end
x = moneyness(idx);
y_sum = early_exercise_ratio_before_maturity(idx) + exercise_at_maturity_ratio(idx);
y_total = early_exercise_ratio(idx);
[x_s, ord] = sort(x);
plot(x_s, y_sum(ord), '-o', 'Color', [0 0.8 0], 'DisplayName', 'early + maturity (sum)', 'MarkerSize', 4);
plot(x_s, y_total(ord), '-s', 'Color', [0.8 0 0], 'DisplayName', 'total exercise ratio', 'MarkerSize', 4);
plot(x_s, y_total(ord) - y_sum(ord), '-^', 'Color', [0 0 0.8], 'DisplayName', 'difference', 'MarkerSize', 4);
xlabel('Moneyness K/S_0'); ylabel('Ratio');
title('Denominator Consistency Check (M2)\n(sum should ≈ total if denominators are same)');
legend('Location', 'best'); grid on; hold off

%% ========== 图 3：Premium 和 DeltaIV 加误差条 ==========
% 需要多次 seed 的数据才能显示误差条，这里先展示带误差带的示例结构

figure('Name', 'Fig3 American Premium with Error Bands');
subplot(1,2,1); hold on
% 假设有多个 seed 的数据，按 model/T/H/moneyness 分组计算 mean 和 std
% 这里用单次数据演示，实际应用需要多次 run
for m = 1:3
    mod = model_order{m};
    idx = strcmp(model, mod) & (T_mat == T_fix);
    if strcmp(mod, 'M0') || strcmp(mod, 'M1'), idx = idx & (H == 0.5); else, h_vals = unique(H(idx)); if ~isempty(h_vals), idx = idx & (H == h_vals(1)); end; end
    if sum(idx) == 0, continue; end
    x = moneyness(idx); y = american_premium(idx);
    [x_s, ord] = sort(x); y_s = y(ord);
    if smooth_win > 1, y_s = movmean(y_s, smooth_win, 'Endpoints', 'shrink'); end
    plot(x_s, y_s, '-o', 'Color', colors(m,:), 'DisplayName', mod, 'MarkerSize', 4);
    % 示例误差带 (实际需要多次 seed)
    % fill([x_s, fliplr(x_s)], [y_s-0.01, fliplr(y_s+0.01)], colors(m,:), 'FaceAlpha', 0.2);
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

sgtitle('American Premium (need multiple seeds for error bands)');

%% ========== 图 4：Delta IV ==========
delta_iv = iv_american_proxy - iv_european;

figure('Name', 'Fig4 Delta IV');
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
xlabel('Moneyness K/S_0'); ylabel('\Delta IV (American proxy - European)');
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
xlabel('Moneyness K/S_0'); ylabel('\Delta IV (American proxy - European)');
title(['\Delta IV (rough+jump), T = ' num2str(T_fix)]);
legend('Location', 'best'); grid on; hold off

sgtitle('\Delta IV = American IV proxy - European IV');

%% ========== 图 5：European IV Smile ==========
figure('Name', 'Fig5 European IV Smile');
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

%% ========== 图 6：Premium by Hurst ==========
figure('Name', 'Fig6 Premium by Hurst');
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

%% ========== 图 7：Summary Table 数据展示 ==========
% 按 model + H + T 计算汇总统计

summary_stats = [];
for mod = model_order
    mod_name = mod{1};
    for h = unique(H)'
        for t = unique(T_mat)'
            idx = strcmp(model, mod_name) & (H == h) & (T_mat == t);
            if sum(idx) == 0, continue; end
            summary_stats = [summary_stats; {mod_name, h, t, ...
                mean(american_price(idx)), ...
                mean(european_price(idx)), ...
                mean(american_premium(idx)), ...
                std(american_price(idx)), ...
                mean(early_exercise_ratio_before_maturity(idx)), ...
                mean(exercise_at_maturity_ratio(idx))}];
        end
    end
end
summary_table = cell2table(summary_stats, 'VariableNames', ...
    {'Model','H','T','Mean_American_Price','Mean_European_Price','Mean_Premium','Std_Price','Mean_Early_Exercise','Mean_Exercise_Maturity'});

% 显示摘要表
figure('Name', 'Fig7 Summary Statistics');
uitable('Data', table2array(summary_table(:,4:end)), ...
    'ColumnName', summary_table.Properties.VariableNames(4:end), ...
    'RowName', cellfun(@(x,y,z) [x '_H' num2str(y) '_T' num2str(z)], summary_table.Model, summary_table.H, summary_table.T, 'UniformOutput', false), ...
    'Position', [20 20 800 600]);
title('Summary Statistics Table');

%% ========== 提示：需要补充的内容 ==========
fprintf('\n========================================\n');
fprintf('V2 绘图完成。已添加：\n');
fprintf('1. Early Exercise 各比例的明确分母定义（All paths）\n');
fprintf('2. 分母一致性验证图\n');
fprintf('3. Premium 和 Delta IV 图（误差带需要多次 seed 数据）\n');
fprintf('4. European IV Smile\n');
fprintf('5. Premium by Hurst\n');
fprintf('6. Summary Table\n');
fprintf('\n还需要补充：\n');
fprintf('- 收敛实验（paths/Nt/Nexp 收敛图）需要重新跑实验\n');
fprintf('- 误差条需要多次 seed 运行才能显示\n');
fprintf('- boundary 分位数图需要补充 boundary_q25/q75 列\n');
fprintf('========================================\n');