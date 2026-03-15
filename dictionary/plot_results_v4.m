%% 第四轮实验绘图：隐含波动率曲面校准 + 多方法对比
% 需要预先运行 calibrate_iv_surface.py 和 benchmark_comparison.py
% 
% 包含:
%   1. IV 曲面校准对比
%   2. 多方法价格对比
%   3. DST vs Hawkes 对比

clear; clc; close all

%% ========== 1. 读取数据 ==========
% 校准结果
calib_params_path = 'results第三轮/iv_calibration_params_latest.csv';
calib_surface_path = 'results第三轮/iv_calibration_surface_latest.csv';
market_iv_path = 'results第三轮/market_iv_synthetic.csv';

% 对比结果
benchmark_path = 'results第三轮/benchmark_comparison_latest.csv';
hawkes_path = 'results第三轮/dst_vs_hawkes_latest.csv';

%% ========== 图 1：隐含波动率曲面校准 ==========
if exist(calib_surface_path, 'file') && exist(market_iv_path, 'file')
    fprintf('读取校准数据...\n');
    
    market_iv = readtable(market_iv_path, 'ReadRowNames', true);
    calib_iv = readtable(calib_surface_path, 'ReadRowNames', true);
    
    % 转换为矩阵
    market_iv_mat = table2array(market_iv);
    calib_iv_mat = table2array(calib_iv);
    
    % 提取 T 和 K
    T_vals = str2double(extractBetween(market_iv.Properties.RowNames, 'T=', ''));
    K_vals = str2double(extractBetween(market_iv.Properties.VariableNames, 'K=', ''));
    
    [T_mesh, K_mesh] = meshgrid(T_vals, K_vals);
    
    figure('Name', 'IV Surface Calibration');
    
    % 3D 曲面
    subplot(1,2,1);
    surf(T_mesh, K_mesh, market_iv_mat');
    xlabel('Maturity T'); ylabel('K/S0'); zlabel('Implied Vol');
    title('Market IV Surface (Synthetic)');
    view(45, 30);
    colorbar;
    
    subplot(1,2,2);
    surf(T_mesh, K_mesh, calib_iv_mat');
    xlabel('Maturity T'); ylabel('K/S0'); zlabel('Implied Vol');
    title('Calibrated Model IV Surface');
    view(45, 30);
    colorbar;
    
    % 误差热力图
    figure('Name', 'IV Calibration Error');
    error_mat = (calib_iv_mat - market_iv_mat) ./ market_iv_mat * 100;
    surf(T_mesh, K_mesh, error_mat');
    xlabel('Maturity T'); ylabel('K/S0'); zlabel('Relative Error (%)');
    title('Calibration Relative Error (%)');
    colorbar;
end

%% ========== 图 2：多方法价格对比 ==========
if exist(benchmark_path, 'file')
    fprintf('读取对比数据...\n');
    
    df = readtable(benchmark_path);
    
    % 按模型和 moneyness 分组
    models = {'M0', 'M2', 'M5'};
    T_fix = 0.5;
    
    figure('Name', 'Model Price Comparison');
    hold on;
    
    for m = 1:length(models)
        mod = models{m};
        idx = (df.model == mod) & (df.T == T_fix);
        
        if sum(idx) > 0
            x = df.moneyness(idx);
            y = df.dst_price(idx);
            y_err = df.dst_ci95(idx);
            
            errorbar(x, y, y_err, '-o', 'DisplayName', mod, ...
                'MarkerSize', 6, 'LineWidth', 2);
        end
    end
    
    % 添加 Heston benchmark
    idx_heston = (df.model == 'M2') & (df.T == T_fix);
    if sum(idx_heston) > 0
        plot(df.moneyness(idx_heston), df.heston_price(idx_heston), ...
            '--k', 'DisplayName', 'Heston MC', 'LineWidth', 2);
    end
    
    xlabel('Moneyness K/S_0');
    ylabel('American Put Price');
    title(sprintf('Price Comparison (T = %.2f)', T_fix));
    legend('Location', 'best');
    grid on;
    hold off;
end

%% ========== 图 3：DST vs Hawkes 对比 ==========
if exist(hawkes_path, 'file')
    fprintf('读取 Hawkes 对比数据...\n');
    
    df_hawkes = readtable(hawkes_path);
    
    % 分离 DST 和 Hawkes
    dst_idx = df_hawkes.type == 'DST';
    hawkes_idx = df_hawkes.type == 'Hawkes';
    
    T_vals = unique(df_hawkes.T);
    T_fix = 0.5;
    
    figure('Name', 'DST vs Hawkes');
    subplot(1,2,1); hold on;
    
    % DST
    idx_dst = dst_idx & (df_hawkes.T == T_fix);
    if sum(idx_dst) > 0
        dst_data = df_hawkes(idx_dst, :);
        for i = 1:height(dst_data)
            bar(i, dst_data.price(i), 'FaceColor', [0.2 0.6 0.8], 'DisplayName', dst_data.model{i});
            hold on;
        end
    end
    
    % Hawkes
    idx_hw = hawkes_idx & (df_hawkes.T == T_fix);
    if sum(idx_hw) > 0
        hw_data = df_hawkes(idx_hw, :);
        n_dst = sum(idx_dst & (df_hawkes.T == T_fix));
        for i = 1:height(hw_data)
            bar(n_dst + i, hw_data.price(i), 'FaceColor', [0.8 0.2 0.2], 'DisplayName', hw_data.model{i});
        end
    end
    
    xlabel('Model');
    ylabel('American Put Price');
    title(sprintf('DST vs Hawkes (T = %.2f)', T_fix));
    legend('Location', 'best');
    grid on;
    hold off;
    
    % Premium 对比
    subplot(1,2,2); hold on;
    
    if sum(idx_dst) > 0
        bar(1:sum(idx_dst), dst_data.premium, 'FaceColor', [0.2 0.6 0.8]);
    end
    if sum(idx_hw) > 0
        bar(sum(idx_dst)+(1:height(hw_data)), hw_data.premium, 'FaceColor', [0.8 0.2 0.2]);
    end
    
    xlabel('Model');
    ylabel('American Premium (American - European)');
    title(sprintf('Premium Comparison (T = %.2f)', T_fix));
    grid on;
end

%% ========== 图 4：早期行权比例对比 ==========
if exist(benchmark_path, 'file')
    df = readtable(benchmark_path);
    
    T_fix = 0.5;
    models = {'M0', 'M2', 'M5'};
    
    figure('Name', 'Early Exercise Ratio');
    hold on;
    
    for m = 1:length(models)
        mod = models{m};
        idx = (df.model == mod) & (df.T == T_fix);
        
        if sum(idx) > 0
            x = df.moneyness(idx);
            y = df.early_ratio(idx);
            plot(x, y, '-o', 'DisplayName', mod, 'MarkerSize', 6, 'LineWidth', 2);
        end
    end
    
    xlabel('Moneyness K/S_0');
    ylabel('Strict Early Exercise Ratio');
    title(sprintf('Early Exercise Ratio (T = %.2f)', T_fix));
    legend('Location', 'best');
    grid on;
    hold off;
end

%% ========== 图 5：收敛性曲线 ==========
% 如果有收敛实验数据
conv_paths_files = dir('results第三轮/convergence_paths*.csv');
conv_nt_files = dir('results第三轮/convergence_nt*.csv');

if ~isempty(conv_paths_files)
    fprintf('读取路径收敛数据...\n');
    
    % 读取最新的收敛数据
    [~, idx] = sort({conv_paths_files.name}, 'descend');
    conv_path = conv_paths_files(idx(1)).name;
    df_conv = readtable(['results第三轮/' conv_path]);
    
    figure('Name', 'Convergence Paths');
    
    subplot(1,2,1);
    errorbar(df_conv.n_paths, df_conv.mean_american_price, df_conv.ci95_price, '-o');
    xlabel('Number of Paths');
    ylabel('American Put Price');
    title('Price vs Path Count');
    grid on;
    
    subplot(1,2,2);
    errorbar(df_conv.n_paths, df_conv.mean_american_premium, df_conv.ci95_premium, '-o');
    xlabel('Number of Paths');
    ylabel('American Premium');
    title('Premium vs Path Count');
    grid on;
end

if ~isempty(conv_nt_files)
    fprintf('读取时间步收敛数据...\n');
    
    [~, idx] = sort({conv_nt_files.name}, 'descend');
    conv_nt = conv_nt_files(idx(1)).name;
    df_conv_nt = readtable(['results第三轮/' conv_nt]);
    
    figure('Name', 'Convergence Time Steps');
    plot(df_conv_nt.n_steps, df_conv_nt.american_price, '-o');
    xlabel('Number of Time Steps');
    ylabel('American Put Price');
    title('Price vs Time Steps');
    grid on;
end

fprintf('\n绘图完成！\n');