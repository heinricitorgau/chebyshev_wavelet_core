%% DEMO_OMI_POM  第二類 Chebyshev 小波運算矩陣示範腳本
%
%   本腳本示範本套件的完整用法，共八個章節：
%
%     1. 建構 OMI 並與論文 Eq.(4.9) 逐項比對
%     2. 小波基底函數視覺化
%     3. 函數逼近與收斂階數驗證 (論文 Thm 3.2)
%     4. 乘積運算矩陣 POM 的作用
%     5. 論文 Example 1：一階線性 ODE
%     6. 變係數 ODE：POM 的實際應用
%     7. 效能與 GPU 加速
%     8. 應用：金融時間序列去噪與趨勢特徵 (wavelet_denoise_series)
%
%   於 MATLAB 編輯器中可用 Ctrl+Enter 逐節執行；亦可直接
%       >> demo_omi_pom
%   完整執行。
%
%   參考文獻：
%     H. K. Nigam and M. M. Alam, Convergence analysis of an efficient
%     Chebyshev wavelet and its applications to differential equations via
%     operational matrices of integration, Tamkang J. Math. 57(3),
%     171-193, 2026. DOI:10.5556/j.tkjm.57.2026.5958
%
%   Author : Kao, En-Tsai
%   License: MIT (see LICENSE)

clear; clc; close all;
SHOW_PLOTS = true;      % 設為 false 可在無圖形環境 (如 CI) 下執行
EXPORT_PNG = false;     % 設為 true 可將圖表輸出成 PNG (README 所用圖檔)
FIG_DIR    = 'figures'; % PNG 輸出目錄
%
% 圖表文字一律使用英文：MATLAB 預設字型不含 CJK 字元，中文標籤在
% exportgraphics 輸出的 PNG 中會顯示為方框，且其他平台未必安裝中文字型。
% (FIG_DIR 於實際輸出時才建立，見檔案末端的 export_png)

fprintf('===============================================================\n');
fprintf(' 第二類 Chebyshev 小波：OMI 與 POM 示範\n');
fprintf(' MATLAB %s\n', version);
fprintf('===============================================================\n');


%% 1. 建構 OMI 並驗證 =====================================================
% 取論文 Sec.4.1 之參數 k = 3, M = 4，矩陣階數 N = 2^(k-1)*M = 16。
fprintf('\n【1】建構 OMI (k = 3, M = 4) 並與論文 Eq.(4.9) 比對\n');

k = 3;  M = 4;
[P, ~, info] = build_chebyshev_matrices(k, M, false, [], 'Verify', true);

fprintf('  矩陣階數 N = %d，建構耗時 %.4f 秒\n', info.N, info.timeOMI);
fprintf('  與論文 Eq.(4.9) 之最大逐項差異 : %.3e\n', info.verify.paperEq49);
fprintf('  正交歸一性誤差                 : %.3e\n', info.verify.orthonormality);
fprintf('  積分精確性 (m <= M-2 之列)     : %.3e\n', info.verify.integrationExactness);

% 對角區塊 M 與離對角區塊 N (論文 Lemma 4.1)
fprintf('\n  對角區塊 M (乘上 2^k = %d 後之整數/分數形式)：\n', 2^k);
disp(info.Mblk * 2^k);
fprintf('  離對角區塊 N (僅第一行非零)：\n');
disp(info.Nblk * 2^k);

% (P 與 POM 的非零元分布圖統一於第 4 節繪製)


%% 2. 小波基底函數視覺化 ==================================================
% \psi_{n,m}(t) 於第 n 個子區間上為 m 次多項式，其餘處為 0。
fprintf('\n【2】繪製 16 個基底函數 \\psi_{n,m}(t)\n');

if SHOW_PLOTS
    tt  = linspace(0, 1, 4001);
    Psi = info.basis(tt);

    figBasis = figure('Name', 'Basis functions', 'Color', 'w', ...
        'Position', [100 100 900 640]);
    for m = 0:M-1
        subplot(M, 1, m+1); hold on; grid on; box on;
        for n = 1:info.L
            idx  = (n-1)*M + m + 1;
            mask = tt >= (n-1)/info.L & tt < n/info.L;
            plot(tt(mask), Psi(idx, mask), 'LineWidth', 1.6);
        end
        for b = 1:info.L-1                      % 子區間分界線
            xline(b/info.L, ':', 'Color', [.6 .6 .6]);
        end
        ylabel(sprintf('$m = %d$', m), 'Interpreter', 'latex', 'FontSize', 12);
        xlim([0 1]); set(gca, 'FontSize', 10);
        if m == 0
            title('Second-kind Chebyshev wavelets $\psi_{n,m}(t)$, $k=3$, $M=4$', ...
                'Interpreter', 'latex', 'FontSize', 13);
        end
    end
    xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 12);
    export_png(figBasis, 'fig_basis', EXPORT_PNG, FIG_DIR);
end
fprintf('  基底函數在各子區間 [%g, %g), ... 上分段定義\n', 0, 1/info.L);


%% 3. 函數逼近與收斂階數 ==================================================
% 論文 Thm 3.2：E_{2^{k-1},J} = O( 1 / (J! 2^{J(k+1)}) )。
% 固定 M，逐次加密解析度 k，觀察 L^2 誤差的收斂速率。
fprintf('\n【3】函數逼近收斂性 (f(t) = exp(t)sin(4t)，固定 M = %d)\n', 4);

f      = @(t) exp(t) .* sin(4*t);
tt     = linspace(0, 1, 20001).';
Mfix   = 4;
kList  = 1:7;
errL2  = zeros(numel(kList), 1);

for ii = 1:numel(kList)
    [~, ~, ik] = build_chebyshev_matrices(kList(ii), Mfix);
    Ck         = ik.expand(f);
    fApprox    = (Ck.' * ik.basis(tt)).';
    errL2(ii)  = sqrt(trapz(tt, (fApprox - f(tt)).^2));
end

rate = [NaN; log2(errL2(1:end-1) ./ errL2(2:end))];
fprintf('   k    N      L2 誤差      收斂階 (log2 比值)\n');
for ii = 1:numel(kList)
    fprintf('  %2d  %4d   %10.3e   %s\n', kList(ii), 2^(kList(ii)-1)*Mfix, ...
        errL2(ii), ternary(isnan(rate(ii)), '   --', sprintf('%7.2f', rate(ii))));
end
fprintf('  理論上每提高一階 k 應約降低 2^M = %d 倍 (M = %d)\n', 2^Mfix, Mfix);

if SHOW_PLOTS
    figConv = figure('Name', 'Convergence', 'Color', 'w', ...
        'Position', [100 100 760 560]);
    Nv = 2.^(kList-1)*Mfix;
    loglog(Nv, errL2, 'o-', 'LineWidth', 1.8, 'MarkerSize', 8, ...
        'MarkerFaceColor', 'w'); hold on; grid on; box on;
    ref = errL2(1) * (Nv/Nv(1)).^(-Mfix);       % 理論斜率 -M
    loglog(Nv, ref, '--', 'Color', [.5 .5 .5], 'LineWidth', 1.4);
    legend({'measured $L^2$ error', ...
            sprintf('reference slope $\\hat{N}^{-%d}$', Mfix)}, ...
        'Interpreter', 'latex', 'FontSize', 11, 'Location', 'southwest');
    xlabel('$\hat{N} = 2^{k-1}M$', 'Interpreter', 'latex', 'FontSize', 13);
    ylabel('$\|f - f_{\mathrm{approx}}\|_{L^2}$', 'Interpreter', 'latex', ...
        'FontSize', 13);
    title({'Convergence of the wavelet approximation', ...
           '$f(t) = e^{t}\sin(4t)$, fixed $M = 4$'}, ...
        'Interpreter', 'latex', 'FontSize', 13);
    set(gca, 'FontSize', 11);
    export_png(figConv, 'fig_convergence', EXPORT_PNG, FIG_DIR);
end


%% 4. 乘積運算矩陣 POM ====================================================
% POM 滿足 F^T \Psi(t)\Psi^T(t) = \Psi^T(t)\tilde{F}，用於處理兩函數相乘。
fprintf('\n【4】乘積運算矩陣 POM 的結構與精度\n');

kp = 4;  Mp = 6;
[~, ~, ip] = build_chebyshev_matrices(kp, Mp);
A  = ip.expand(@(t) t);                       % a(t) = t
[~, Atil, ipv] = build_chebyshev_matrices(kp, Mp, false, A, 'Verify', true);

fprintf('  k = %d, M = %d -> \\tilde{F} 為 %d 個 %dx%d 區塊組成的區塊對角矩陣\n', ...
    kp, Mp, ipv.L, Mp, Mp);
fprintf('  POM 投影一致性誤差 : %.3e\n', ipv.verify.pomProjection);
fprintf('  POM 區塊對稱性誤差 : %.3e\n', ipv.verify.pomSymmetry);

% 以 a(t)*b(t) 檢視截斷效應。關鍵在於乘積的「局部多項式次數」：
%   deg(a)+deg(b) <= M-1 -> 乘積仍在基底空間內，逼近為精確
%   deg(a)+deg(b) >  M-1 -> 高次成分被投影截斷，出現逼近誤差
tq = linspace(0, 1, 2001).';
Pq = ip.basis(tq);
fprintf('\n  a(t) = t (局部次數 1)，M = %d (可表示至 %d 次)：\n', Mp, Mp-1);
for deg = [2 5]
    B  = ip.expand(@(t) t.^deg);
    prodApprox = (Atil * B).' * Pq;            % a*b 的小波係數 = \tilde{A} B
    prodExact  = (tq.^(deg+1)).';
    tag = '截斷 -> 有誤差';
    if 1 + deg <= Mp - 1
        tag = '未截斷 -> 精確';
    end
    fprintf('    b(t) = t^%d : deg(a*b) = %d, 最大誤差 %.3e  (%s)\n', ...
        deg, deg+1, max(abs(prodApprox - prodExact)), tag);
end

if SHOW_PLOTS
    figStruct = figure('Name', 'Matrix structure', 'Color', 'w', ...
        'Position', [100 100 900 440]);
    subplot(1,2,1);
    spy(sparse(P), 8);                          % 第 1 節建構之 OMI (k=3, M=4)
    title({'OMI $P$ : block upper triangular', '$k=3$, $M=4$, $\hat{N}=16$'}, ...
        'Interpreter', 'latex', 'FontSize', 12);
    xlabel(sprintf('nnz = %d (%.1f%%)', nnz(P), 100*nnz(P)/numel(P)), ...
        'FontSize', 10);
    ylabel('row'); set(gca, 'FontSize', 10);

    subplot(1,2,2);
    spy(sparse(Atil), 6);
    title({'POM $\tilde{F}$ : block diagonal', ...
           sprintf('$k=%d$, $M=%d$, $\\hat{N}=%d$', kp, Mp, ipv.N)}, ...
        'Interpreter', 'latex', 'FontSize', 12);
    xlabel(sprintf('nnz = %d (%.1f%%)', nnz(Atil), 100*nnz(Atil)/numel(Atil)), ...
        'FontSize', 10);
    ylabel('row'); set(gca, 'FontSize', 10);
    export_png(figStruct, 'fig_structure', EXPORT_PNG, FIG_DIR);
end


%% 5. 論文 Example 1：一階線性 ODE ========================================
%   y'(t) + 2y(t) = t,  y(0) = 0,   精確解 y = t/2 - 1/4 + e^{-2t}/4
%
%   令 y'(t) = C^T\Psi(t)，兩側自 0 積分並代入 y(0)=0 與 t = E^T\Psi(t)：
%       C^T\Psi + 2C^T P\Psi = E^T P\Psi  =>  (I + 2P^T)C = P^T E
fprintf('\n【5】論文 Example 1： y'' + 2y = t,  y(0) = 0\n');

[P1, ~, i1] = build_chebyshev_matrices(3, 4);
E   = i1.expand(@(t) t);                       % 論文 Eq.(5.5)
C1  = (eye(i1.N) + 2*P1.') \ (P1.'*E);         % 論文 Eq.(5.7)
y1  = @(t) (C1.' * i1.basis(t)).';

fprintf('  c_{1,0} = %.15f\n', C1(1));
fprintf('  論文值  = 0.003866313244751 (Sec.5, Example 1)\n');

tt   = linspace(0, 1, 501).';
yExa = tt/2 - 1/4 + exp(-2*tt)/4;
fprintf('  最大絕對誤差 (k=3, M=4) : %.3e\n', max(abs(y1(tt) - yExa)));

% 提高解析度後的誤差
for kk = [4 5 6]
    [Pk, ~, ik] = build_chebyshev_matrices(kk, 4);
    Ek = ik.expand(@(t) t);
    Ck = (eye(ik.N) + 2*Pk.') \ (Pk.'*Ek);
    yk = (Ck.' * ik.basis(tt)).';
    fprintf('  最大絕對誤差 (k=%d, M=4) : %.3e\n', kk, max(abs(yk - yExa)));
end

if SHOW_PLOTS
    figEx1 = figure('Name', 'Example 1', 'Color', 'w', ...
        'Position', [100 100 820 620]);
    subplot(2,1,1);
    plot(tt, yExa, 'k-', 'LineWidth', 2.4); hold on; grid on; box on;
    plot(tt, y1(tt), '--', 'Color', [.85 .16 .16], 'LineWidth', 1.8);
    legend({'exact solution', 'wavelet solution ($k=3$, $M=4$)'}, ...
        'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northwest');
    ylabel('$y(t)$', 'Interpreter', 'latex', 'FontSize', 13);
    title(['Example 1 (paper Sec.5): $y'' + 2y = t$, $y(0)=0$, ' ...
           'exact $y = \frac{t}{2}-\frac14+\frac14e^{-2t}$'], ...
        'Interpreter', 'latex', 'FontSize', 13);
    set(gca, 'FontSize', 11);

    subplot(2,1,2); hold on; grid on; box on;
    cmap = lines(4);
    for jj = 1:4
        kk = jj + 2;                            % k = 3,4,5,6
        [Pk, ~, ik] = build_chebyshev_matrices(kk, 4);
        Ek = ik.expand(@(t) t);
        Ck = (eye(ik.N) + 2*Pk.') \ (Pk.'*Ek);
        semilogy(tt, abs((Ck.'*ik.basis(tt)).' - yExa), 'LineWidth', 1.5, ...
            'Color', cmap(jj,:), 'DisplayName', sprintf('$k = %d$', kk));
    end
    set(gca, 'YScale', 'log', 'FontSize', 11);
    legend('Interpreter', 'latex', 'FontSize', 10, 'Location', 'east', ...
        'NumColumns', 2);
    xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 13);
    ylabel('absolute error', 'FontSize', 12);
    title('Error decay with refinement ($M = 4$ fixed)', ...
        'Interpreter', 'latex', 'FontSize', 12);
    export_png(figEx1, 'fig_example1', EXPORT_PNG, FIG_DIR);
end


%% 6. 變係數 ODE：POM 的實際應用 ==========================================
%   y'(t) + t\,y(t) = t,  y(0) = 0,   精確解 y = 1 - e^{-t^2/2}
%
%   令 y' = C^T\Psi，則 y = (P^T C)^T\Psi =: D^T\Psi。變係數項用 POM 處理：
%       a(t)y(t) = (A^T\Psi)(\Psi^T D) = \Psi^T \tilde{A} D,
%   代入方程並比較係數，得線性系統
%       (I + \tilde{A}P^T)\,C = G.
fprintf('\n【6】變係數 ODE： y'' + t*y = t,  y(0) = 0  (需用到 POM)\n');

for kk = [3 5 7]
    Mv = 6;
    [Pv, ~, iv] = build_chebyshev_matrices(kk, Mv);
    Av = iv.expand(@(t) t);                    % a(t) = t
    Gv = iv.expand(@(t) t);                    % g(t) = t
    [~, Atl] = build_chebyshev_matrices(kk, Mv, false, Av);

    Cv = (eye(iv.N) + Atl*Pv.') \ Gv;          % 求解 y' 的係數
    Dv = Pv.' * Cv;                            % y 的係數 (y(0) = 0)
    yv = (Dv.' * iv.basis(tt)).';

    fprintf('  k = %d, M = %d : 最大絕對誤差 %.3e\n', kk, Mv, ...
        max(abs(yv - (1 - exp(-tt.^2/2)))));
    if kk == 7 && SHOW_PLOTS
        figVar = figure('Name', 'Variable-coefficient ODE', 'Color', 'w', ...
            'Position', [100 100 820 460]);
        plot(tt, 1 - exp(-tt.^2/2), 'k-', 'LineWidth', 2.4); hold on;
        grid on; box on;
        plot(tt, yv, '--', 'Color', [.85 .16 .16], 'LineWidth', 1.8);
        legend({'exact solution $y = 1-e^{-t^2/2}$', ...
                sprintf('wavelet solution ($k=%d$, $M=%d$)', kk, Mv)}, ...
            'Interpreter', 'latex', 'FontSize', 11, 'Location', 'northwest');
        xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 13);
        ylabel('$y(t)$', 'Interpreter', 'latex', 'FontSize', 13);
        title(sprintf(['Variable-coefficient ODE $y'' + t\\,y = t$, ' ...
                       '$y(0)=0$ (solved via POM), max error %.1e'], ...
                      max(abs(yv - (1 - exp(-tt.^2/2))))), ...
            'Interpreter', 'latex', 'FontSize', 13);
        set(gca, 'FontSize', 11);
        export_png(figVar, 'fig_varcoef', EXPORT_PNG, FIG_DIR);
    end
end
fprintf(['  註：本例 a(t) = t 的局部次數僅為 1，POM 截斷影響輕微，\n' ...
         '      故誤差可隨 k 加密迅速降至機器精度。若變係數本身為\n' ...
         '      高次多項式，截斷效應會明顯得多 (見第 4 節)。\n']);


%% 7. 效能與 GPU 加速 =====================================================
fprintf('\n【7】效能：稀疏格式與 GPU 加速\n');

% 稀疏格式：密度趨近 1/(4M)
[Ps, ~, is] = build_chebyshev_matrices(12, 8, false, [], 'Format', 'sparse');
fprintf('  sparse k=12, M=8 : N = %d, nnz = %d, 密度 = %.5f (理論 %.5f)\n', ...
    is.N, nnz(Ps), nnz(Ps)/is.N^2, 1/(4*8));
fprintf('                     記憶體 %.1f MB，建構 %.3f 秒\n', ...
    is.bytes/2^20, is.timeOMI);
clear Ps

% CPU / GPU 矩陣相乘：P * X，X 為稠密資料矩陣 (貼近實際用途)
%
%   量測方法說明：
%   * 以 timeit / gputimeit 取穩健中位數，而非單次 tic/toc。
%   * 先做 P = P + 0 強制實體化所有記憶體頁；MATLAB 的 zeros() 對未寫入
%     的頁面採惰性配置，直接量測 P*P 會得到不具代表性的結果。
%   * 即便如此，CPU 端的 BLAS 執行緒狀態在部分機器上仍會在不同 MATLAB
%     行程間波動 (本機實測可達數倍)，故「加速比」僅供參考；GPU 端的
%     吞吐量則相當穩定。請以自己硬體上的實測結果為準。
kg = 9;  Mg = 8;                               % N = 2048，控制示範耗時
hasGPU = exist('canUseGPU', 'file') == 2 && canUseGPU();

for pc = {'double', 'single'}
    p = pc{1};
    [Pc, ~, ic] = build_chebyshev_matrices(kg, Mg, false, [], 'Precision', p);
    Pc = Pc + 0;                               % 實體化所有頁面
    Nc = ic.N;
    X  = rand(Nc, Nc, p);
    Yc = Pc * X;                               % 暖機
    tCPU = timeit(@() Pc*X, 1);
    fprintf('  [%-6s N=%d] CPU : %7.4f 秒 (%5.0f GFLOPS)', ...
        p, Nc, tCPU, 2*Nc^3/tCPU/1e9);
    if hasGPU
        Pg = gpuArray(Pc);  Xg = gpuArray(X);
        Yg = Pg * Xg;  wait(gpuDevice);        % 暖機
        tGPU = gputimeit(@() Pg*Xg);
        fprintf(' | GPU : %7.4f 秒 (%5.0f GFLOPS) | 比值 %.1fx | 差異 %.1e\n', ...
            tGPU, 2*Nc^3/tGPU/1e9, tCPU/tGPU, ...
            max(max(abs(double(gather(Yg)) - double(Yc)))));
        clear Pg Xg Yg
    else
        fprintf(' | (無可用 GPU)\n');
    end
    clear Pc X Yc
end
fprintf(['  註：消費級 GeForce 顯卡的 FP64 吞吐量僅為 FP32 的 1/64，\n' ...
         '      double 精度下 GPU 未必勝過 CPU；single 精度才能發揮效能。\n']);

%% 8. 應用：金融時間序列去噪與趨勢特徵 ====================================
% 以 wavelet_denoise_series 將含噪價格投影至小波空間、濾除高頻成分，
% 並由解析微分矩陣同步取得變化率。
fprintf('\n【8】金融時間序列去噪 (wavelet_denoise_series)\n');

rng(42);
nObs    = 750;
tDays   = (1:nObs).';
truePx  = 100 + 12*sin(2*pi*tDays/250) + cumsum(0.06*randn(nObs,1));  % sine + random walk
noisyPx = truePx + 1.2*randn(nObs,1);                                 % 加入高斯雜訊

[pxSmooth, pxSlope, ~, dnDiag] = wavelet_denoise_series(noisyPx, tDays, 4, 4);

fprintf('  雜訊估計 %.3f (真實 1.200) | 殘差 std %.3f | SNR 改善 %.1f dB\n', ...
    dnDiag.sigmaHat, dnDiag.residualStd, dnDiag.snrGainDb);
fprintf('  RMSE vs 真實訊號 : 原始 %.3f -> 去噪後 %.3f\n', ...
    sqrt(mean((noisyPx - truePx).^2)), sqrt(mean((pxSmooth - truePx).^2)));
fprintf('  子區間界點最大跳躍 : %.1f%% of std(S)\n', 100*dnDiag.maxBlockJumpRel);

% 參數掃描：殘差與界點跳躍需一併考量
fprintf('   k    每區塊點數   殘差std   界點跳躍   RMSE\n');
for kk = 3:7
    [ss, ~, ~, dd] = wavelet_denoise_series(noisyPx, tDays, kk, 4);
    fprintf('  %2d   %8.1f   %7.3f   %6.1f%%   %6.3f\n', kk, dd.pointsPerBlock, ...
        dd.residualStd, 100*dd.maxBlockJumpRel, sqrt(mean((ss - truePx).^2)));
end

if SHOW_PLOTS
    figDn = figure('Name', 'Wavelet denoising', 'Color', 'w', ...
        'Position', [80 80 980 720]);
    bnd = (1:info.L-1) * nObs / info.L;      % 以 k=4 的 8 個子區間為界

    subplot(3,1,1);
    plot(tDays, noisyPx, '-', 'Color', [.72 .72 .75], 'LineWidth', .8);
    hold on; grid on; box on;
    plot(tDays, truePx, 'k-', 'LineWidth', 1.8);
    plot(tDays, pxSmooth, '-', 'Color', [.85 .16 .16], 'LineWidth', 1.8);
    xline((1:7)*nObs/8, ':', 'Color', [.6 .6 .6]);
    legend({'noisy price', 'true signal', 'wavelet denoised'}, ...
        'Location', 'northwest', 'FontSize', 10);
    ylabel('price', 'FontSize', 12);
    title('Chebyshev wavelet denoising of a simulated price series ($k=4$, $M=4$)', ...
        'Interpreter', 'latex', 'FontSize', 13);
    set(gca, 'FontSize', 10);

    subplot(3,1,2);
    plot(tDays, noisyPx - pxSmooth, '-', 'Color', [.25 .45 .85], 'LineWidth', .8);
    grid on; box on;
    xline((1:7)*nObs/8, ':', 'Color', [.6 .6 .6]);
    ylabel('residual', 'FontSize', 12);
    title(sprintf(['Residual: std = %.3f vs true noise std = 1.200 ' ...
                   '(structureless $\\Rightarrow$ good fit)'], dnDiag.residualStd), ...
        'Interpreter', 'latex', 'FontSize', 12);
    set(gca, 'FontSize', 10);

    subplot(3,1,3);
    plot(tDays, pxSlope, '-', 'Color', [.1 .5 .3], 'LineWidth', 1.5);
    hold on; grid on; box on;
    plot(tDays, 12*(2*pi/250)*cos(2*pi*tDays/250), 'k--', 'LineWidth', 1.2);
    yline(0, '-', 'Color', [.4 .4 .4]);
    xline((1:7)*nObs/8, ':', 'Color', [.6 .6 .6]);
    legend({'$dS/dt$ (wavelet)', 'derivative of the sine component'}, ...
        'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 10);
    xlabel('$t$ (trading days)', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('$dS/dt$', 'Interpreter', 'latex', 'FontSize', 12);
    title(['Derivative as a trend-strength feature ' ...
           '(note the jumps at the dotted block boundaries)'], ...
        'Interpreter', 'latex', 'FontSize', 12);
    set(gca, 'FontSize', 10);
    export_png(figDn, 'fig_denoise', EXPORT_PNG, FIG_DIR);
end
fprintf(['  註：本基底於子區間界點不連續，導數在界點（圖中虛線）會出現\n' ...
         '      可見的跳躍，此為方法固有性質，詳見 help wavelet_denoise_series。\n']);


fprintf('\n===============================================================\n');
fprintf(' 示範結束。詳細 API 說明請執行：\n');
fprintf('   help build_chebyshev_matrices\n');
fprintf('   help wavelet_denoise_series\n');
fprintf('===============================================================\n');


%% 局部工具函數 ===========================================================
function export_png(figHandle, name, doExport, figDir)
%EXPORT_PNG 將圖形輸出為 PNG (供 README 使用)
if ~doExport
    return;
end
if ~isfolder(figDir)
    mkdir(figDir);
end
fpath = fullfile(figDir, [name '.png']);
exportgraphics(figHandle, fpath, 'Resolution', 150, 'BackgroundColor', 'white');
fprintf('  已輸出圖檔 : %s\n', fpath);
end


function out = ternary(cond, a, b)
%TERNARY 三元選擇，僅供本腳本排版使用
if cond
    out = a;
else
    out = b;
end
end
