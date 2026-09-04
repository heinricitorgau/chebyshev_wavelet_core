%% DEMO_OMI_POM  第二類 Chebyshev 小波運算矩陣示範腳本
%
%   本腳本示範 build_chebyshev_matrices 的完整用法，共六個章節：
%
%     1. 建構 OMI 並與論文 Eq.(4.9) 逐項比對
%     2. 小波基底函數視覺化
%     3. 函數逼近與收斂階數驗證 (論文 Thm 3.2)
%     4. 乘積運算矩陣 POM 的作用
%     5. 論文 Example 1：一階線性 ODE
%     6. 變係數 ODE：POM 的實際應用
%     7. 效能與 GPU 加速
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

% P 的區塊上三角結構：非零元分布圖
if SHOW_PLOTS
    figure('Name', 'OMI 稀疏結構', 'Color', 'w');
    spy(sparse(P)); title('OMI $P$ 的區塊上三角結構 ($k=3,\ M=4$)', ...
        'Interpreter', 'latex');
    xlabel('column'); ylabel('row');
end


%% 2. 小波基底函數視覺化 ==================================================
% \psi_{n,m}(t) 於第 n 個子區間上為 m 次多項式，其餘處為 0。
fprintf('\n【2】繪製 16 個基底函數 \\psi_{n,m}(t)\n');

if SHOW_PLOTS
    tt  = linspace(0, 1, 4001);
    Psi = info.basis(tt);

    figure('Name', '第二類 Chebyshev 小波基底', 'Color', 'w');
    for m = 0:M-1
        subplot(M, 1, m+1); hold on; grid on;
        for n = 1:info.L
            idx  = (n-1)*M + m + 1;
            mask = tt >= (n-1)/info.L & tt < n/info.L;
            plot(tt(mask), Psi(idx, mask), 'LineWidth', 1.4);
        end
        ylabel(sprintf('$m = %d$', m), 'Interpreter', 'latex');
        xlim([0 1]);
        if m == 0
            title('$\psi_{n,m}(t),\quad n = 1,\dots,4$', 'Interpreter', 'latex');
        end
    end
    xlabel('$t$', 'Interpreter', 'latex');
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
    figure('Name', '收斂性', 'Color', 'w');
    loglog(2.^(kList-1)*Mfix, errL2, 'o-', 'LineWidth', 1.6, 'MarkerFaceColor', 'w');
    grid on; xlabel('$\hat{N} = 2^{k-1}M$', 'Interpreter', 'latex');
    ylabel('$\|f - f_{\rm approx}\|_{L^2}$', 'Interpreter', 'latex');
    title('第二類 Chebyshev 小波逼近之收斂性', 'Interpreter', 'latex');
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
    figure('Name', 'POM 區塊對角結構', 'Color', 'w');
    spy(sparse(Atil));
    title(sprintf('POM $\\tilde{F}$ 之區塊對角結構 ($k=%d,\\ M=%d$)', kp, Mp), ...
        'Interpreter', 'latex');
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
    figure('Name', 'Example 1', 'Color', 'w');
    subplot(2,1,1);
    plot(tt, yExa, 'k-', 'LineWidth', 2); hold on; grid on;
    plot(tt, y1(tt), 'r--', 'LineWidth', 1.6);
    legend('精確解', '小波解 ($k=3,M=4$)', 'Interpreter', 'latex', ...
        'Location', 'northwest');
    ylabel('$y(t)$', 'Interpreter', 'latex');
    title('論文 Example 1: $y'' + 2y = t,\ y(0)=0$', 'Interpreter', 'latex');
    subplot(2,1,2);
    semilogy(tt, abs(y1(tt) - yExa), 'b-', 'LineWidth', 1.4); grid on;
    xlabel('$t$', 'Interpreter', 'latex');
    ylabel('絕對誤差');
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
        figure('Name', '變係數 ODE', 'Color', 'w');
        plot(tt, 1 - exp(-tt.^2/2), 'k-', 'LineWidth', 2); hold on; grid on;
        plot(tt, yv, 'r--', 'LineWidth', 1.6);
        legend('精確解', sprintf('小波解 ($k=%d,M=%d$)', kk, Mv), ...
            'Interpreter', 'latex', 'Location', 'northwest');
        xlabel('$t$', 'Interpreter', 'latex');
        ylabel('$y(t)$', 'Interpreter', 'latex');
        title('$y'' + t\,y = t,\ y(0)=0$（以 POM 處理變係數項）', ...
            'Interpreter', 'latex');
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

fprintf('\n===============================================================\n');
fprintf(' 示範結束。詳細 API 說明請執行： help build_chebyshev_matrices\n');
fprintf('===============================================================\n');


%% 局部工具函數 ===========================================================
function out = ternary(cond, a, b)
%TERNARY 三元選擇，僅供本腳本排版使用
if cond
    out = a;
else
    out = b;
end
end
