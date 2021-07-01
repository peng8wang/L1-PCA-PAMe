
clear all; clc;

%% Load real-world data news20 downloaded from LIBSVM
addpath('D:\MATLAB-Code\libsvm-3.24\matlab');
[~, X] = libsvmread('dataset\news20'); 
X = X'; [d, n] = size(X); K = 20;

%% choose the running algorithm
run_PAM = 1; run_PAMe = 1; run_FPM = 1; run_DC = 1; run_iPAM = 1; run_GS = 1;

%% set the parameters
numinit=1;  maxiter = 1e3; tol = 1e-8; print = 1; extra = 1; 

%% compute the K leading vectors 
[Q,S] = eigs(X'*X,K); var = sum(diag(S));

for j = 1:numinit

    %% generate initial point: P0, Q0
    F = randn(d, K); [U,S,V] = svd(F,'econ'); Q0 = U(:,1:K);
    P0 = ones(n,K).*sign(randn(n,K)); 
    
    %% set the step-size parameter
    alpha = 1e-6; beta = 2e1;

    %% Standard Proximal Alternating Mimization (PAM)
    if run_PAM == 1            
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', 0);
        tic; [Q_PA, P_PA, fval_collect_PA, Q_collect_PA] = PAMe(X, Q0, P0, alpha, beta, opts);
        time_PA = toc; optval_PA = sum(sum(abs(X'*Q_PA)));         
        fprintf('PAM: fval = %f, critical gap = %f\n', optval_PA, norm(P_PA-sign(X'*Q_PA),'fro'));
    end

    %% Proximal Alternating Mimization with extrapolation (PAMe)
    if run_PAMe == 1
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        tic; [Q_PE, P_PE, fval_collect_PE, Q_collect_PE] = PAMe(X, Q0, P0, alpha, beta, opts);
        time_PE = toc; optval_PE = sum(sum(abs(X'*Q_PE)));            
        fprintf('PAMe: fval of = %f, critical gap = %f\n', optval_PE, norm(P_PE-sign(X'*Q_PE),'fro'));
    end
    %% Fixed-Point Iteration Method (FPM)
    if run_FPM == 1
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        tic; [Q_FP, fval_collect_FP, Q_collect_FP, iter] = FPM(X, Q0, opts);
        time_FP = toc; optval_FP = sum(sum(abs(X'*Q_FP)));             
        fprintf('FPM: fval of = %f \n', optval_FP);
    end
    
    %% Proximal Difference-of-Convex Algorithm with extrapolation (pDCAe)
    if run_DC == 1
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        tic; [Q_DC, fval_collect_DC, Q_collect_DC, iter] = PDCe(X, Q0, 1e0, opts);
        time_DC = toc; optval_DC = sum(sum(abs(X'*Q_DC)));             
        fprintf('FPM: fval of = %f \n', optval_DC);
    end
        
    %% Inertial Proximal Alternating Mimization (iPAM)
    if run_iPAM == 1
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        tic; [Q_IP, P_IP, fval_collect_IP, Q_collect_IP] = iPAM(X, Q0, P0, alpha, beta, opts);
        time_IP = toc; optval_IP = sum(sum(abs(X'*Q_IP)));  
        fprintf('iPAM: fval of = %f, critical gap = %f\n', optval_IP, norm(P_IP-sign(X'*Q_IP),'fro'));
    end

    %% Gauss-Seidel Inertial Proximal Alternating Mimization (GS-iPAM)
    if run_GS == 1
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        tic; [Q_GS, P_GS, fval_collect_GS, Q_collect_GS] = GS_iPAM(X, Q0, P0, alpha, beta, opts);
        time_GS = toc; optval_GS = sum(sum(abs(X'*Q_GS)));   
        fprintf('GS-iPAM: fval = %f, critical gap = %f\n', optval_GS, norm(P_GS-sign(X'*Q_GS),'fro'));
    end

end

fprintf('Explained Variance: PAMe = %f, PAM = %f, FPM = %f, pDCAe = %f, iPAM = %f, GS-iPAM = %f \n', ...
    norm(X'*Q_PE,'fro')^2/var, norm(X'*Q_PA,'fro')^2/var, norm(X'*Q_FP,'fro')^2/var, norm(X'*Q_DC,'fro')^2/var, ...
    norm(X'*Q_IP,'fro')^2/var, norm(X'*Q_GS,'fro')^2/var);

%% plot the figures of convergence rate in terms of the iterate Qk
color1 = [0, 0.4470, 0.7410]; color2 = [0.8500, 0.3250, 0.0980];
color3 = [0.9290 0.6940 0.1250]; color4 = [0.4940 0.1840 0.5560];
color5 = [0.4660 0.6740 0.1880]; color6 = [0.6350 0.0780 0.1840];

figure(); tol = 1e-5;
if run_PAM == 1
    size_1 = size(fval_collect_PA,2); Q_dist = zeros(size_1,1);
    for i = 1:size_1
       Q_dist(i) = norm(Q_collect_PA(:,:,i) - Q_collect_PA(:,:,size_1), 'fro') + tol; 
    end
    semilogy(Q_dist, '-s', 'Color', color1, 'LineWidth', 2); hold on;
end
if run_PAMe == 1
    size_2 = size(fval_collect_PE,2); Q_dist = zeros(size_2,1);
    for i = 1:size_2
       Q_dist(i) = norm(Q_collect_PE(:,:,i) - Q_collect_PE(:,:,size_2), 'fro') + tol; 
    end
    semilogy(Q_dist, '-o', 'Color', color2, 'LineWidth', 2); hold on;
end
if run_FPM == 1
    size_5 = size(fval_collect_FP,2); Q_dist = zeros(size_5,1);
    for i = 1:size_5
       Q_dist(i) = norm(Q_collect_FP(:,:,i) - Q_collect_FP(:,:,size_5), 'fro') + tol; 
    end
    semilogy(Q_dist, '-*', 'Color', color5, 'LineWidth', 2); hold on;
end
if run_DC == 1
    size_6 = size(fval_collect_DC,2); Q_dist = zeros(size_6,1);
    for i = 1:size_6
       Q_dist(i) = norm(Q_collect_DC(:,:,i) - Q_collect_DC(:,:,size_6), 'fro') + tol; 
    end
    semilogy(Q_dist, '-<', 'Color', color6, 'LineWidth', 2); hold on;
end
if run_iPAM == 1
    size_3 = size(fval_collect_IP,2); Q_dist = zeros(size_3,1);
    for i = 1:size_3
       Q_dist(i) = norm(Q_collect_IP(:,:,i) - Q_collect_IP(:,:,size_3), 'fro') + tol; 
    end
    semilogy(Q_dist, '-d', 'Color', color3, 'LineWidth', 2); hold on;
end
if run_GS == 1
    size_4 = size(fval_collect_GS,2); Q_dist = zeros(size_4,1);
    for i = 1:size_4
       Q_dist(i) = norm(Q_collect_GS(:,:,i) - Q_collect_GS(:,:,size_4), 'fro') + tol; 
    end
    semilogy(Q_dist, '->', 'Color', color4, 'LineWidth', 2); hold on;
end

legend('PAM', 'PAMe', 'FPM', 'pDCAe', 'iPAM', 'GS-iPAM', 'FontSize', 11);
xlabel('Iterations', 'FontSize', 13); 
ylabel('$\|\mathbf{Q}^\mathbf{k}-\mathbf{Q}^\mathbf{*}\|_\mathbf{F}$', 'Interpreter', 'latex', 'FontSize', 13); 
xrange = max([size_1,size_2,size_3,size_4,size_5,size_6]);
xlim([0 xrange+15]); % ylim([1e-7 1e2]);
 
 
%% plot the figures of convergence rate in terms of function value
figure(); tol = 1e-8;
if run_PAM == 1
    minval_PA = min(-fval_collect_PA); fval_collect = -fval_collect_PA - minval_PA + tol; 
    semilogy(fval_collect, '-s', 'Color', color1, 'LineWidth', 2); hold on;
end
if run_PAMe == 1
    minval_PE = min(-fval_collect_PE); fval_collect = -fval_collect_PE - minval_PE + tol; 
    semilogy(fval_collect, '-o', 'Color', color2, 'LineWidth', 2); hold on;
end
if run_FPM == 1
    minval_FP = min(-fval_collect_FP); fval_collect = -fval_collect_FP - minval_FP + tol; 
    semilogy(fval_collect, '-*', 'Color', color5, 'LineWidth', 2); hold on;
end
if run_DC == 1
    minval_DC = min(-fval_collect_DC); fval_collect = -fval_collect_DC - minval_DC + tol; 
    semilogy(fval_collect, '-<', 'Color', color6, 'LineWidth', 2); hold on;
end
if run_iPAM == 1
    minval_IP = min(-fval_collect_IP); fval_collect = -fval_collect_IP - minval_IP + tol; 
    semilogy(fval_collect, '-d', 'Color', color3, 'LineWidth', 2); hold on;
end
if run_GS == 1
    minval_GS = min(-fval_collect_GS); fval_collect = -fval_collect_GS - minval_GS + tol; 
    semilogy(fval_collect, '->', 'Color', color4, 'LineWidth', 2); hold on;
end

legend('PAM', 'PAMe', 'FPM', 'pDCAe', 'iPAM', 'GS-iPAM', 'FontSize', 11); 
xlabel('Iterations', 'FontSize', 13);  
ylabel('$\mathbf{h}^\mathbf{k}-\mathbf{h}^\mathbf{*}$', 'Interpreter', 'latex', 'FontSize', 14); %  
xlim([0 xrange+15]);

