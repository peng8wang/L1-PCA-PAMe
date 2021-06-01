
clear; clc;

%% generate the synthetic data using the l1-fixed effect model
n = 4000; %%% n = number of samples
d = 2000; %%% d = number of features
K = 50;   %%% K = dimension of subspace
X = randn(d,n) + laprnd(d,n,0,0.5); 

%% choose the running algorithm
run_PAM = 1; run_PAMe = 1; run_iPAM = 1; run_GS = 1;

%% set the parameters
numinit = 1;  maxiter = 1e3; tol = 1e-6; print = 1; extra = 1; 

for j = 1:numinit
    
    %% generate initial point: P0, Q0
    F = randn(d, K); [U,S,V] = svd(F,'econ'); Q0 = U(:,1:K);
    P0 = ones(n,K).*sign(randn(n,K)); 
    
    %% set the step-size parameter
    alpha = 1e-5; beta = 1e3;

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
        tic; [Q_PE, P_PE, fval_collect_PE, Q_collect_PE, iter] = PAMe(X, Q0, P0, alpha, beta, opts);
        time_PE = toc; optval_PE = sum(sum(abs(X'*Q_PE)));             
        fprintf('PAMe: fval of = %f, critical gap = %f\n', optval_PE, norm(P_PE-sign(X'*Q_PE),'fro'));
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

%% plot the figures of convergence rate in terms of the iterate Qk       
color1 = [0, 0.4470, 0.7410]; color2 = [0.8500, 0.3250, 0.0980];
color3 = [0.9290 0.6940 0.1250]; color4 = [0.4940 0.1840 0.5560];

figure();
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

legend('PAM', 'PAMe', 'iPAM','GS-iPAM', 'FontSize', 11);
xlabel('Iterations', 'FontSize', 13); 
ylabel('$\|\mathbf{Q}^\mathbf{k}-\mathbf{Q}^\mathbf{*}\|_\mathbf{F}$', 'Interpreter', 'latex', 'FontSize', 13); % , 
xrange = max([size_1,size_2,size_3,size_4]);
xlim([0 xrange+5]);


%% plot the figures of convergence rate in terms of function value
figure(); tol = 1e-10;
if run_PAM == 1
    minval_PA = min(-fval_collect_PA); fval_collect = -fval_collect_PA - minval_PA + tol; 
    semilogy(fval_collect, '-s', 'Color', color1, 'LineWidth', 2); hold on;
end
if run_PAMe == 1
    minval_PE = min(-fval_collect_PE); fval_collect = -fval_collect_PE - minval_PE + tol; 
    semilogy(fval_collect, '-o', 'Color', color2, 'LineWidth', 2); hold on;
end
if run_iPAM == 1
    minval_IP = min(-fval_collect_IP); fval_collect = -fval_collect_IP - minval_IP + tol; 
    semilogy(fval_collect, '-d', 'Color', color3, 'LineWidth', 2); hold on;
end
if run_GS == 1
    minval_GS = min(-fval_collect_GS); fval_collect = -fval_collect_GS - minval_GS + tol; 
    semilogy(fval_collect, '->', 'Color', color4, 'LineWidth', 2); hold on;
end

legend('PAM', 'PAMe', 'iPAM', 'GS-iPAM', 'FontSize', 11); 
xlabel('Iterations', 'FontSize', 13);  
ylabel('$\mathbf{f}^\mathbf{k}-\mathbf{f}^\mathbf{*}$', 'Interpreter', 'latex', 'FontSize', 14); %  
xrange = max([size_1,size_2,size_3,size_4]);
xlim([0 xrange+5]);


