
addpath('D:\MATLAB-Code\libsvm-3.24\matlab');
clear all; clc;

%% set step-size parameter on each data set
% a9a: alpha = 1e-8, beta = 1e-1
% colon-cancer: alpha = 1e-6, beta = 1
% gisette: alpha = 1e-6, beta = 1
% ijcnn1: alpha = 1e-10, beta = 1 
% rcv1.binary: alpha = 1e-10, beta = 1e1
% real-sim: alpha = 1e-10, beta = 1
% w8a: alpha = 1e-8, beta = 1e-1

%% load real-world data set
[y, X] = libsvmread('datasets\w8a'); 
X = X'; [d, n] = size(X); 

%% choose the dimension of subspace by the explained variance of PCA
p = min(n,d); 
if p < 10000
    [U,S,V] = svds(X,p); s = diag(S);
    for k = 1:p
        if sqrt(norm(s(1:k))^2/norm(s)^2) >= 0.8
            break;
        end
    end
    K = k;
else
    K = 50;
end

%% choose the running algorithm
run_PD = 1; run_PAMe = 1; run_PAM = 1; run_FP = 1; run_IP = 1; run_GS = 1;

%% set the parameters 
num_repeat = 10; maxiter = 1e3; extra = 1; print = 0; tol = 1e-6; 

%% set the step-size parameters
alpha = 1e-8; beta = 1e-1;

for j = 1:num_repeat
    
    fprintf('Number of test: %d \n', j);
    F = randn(d,K); [Q0,~,~] = svd(F,'econ'); P0 = sign(randn(n,K));

    %% proximal DC with extrapolation (PDCe)
    if run_PD == 1 
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        tic; [Q_PD, fval_collect_PD] = PDCe(X, Q0, beta, opts); 
        time_PD(j) = toc; fval_PD(j) = sum(sum(abs(X'*Q_PD)));
        accuracy_PD(j) = clustering_accuracy(X'*Q_PD, y, n, 2);
        fprintf('PDCe: accuracy = %f, time = %f, fval = %f\n', accuracy_PD(j), time_PD(j), fval_PD(j));
    end

    %% Fixed Point Iteration Method (FPM)
    if run_FP == 1 
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print);
        tic; [Q_FP, fval_collect_FP] = FPM(X, Q0, opts);
        time_FP(j) = toc; fval_FP(j) = sum(sum(abs(X'*Q_FP)));
        accuracy_FP(j) = clustering_accuracy(X'*Q_FP, y, n, 2);
        fprintf('FPM: accuracy = %f, time = %f, fval = %f\n', accuracy_FP(j), time_FP(j), fval_FP(j));
    end
    
    %% Proximal Alternating Minimization with extrapolation (PAMe)
    if run_PAMe == 1
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        tic; [Q_PE, P_PE, fval_collect_PE] = PAMe(X, Q0, P0, alpha, beta, opts);
        time_PE(j) = toc; fval_PE(j) = sum(sum(abs(X'*Q_PE)));      
        accuracy_PE(j) = clustering_accuracy(X'*Q_PE, y, n, 2);
        fprintf('PAMe: accuracy = %f, critical gap = %f, time = %f, fval = %f\n',...
            accuracy_PE(j), norm(P_PE-sign(X'*Q_PE),'fro'), time_PE(j), fval_PE(j));
    end
    
    %% Proximal Alternating Minimization (PAM)
    if run_PAM == 1
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', 0);
        tic; [Q_PA, P_PA, fval_collect_PA] = PAMe(X, Q0, P0, alpha, beta, opts);
        time_PA(j) = toc; fval_PA(j) = sum(sum(abs(X'*Q_PA)));  
        accuracy_PA(j) = clustering_accuracy(X'*Q_PA, y, n, 2);
        fprintf('PAM: accuracy = %f, critical gap = %f, time = %f, fval = %f\n',...
            accuracy_PA(j), norm(P_PA-sign(X'*Q_PA),'fro'), time_PA(j), fval_PA(j));
    end
    
    %% Inertial Proximal Alternating Linearized Mimization (iPALM)
    if run_IP == 1
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        tic; [Q_IP, P_IP, fval_collect_IP] = iPALM(X, Q0, P0, alpha, beta, opts);
        time_IP(j) = toc; fval_IP(j) = sum(sum(abs(X'*Q_IP)));   
        accuracy_IP(j) = clustering_accuracy(X'*Q_IP, y, n, 2);
        fprintf('iPALM: accuracy = %f, critical gap = %f, time = %f, fval = %f\n',...
            accuracy_IP(j), norm(P_IP-sign(X'*Q_IP),'fro'), time_IP(j), fval_IP(j));
    end

    %% Gauss-Seidel Inertial Proximal Alternating Linearized Mimization (GiPALM)
    if run_GS == 1
        opts = struct('iternum', maxiter, 'tol', tol, 'print', print, 'extra', extra);
        tic; [Q_GS, P_GS, fval_collect_GS] = GiPALM(X, Q0, P0, alpha, beta, opts);
        time_GS(j) = toc; fval_GS(j) = sum(sum(abs(X'*Q_GS)));   
        accuracy_GS(j) = clustering_accuracy(X'*Q_GS, y, n, 2);
        fprintf('GiPALM: accuracy = %f, critical gap = %f, time = %f, fval = %f\n',...
            accuracy_GS(j), norm(P_GS-sign(X'*Q_GS),'fro'), time_GS(j), fval_GS(j));
    end
    
end

%% record the information
fprintf('********** average accuracy and time of each method ********** \n')
if run_PAMe == 1 
    ave_accuracy_PE = sum(accuracy_PE) / num_repeat;
    ave_time_PE = sum(time_PE) / num_repeat;
    fprintf('PAMe: accuracy = %f, time = %f\n', ave_accuracy_PE, ave_time_PE);
end
if run_PAM == 1 
    ave_accuracy_PA = sum(accuracy_PA) / num_repeat;
    ave_time_PA = sum(time_PA) / num_repeat;
    fprintf('PAM: accuracy = %f, time = %f\n', ave_accuracy_PA, ave_time_PA);
end
if run_FP == 1 
    ave_accuracy_FP = sum(accuracy_FP) / num_repeat;
    ave_time_FP = sum(time_FP) / num_repeat;
    fprintf('FPM: accuracy = %f, time = %f\n', ave_accuracy_FP, ave_time_FP);
end
if run_PD == 1 
    ave_accuracy_PD = sum(accuracy_PD) / num_repeat;
    ave_time_PD = sum(time_PD) / num_repeat;
    fprintf('PDCe: accuracy = %f, time = %f\n', ave_accuracy_PD, ave_time_PD);
end
if run_IP == 1 
    ave_accuracy_IP = sum(accuracy_IP) / num_repeat;
    ave_time_IP = sum(time_IP) / num_repeat;
    fprintf('iPALM: accuracy = %f, time = %f\n', ave_accuracy_IP, ave_time_IP);
end
if run_GS == 1 
    ave_accuracy_GS = sum(accuracy_GS) / num_repeat;
    ave_time_GS = sum(time_GS) / num_repeat;
    fprintf('GiPALM: accuracy = %f, time = %f\n', ave_accuracy_GS, ave_time_GS);
end
