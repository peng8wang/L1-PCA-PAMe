
function [Q, P, fval_collect] = GS_iPAM(X, Q, P, alpha, beta, opts)
    
    %%%% implement the Guass-Seidel type inertial PALM in Gao et al. (2019) for L1-PCA %%%% 
    fprintf('Gauss-Seidel Inertial Proximal Alternating Mimization for L1-PCA \n');

    %% Default parameter setting
    iternum = opts.iternum;  
    tol = opts.tol; 
    if isfield(opts,'print')
        print = opts.print;
    else
        print = 0;
    end
    if isfield(opts, 'extra')
        extra = opts.extra;
    else
        extra = 0;
    end
    
    %% initial setting
    residu_P = []; residu_Q = []; fval_collect = [];
    fval = trace(P'*X'*Q); fval_collect(1) = fval;
    X_Q = X'*Q; E_P = P; E_Q = Q; X_E_Q = X_Q;
    
    for iter = 1:iternum
            
            fval_old = fval;
            
            %% inertial scheme
            if extra == 1
                gamma_P = 0; gamma_Q = 1/8; 
            else
                gamma_P = 0; gamma_Q = 0;
            end    
                       
            %% check Optimality for P
            if print == 1
                P1 = P + X_Q;
                P1(P1>=0) = 1; P1(P1<0) = -1;               
                P_residu = norm(P-P1); residu_P(iter) = P_residu; 
            end
            
            %% update P
            E_P_old = E_P; 
            P = alpha*E_P + X_E_Q; P(P>=0) = 1; P(P<0) = -1;  
            E_P = P + gamma_P * (P-E_P_old); 
            
            %% check optimality for Q
            if print == 1
                X_P = X*P;
                Q1 = Q + X_P; [U1, ~, V1] = svd(Q1, 'econ'); Q1 = U1*V1';  
                Q_residu = norm(Q - Q1); residu_Q(iter) = Q_residu;             
            end
            
            %% update Q
            E_Q_old = E_Q; X_E_Q_old = X_E_Q;
            [U, ~, V] = svd(beta*E_Q + X*E_P, 'econ'); Q = U * V';                     
            E_Q = Q + gamma_Q * (Q - E_Q_old); 
            X_Q = X'*Q;
            X_E_Q = X_Q + gamma_Q * (X_Q - X_E_Q_old);
            
            %% collect and print the iterate information
            fval = trace(P'*X_Q); fval_collect(iter+1) = fval; Q_collect(:,:,iter+1) = Q;
            
            if print == 1
                fprintf('Iternum: %d, Residual of P: %f, Residual of Q: %f\n', iter, P_residu, Q_residu); 
            end
            
            %% check the stopping criterion
            if abs(fval - fval_old) <= tol % Q_residu + P_residu
                break;
            end

    end
    
end



