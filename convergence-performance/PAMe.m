
function [Q, P, fval_collect, Q_collect, iter] = PAMe(X, Q, P, alpha, beta, opts)
    
    fprintf('Proximal Alternating Maximation with Extrapolation for L1-PCA \n');

    %% default parameter setting
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
    fval = trace(P'*X'*Q); fval_collect(1) = fval; Q_collect(:,:,1) = Q;
    X_Q = X'*Q; X_E = X_Q;
    
    for iter = 1:iternum          
            
            X_Q_old = X_Q;
                        
            %% check the optimality for P
            P1 = P + X_E;
            P1(P1>=0) = 1; P1(P1<0) = -1;               
            P_residu = norm(P-P1); residu_P(iter) = P_residu; 

            %% update P
            P = alpha*P + X_E; P(P>=0) = 1; P(P<0) = -1;  
            
            %% check the optimality for Q
            X_P = X*P;
            Q1 = Q + X_P; [U1, ~, V1] = svd(Q1, 'econ'); Q1 = U1*V1';  
            Q_residu = norm(Q - Q1); residu_Q(iter) = Q_residu;          
            
            %% update Q
            [U, ~, V] = svd(beta*Q + X_P, 'econ'); Q = U * V'; 
            
            %% extrapolation scheme
            if extra == 1
                gamma = 1;
            else
                gamma = 0;
            end     
            X_Q = X'*Q; 
            X_E = (1+gamma)*X_Q - gamma*X_Q_old; 
            
            %% collect and print the iterate information
            fval = trace(X_P'*Q); fval_collect(iter+1) = fval; Q_collect(:,:,iter+1) = Q;
                        
            if print == 1
                fprintf('Iternum: %d, Residual of P: %f, Residual of Q: %f\n',  iter, P_residu, Q_residu); 
            end
            
            %% check the stopping criterion
            if Q_residu + P_residu < tol
                break;
            end

    end
    
end



