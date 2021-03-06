
function [Q, P, fval_collect] = iPAM(X, Q, P, alpha, beta, opts)
    
    %%%% implement the inertial PALM in Pock and Sabach (2016) for L1-PCA %%%% 
    fprintf('Inertial Proximal Alternating Mimization for L1-PCA \n');

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
    fval = trace(P'*X'*Q); fval_collect(1) = fval; Q_collect(:,:,1) = Q;
    X_Q = X'*Q; Q_old = Q; P_old = P;
    
    for iter = 1:iternum
            
            fval_old = fval;
           
            %% inertial scheme
            if extra == 1
                gamma_P = 1; gamma_Q = 1/2; 
            else
                gamma_P = 0; gamma_Q = 0;
            end    
                       
            %% check the optimality for P
            if print == 1
                P1 = P + X_Q;
                P1(P1>=0) = 1; P1(P1<0) = -1;               
                P_residu = norm(P-P1); residu_P(iter) = P_residu; 
            end
            
            %% update P
            E_P = P + gamma_P * (P-P_old); P_old = P; 
            P = alpha*E_P + X_Q; P(P>=0) = 1; P(P<0) = -1;  
            
            %% check the optimality for Q
            X_P = X*P;
            if print == 1
                Q1 = Q + X_P; [U1, ~, V1] = svd(Q1, 'econ'); Q1 = U1*V1';  
                Q_residu = norm(Q - Q1); residu_Q(iter) = Q_residu;        
            end
            
            %% update Q
            E_Q = Q + gamma_Q * (Q - Q_old); Q_old = Q;
            [U, ~, V] = svd(beta*E_Q + X_P, 'econ'); Q = U * V';         
            X_Q = X'*Q;
           
            %% collect and print the iterate information
            fval = trace(X_P'*Q); fval_collect(iter+1) = fval;
            
            if print == 1
                fprintf('Iternum: %d, Residual of P: %f, Residual of Q: %f\n',  iter, P_residu, Q_residu); 
            end
            
            %% check the stopping criterion
            if abs(fval - fval_old) <= tol % Q_residu + P_residu < tol
                break;
            end

    end
    
end



