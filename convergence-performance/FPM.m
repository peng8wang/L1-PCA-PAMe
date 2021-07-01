
function [Q, fval_collector, Q_collect, iter] = FPM(X, Q, opts)
    
    
    %% Default parameter setting
    iternum = opts.iternum;  
    tol = opts.tol; 
    if isfield(opts,'print')
        print = opts.print;
    else
        print = 0;
    end

    %% initial setting   
    fval_collector=[]; Q_collect(:,:,1) = Q;    
    X_Q = X'*Q;
    fval = sum(sum(abs(X_Q))); fval_collector(1) = fval;
    
    fprintf('********* Fixed-Point Iterations for L1-PCA *********\n'); 
    
    for iter = 1:iternum   
        
        fval_old = fval;
        
        %% fixed-point iteration
        P = sign(X_Q); X_P = X * P;
        [U, ~, V] = svd(X_P, 'econ'); 
        if print == 1
            residual = norm(Q - U*V', 'fro');
        end
        Q = U * V';  
        X_Q = X'*Q;
        
        %% collect and print the iterate information
        fval = sum(sum(abs(X'*Q))); fval_collector(iter+1) = fval; Q_collect(:,:,iter+1) = Q;
        
        if print == 1
            fprintf('Iter: %d, Proj subgrad norm: %f\n', iter, residual)
        end
        
        %% check the stopping criterion
        if abs(fval - fval_old) <= tol
            break;
        end
        
    end
end



