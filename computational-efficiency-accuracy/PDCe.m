
function [Q, fval_collector] = PDCe(X, Q, beta, opts)

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
    fval_collector=[];   
    fval = sum(sum(abs(X'*Q))); fval_collector(1) = fval;
       
    theta = 1; %% extrapolation stepsize
    Q_extra = Q;
    X_Q = X'*Q; subg = X*sign(X_Q); 
    fprintf('********* Proximal DC with extrapolation for L1-PCA *********\n');

    for iter = 1:iternum
        
        fval_old = fval;
        
        %% proximal DC step
        Q_old = Q; theta_old = theta;              
        [U, ~, V] = svd(Q_extra + beta*subg, 'econ'); Q = U * V';   

        %% adaptive restarting scheme
        if extra == 1
            if mod(iternum, 10) == 1 
                theta = 1; theta_old = theta;
            else
                theta = 0.5*(1+sqrt(1+4*theta^2)); 
            end     
            gamma = (theta_old - 1)/theta;
        else
            gamma = 0;
        end
        
        %% extrapolation update        
        Q_extra = Q + gamma * (Q - Q_old);
        X_Q = X'*Q;
        fval = sum(sum(abs(X_Q))); 

        %% collect and print the iterate information
        fval_collector(iter+1) = fval; 
        subg = X*sign(X_Q); 
        
        if print == 1
            %% suboptimality check        
            [U, ~, V] = svd(Q + subg, 'econ');
            residual = norm(Q - U*V', 'fro');
            residu_collector(iter) = residual;
            fprintf('Iter: %d, Proj subgrad norm: %f\n', iter, residual);
        end
        
        %% check the stopping criterion
        if abs(fval - fval_old) < tol
            break;
        end
        
    end
end



