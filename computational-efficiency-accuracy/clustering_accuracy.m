
function ratio = clustering_accuracy(Y, y, n, K)

    %% applying K-means to do clustering
    e = kmeans(Y, K, 'replicates', 30);  
    e(e==2) = -1; 
    
    %% compute the clustering error
    num_misclassified = min(sum(abs(y-e)), sum(abs(y+e)))/2;
    ratio = 1 - num_misclassified/n;
    
end