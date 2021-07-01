# Linear Convergence of Proximal Alternating Maximization Method with Extrapolation for L1-PCA

This folder contains the MATLAB source codes for the implementation of all the experiments in the paper

"Linear Convergence of Proximal Alternating Maximization Method with Extrapolation for L1-PCA" (Submitted to SIAM Journal on Optimization (SIOPT))
by Peng Wang, Huikang Liu, Anthony Man-Cho So.

* Contact: Peng Wang
* If you have any questions, please feel free to contact "wp19940121@gmail.com".

===========================================================================

In the experimens, we report convergence performance and numerical efficiency and accuracy of the proposed method for solving L1-PCA on both synthetic and real data sets. We also compare it with fiving existing ones, which are the standard proximal alternating minimization method (PAM),  the fixed-point iteration method (FPM) in Nie et al. (2011), and the proximal difference-of-convex algorithm with extrapolation (pDCAe) in Wen et al. (2018), the inertial PAM (iPAM) in Pock & Sabach (2016), and the Gauss-Seidel type inertial PAM (GS-iPAM) in Gao et al. (2019). This package contains 2 experimental tests to output the results in the paper:

* In the folder named convergence-performance, we conduct the experiment to investigate the convergence performance of the proposed proximal alternating minimization method with extrapolation (PAMe) on synthetic and real data sets.  
  - demo_real.m: Output the convergence performance of the tested methods on real data set news20 downloaded from LIBSVM
  - demo_synthetic: Output the convergence performance of the tested methods on synthetic data sets
  - PAMe.m: Implement the proposed PAMe
  - FPM.m: Implement FPM in Nie et al. (2011)
  - PDCe.m: Implement PDCe in Wen et al. (2018) 
  - iPAM.m: Implement iPAM in Pock and Sabach (2016)
  - GS_iPAM.m: Implement GS-iPAM in Gao et al. (2019)
  - laprnd.m: Generate i.i.d. laplacian random number drawn from the laplacian distribution with mean mu and standard deviation sigma. 

* In the folder named computational-efficiency-accuracy, we apply L1-PCA to do clustering to investigate the computational efficiency and clustering accuracy of the proposed method. 
  - demo.m: Output the CPU time and ratios of correctly clustered points of the tested methods
  - clustering_error.m: Compute the clustering error of the returned solution of each tested method over the ground truth
 
