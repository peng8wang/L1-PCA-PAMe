# Linear Convergence of Proximal Alternating Maximization Method with Extrapolation for L1-PCA

This folder contains the MATLAB source codes for the implementation of all the experiments in the paper

"Linear Convergence of Proximal Alternating Maximization Method with Extrapolation for L1-PCA" (Submitted to SIAM Journal on Optimization (SIOPT))
by Peng Wang, Huikang Liu, Anthony Man-Cho So.

* Contact: Peng Wang
* If you have any questions, please feel free to contact "wp19940121@gmail.com".

===========================================================================

This package contains 2 experimental tests to output the results in the paper:

* In the folder named convergence-performance, we conduct the experiment to investigate the convergence performance of the proposed method on synthetic and real data sets.  We also compare it with the standard PAM, the inertial PAM (iPAM) in Pock & Sabach (2016), and the Gauss-Seidel type inertial PAM (GS-iPAM) in Gao et al. (2019).
  - demo_real.m: Output the convergence performance of the tested methods on real data set news20 downloaded from LIBSVM
  - demo_synthetic: Output the convergence performance of the tested methods on synthetic data sets
  - PAMe.m: Implement the proposed proximal alternating minimization method with extrapolation (PAMe)
  - iPAM.m: Implement the inertial proximal alternating minimization method (iPAM) in Pock and Sabach (2016)
  - GS_iPAM.m: Implement the Guass-Seidel type inertial proximal alternating minimization method (GS-iPAM) in Gao et al. (2019)

* In the folder named convergence-performance, we conduct the experiments of convergence performance to test the number of iterations needed by our proposed
PPM to exactly identify the underlying communities.
  - demo_convergence.m: Output the convergence performance of our method PPM
  - dist_to_GD.m: Compute the distance from an iterate to a ground truth
