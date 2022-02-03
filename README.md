# Deep Learning for Sub-Precision CFD

Source code repository for the paper "Using Deep Neural Networks to Correct Numerical Errors in Reduced-Precision Computational Fluid Dynamics" by Akash Haridas, Nagabhushana Rao Vadlamani and Yuki Minamoto

[IITM](https://www.iitm.ac.in/), 
[Tokyo Tech](https://www.titech.ac.jp/english)

## Abstract:

Loss of information in numerical simulations can arise from various sources while solving discretized partial differential equations. In particular, precision related errors can accumulate in the quantities of interest when the simulations are performed using low-precision 16-bit floating-point arithmetic compared to an equivalent 64-bit simulation. Several machine learning (ML) techniques proposed recently have been successful in correcting the errors arising from spatial discretization. In this work, we extend these techniques to improve Computational Fluid Dynamics (CFD) simulations performed using low numerical precision. We first quantify the precision related errors accumulated in a Kolmogorov forced turbulence test case. Subsequently, we employ a Convolutional Neural Network together with a fully differentiable numerical solver performing 16-bit arithmetic to learn a tightly-coupled ML-CFD hybrid solver. Compared to the 16-bit solver, we demonstrate the efficacy of the ML-CFD hybrid solver towards reducing the error accumulation in the velocity field and improving the kinetic energy spectrum at higher frequencies.

