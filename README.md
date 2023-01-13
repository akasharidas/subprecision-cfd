# Deep Learning for Sub-Precision CFD

Source code repository for the paper ["Deep neural networks to correct sub-precision errors in CFD"](https://www.sciencedirect.com/science/article/pii/S2666352X22000243) by Akash Haridas, Nagabhushana Rao Vadlamani and Yuki Minamoto, appearing in the journal Applications in Energy and Combustion Science.

[IIT Madras](https://www.iitm.ac.in/), 
[Tokyo Institute of Technology](https://www.titech.ac.jp/english)

## Abstract:

Information loss in numerical physics simulations can arise from various sources when solving discretised partial differential equations. In particular, errors related to numerical precision (“sub-precision errors”) can accumulate in the quantities of interest when the simulations are performed using low-precision 16-bit floating-point arithmetic compared to an equivalent 64-bit simulation. On the other hand, low-precision computation is less resource intensive than high-precision computation. Several machine learning techniques proposed recently have been successful in correcting errors due to coarse spatial discretisation. In this work, we extend these techniques to improve CFD simulations performed with low numerical precision. We quantify the precision-related errors accumulated in a Kolmogorov forced turbulence test case. Subsequently, we employ a Convolutional Neural Network together with a fully differentiable numerical solver performing 16-bit arithmetic to learn a tightly-coupled ML-CFD hybrid solver. Compared to the 16-bit solver, we demonstrate the efficacy of the hybrid solver towards improving various metrics pertaining to the statistical and pointwise accuracy of the simulation.
