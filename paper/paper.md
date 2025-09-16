---
title: 'Mutliband MRI: A robust Cartesian SMS reconstruction toolbox'
tags:
  - Python
  - MRI
  - Simulatneous multislice imaging
  - Reconstruction
authors:
  - name: Tyler Gallun
    orcid: 0009-0008-5659-2240
    affiliation: 1
  - name: Nikolai J. Mickevicius
    orcid: 0000-0002-4169-8447
    affiliation: 1
affiliations:
 - name: Department of Biophysics, Medical College of Wisconsin, WI, USA
   index: 1
date: 16 September 2025
bibliography: paper.bib
---

# Summary
Multiband-MRI is a Python package that specializes in Cartesian reconstructions of simultaneous multislice (SMS) magnetic resonance imaging (MRI) data running on PyTorch. It offers six common reconstruction methods, slice-GRAPPA, split-slice-GRAPPA, readout SENSE-GRAPPA, slice-RAKI, split-slice-RAKI, and readout SENSE-RAKI as well as a variety of customization parameters to guide reconstructions as desired. The deep learning aspects of RAKI include multiple network types and activation types, in addition to flexible convolutional neural network training parameters. Finally, a Jupyter notebook with sample reconstructions with collected phantom data are provided for testing the various methods. With Multiband-MRI, one can use individual reconstructions as desired or experiment with finding the optimal reconstruction method for a particular use-case.

# Statement of Need
Many existing reconstruction repositories do not contain applications for simultaneous multislice (SMS) image reconstructions[@zimmermann_mrpro_2025], while some repositories only contain one or two methods for SMS reconstruction[@uecker_bart_2015] [@huang_robust_2025] [@mckibben_pygrappa_2024]. These repositories lack the flexibility to leverage strengths of slice-GRAPPA[@setsompop_blipped-controlled_2012] (SG), split-slice-GRAPPA[@cauley_interslice_2014] (SPSG), readout-SENSE-GRAPPA[@koopmans_two-dimensional-ngc-sense-grappa_2017] (ROSG), and their RAKI[@akcakaya_scan-specific_2019] counterparts.

Having access to simple, easy-to-use reconstructions will allow other scientists to explore the strengths and weaknesses of these reconstructions. Furthermore, having access to harmonious reconstructions could help drive development for SMS pulse sequences, since data from new sequence design typically has to be reconstructed offline. Additionally, since RAKI neural network implementations are trained per reconstruction, these reconstructions will work with new SMS pulse sequences without the need to retrain a neural network on a large database of images, unlike many other neural-network-based reconstructions. This repository also contains raw, fully sampled k-space data for SMS factors of 2-5 for testing and reference purposes.

# Features
## Various reconstruction methods
Three separate GRAPPA-based SMS reconstructions (SG, SPSG, ROSG) are implemented, just as described in their respective papers[@setsompop_blipped-controlled_2012] [@cauley_interslice_2014] [@koopmans_two-dimensional-ngc-sense-grappa_2017]. Robust artificial-neural-networks for k-space interpolation[@akcakaya_scan-specific_2019] (RAKI) implementations of each GRAPPA variant are also provided. RAKI, like GRAPPA, estimates the missing k-space data from in-plane acceleration or SMS imaging. Let $f_j$ be a convolutional neural network that estimates the unacquired k-space points for coil $j$, $\boldsymbol{s}_{U(k_x,k_y,j)}$. The gathered k-space points, $\boldsymbol{s}_{\boldsymbol{N}(k_x,k_y)}$, can be used as an input to the CNN to interpolate the missing data by: $\boldsymbol{s}_{U(k_x,k_y,j)} = f_j(\boldsymbol{s}_{\boldsymbol{N}(k_x,k_y)})$.

## Residual RAKI
Residual RAKI[@zhang_residual_2022] (rRAKI) is available in the toolbox. Let $G_j$ represent a linear GRAPPA term for coil $j$ and $F_j$ represent the nonlinear CNN residual error from the linear GRAPPA term at the same coil. Assuming the signal for a k-space point at a given coil, $j$, is $𝒔_{U(k_x,k_y,j)}$, then the residual RAKI reconstruction for a given channel is: $\boldsymbol{s}_{U(k_x,k_y,j)} = \lambda G_j(\boldsymbol{s}_{\boldsymbol{N}(k_x,k_y)})+(1-\lambda)F_j(\boldsymbol{s}_{\boldsymbol{N}(k_x,k_y)})$. A programmable weight parameter ($\lambda$) is available, where the presence and strength of rRAKI can be controlled.

## Flexible neural network parameters for RAKI implementation
The neural network has tunable parameters in the number of layers, neurons per layer, and number of epochs used for training, with suggested values that balance speed and quality. Additional customization arrives in the form of regularization, network selection, and activation function selection. Multiband-MRI contains both L1 and L2 (MSE) regularization, as well as a programmable linear weight to obtain a balance of L1 and L2. Both multilayer perceptron (MLP) and RESNet are available options for network types for RAKI implementations. Two different activation functions exist for use in the toolbox, complex rectified linear unit (CReLU) and complex B-spline (cbspline). A complex rectified linear unit (CReLU) is a rectified linear unit applied to the real part of complex-valued k-space and another rectified linear unit applied to the imaginary part of complex-valued k-space as follows: $CReLU = ReLU(r(x))+iReLU(i(x))$ [@el-rewaidy_deep_2020]. A complex B-spline (cbspline) activation function fits a cubic B-spline to learn the nonlinear transformations. Cbsplines show promise in improving reconstruction quality over standard CReLU activation functions [@terpstra_smooth_2025].

## Coil compression for faster image reconstruction
Due to the large memory required for high acceleration factor reconstructions in addition to hefty time requirements, a coil compression function is provided within the toolbox to reduce reconstruction time and memory with limited drawback on overall image quality. This implementation is based on the work of Huang et al[@huang_software_2008]., where principal component analysis (PCA) is utilized to reduce redundant information from individual coils to create a number of virtual coils with more independent data.

## Sample Jupyter notebook and sample data
Multiband-MRI offers a Jupyter notebook for reconstructing SMS data, with sample SMS data provided, so no additional data is required to be collected for testing or using the toolbox. This notebook contains extensive comments for ease of use and allows researchers of any skill level to understand and apply desired reconstructions. The notebook also contains instruction on how to run reconstructions for non-provided datasets.

# Acknowledgements
None reported.

# References
