---
title: 'PyGRASP: A library to reconstruct dynamic contrast enhanced MR images acquired with radial sampling'
tags:
  - Python
  - dynamic contrast enhanced MRI
  - radial sampling
  - GRASP
authors:
  - name: Cemre Ariyurek
    orcid: 0000-0002-1691-7097
    equal-contrib: true
    corresponding: true 
    affiliation: 1
  - name: Aziz Kocanaogullari
    orcid: 0000-0002-4776-4206
    equal-contrib: true
    affiliation: 1
  - name: Can Taylan Sari
    orcid: 0000-0001-6140-3717
    affiliation: 1
  - name:  Serge Vasylechko
    orcid: 0000-0002-5691-0607
    affiliation: 1
  - name:  Onur Afacan 
    orcid: 0000-0003-2112-3205
    affiliation: 1
  - name:  Sila Kurugol 
    orcid: 0000-0002-5081-4569
    affiliation: 1
affiliations:
 - name: Quantitative Intelligent Imaging (QUIN) Lab, Boston Children's Hospital and Harvard Medical School
   index: 1
date: 10 August 2022
bibliography: references.bib
---

# Summary

In dynamic contrast-enhanced (DCE) magnetic resonance imaging (MRI), a dynamic series of image stacks are acquired capturing the passage of contrast agent through the tissue compartments.  Dynamic radial imaging with a golden angle stack of stars sampling trajectory has been increasingly used to acquire DCE-MRI data due to its advantages compared to dynamic cartesian imaging, such as being robust to breathing motion, and ability to achieve high temporal resolution imaging. In order to reconstruct a dynamic series of volumes at high temporal resolution (~3 seconds/volume), the data needs to be reconstructed from undersampled measurements. Different regularizers have been proposed to reduce the incoherent undersampling artifacts over the dynamic image series in radial acquisition, which are in the form of streaking artifacts [@Jung:2010; @Uecker:2010; @Poddar:2016; @Prieto:2010]. Golden-angle RAdial Sparse Parallel (GRASP) [@Feng:2014] reconstruction algorithm uses temporal regularization to reduce artifacts and can reconstruct a dynamic series of volumes at the desired temporal resolution.  In this software, we have developed an efficient open-source, purely Python-based GRASP reconstruction library called pyGRASP that allows researchers to access the source code for development, facilitating flexible deployment with readable code and no compilation. This library includes preprocessing by extracting information from raw data acquired in Siemens MRI scanners and coil sensitivity map estimation, coil compression and coil removal options for improving speed and image quality and image reconstruction using a compressed sensing approach based on GRASP technique, followed by post-processing. The proposed library has options to accelerate computations by parallel processing using either GPU or CPU resources. We also provide Docker files for quick and easy deployment over any operating system or over a cloud.

# Statement of need

Dynamic contrast-enhanced (DCE) imaging is a method for acquiring a series of MR images before and after injection of a contrast agent. DCE-MRI offers a detailed anatomical evaluation by assessing the contrast uptake visually and functional evaluation by fitting a tracer kinetic model and estimating its parameters. Viewing the "wash-in" and "wash-out" of contrast on MRI may improve the detection and delineation of tumors and vascular lesions, or evaluation of the function of kidneys or liver. DCE-MRI requires rapid acquisition (with high temporal resolution, 3 sec/volume) to capture the passage of contrast in the vascular system and through the organs. An example application is functional imaging of kidneys for assessment of renal function (glomerular filtration rate). Conventional DCE-MRI with Cartesian imaging is highly sensitive to physiological motion due to breathing. It also fails to achieve high temporal resolution required to capture rapid dynamics of contrast for accurate measurement of the arterial input function peak and for accurate tracer kinetic model fitting to compute functional parameters. Accelerated imaging using parallel imaging and compressed sensing (CS) reconstruction can be used to achieve high temporal resolution from  undersampled acquisitions. CS methods take advantage of sparsity and compressibility of signals in spatial and/or time dimensions to improve image quality. A sparsity enforcing prior is used after applying a sparsifying transform to the signal. In DCE-MRI, the sparsifying transform, e.g., a total variation term is often computed in temporal dimension followed by the computation of its L1 norm for enforcing sparsity. The cost function composed of this sparsity enforcing prior and the data consistency term are jointly minimized using an iterative non-linear optimization algorithm such as conjugate gradient descent. 
Non-cartesian sampling using a radial trajectory with a stack of stars sampling scheme improves robustness to motion when acquiring free-breathing DCE-MRI of the abdomen.  In dynamic radial imaging, radial lines passing through the center of k-space are continuously acquired throughout the scan and each sampled line contains equally important information, especially the contrast information. This balanced sampling of k-space makes the acquisition motion-robust. Another advantage is that, with the use of golden angle radial ordering, it is possible to choose the temporal resolution by modifying the number of radial lines (spokes) per volume, even after acquisition of the data. Using CS reconstruction with a sparsity enforcing prior in temporal dimension, i.e., GRASP MRI reconstruction, it is possible to achieve high quality images. 
The implementation of the GRASP reconstruction in MATLAB is distributed by the authors [@Feng:2014]. However, the reconstruction of very large 4D DCE-MRI data of 3D dynamic volumes acquired over a window of 6 to 8 minutes lead to very long computation times to perform iterative reconstruction. Due to the nature of non-cartesian radial sampling, non-uniform fourier transform (NUFFT) of each volume over time needs to be computed which is computationally expensive. We developed a stand alone Python library for GRASP reconstruction with two goals, one is to achieve faster reconstruction speed using parallel processing using either CPU or GPU parallelization, and the other one is for open source distribution of the algorithm for developers and researchers, written purely in Python, which is easy to read and does not require compilation. By using the torchkbnufft library for fast NUFFT computation  with Kaiser-Bessel gridding in PyTorch [@muckley:20:tah] and by parallelization of NUFFT computation using either GPU or CPU resources over a large number of dynamic volumes, pyGRASP library speeds up the GRASP reconstruction. 
It is also possible to adjust the weight ($\lambda$) of the sparsity enforcing prior in temporal dimension according to the purpose of the reconstruction. While larger lambda values may be used to improve image quality by reducing the effect of radial undersampling artifacts in the form of streaking and to reduce the effect of noise for radiological evaluation, smaller lambda values can be used to capture fast dynamics of contrast enhancement of the arterial input function, without oversmoothing its temporal signal.This is important for accurate tracer kinetic modeling using an accurate arterial input function to estimate the perfusion and filtration rates. 
In addition to MATLAB GRASP, Berkeley Advanced Reconstruction Toolbox (BART) is an extensive, C programming based iterative image reconstruction library which includes GRASP reconstruction [@tamir:2016] but it requires installation/compilation. We believe our purely Python-based version, and Docker containerized versions will be useful to the community due to increasing availability and use of Python based tools in the MRI research community. 
pyGRASP has been used in [@AriyurekISMRMMoCo:2022], the coil selection method and a modified version of this code including motion correction were presented in ISMRM 2022 Meeting [@KocanaogullariISMRMCoil:2022; @KocanaogullariISMRMMoCo:2022]. Earlier version of this project was also employed in [@coll2020bulk; @coll2021modeling]. 

# Algorithm

pyGRASP consists of three modules. The first one reads the raw data acquired in Siemens MRI scanners, writes the k-space data into a file and calculates coil sensitivity maps based on Walsh method [@inati:2014] and the code is available in Github repository of ismrmrd-python-tools[@ismrmdpytools]. There are options for coil rejection and coil compression that can be enabled by the user. For coil rejection, the quality of information each coil carries is evaluated based on a mutual information based metric computed between a reference image (which is a combination of all data) and each coil image, and a set of coils below the threshold are removed, reducing streaking artifacts and improving the SNR of the reconstructed image [@KocanaogullariISMRMCoil:2022]. For coil compression, PCA is employed by compressing data from many channels into fewer virtual coils, with the goal of reducing the computational times. In the second module, GRASP reconstruction is computed, solving $f(X) = \| FC x_t - k_t\|_2^2 + \lambda \| TV(X)\|_{1,1}$ where $X = \lbrace x_t | t\in\lbrace 1,2,\cdots ,N \rbrace \rbrace$, $F$ is the nonuniform Fourier transform (NUFT), $C$ is the coil profile transform, $TV(.)$ is the total variation, $k_t$ and $x_t$ are the k-space data and image for a time frame t, respectively.  The first term in the loss function is for the data consistency and the second term is for the total variation of reconstructed volumes in temporal direction used as a sparsifying transform followed by computation of its L1 norm for enforcing sparsity in time. Users can tune the weight of ($\lambda$) for the regularization term (second term) and choose the number of spokes per volume to adjust temporal resolution of reconstructed volumes. In pyGRASP, we enabled parallelization over dynamic volumes as observed in flowchart of the pyGRASP (\autoref{fig:fig1}).  GPU parallelization reduces the computation time to ~3.5 mins/slice using a batch size of 10 ( GPU’s memory was 24564MiB), whereas computation time was ~ 7.3 mins/slice when it was run on CPU with 32 cores. On the other hand, computation time of MATLAB based GRASP was very long with ~99 mins/slice for a dataset of 8 mins of data including 2600 spokes, 21 slices, 38 channels, and a matrix size of 448. 

![Flowchart of pyGRASP algorithm. $numSlice$, $N_{iter}$, $N_{vol}$, are the number of reconstructed slices, NUFFT iterations and reconstructed dynamic volumes, respectively. F$ is the nonuniform Fourier transform (NUFT), $C$ is the coil profile transform, $ci$ is the channel index, $TV(.)$ is the total variation, $k_t$ and $x_t$ are the k-space data and image for a time frame t, respectively. pyGRASP enables parallelization over dynamic volumes, hence reducing computation time.\label{fig:fig1}](pyGRASPAlgo.png)


![Reconstructed dynamic volumes demonstrating aorta and kidneys for three different regularization parameters ($\lambda$), and corresponding concentration curves. When low regularization is employed, images are more noisy compared to high regularization. On the other hand, high regularization causes the second peak of arterial input function to diminish but images are less noisy with less undersampling artifacts.\label{fig:fig2}](GRASPRegularization.png)

# Acknowledgements

This work was supported partially by the National Institute of Diabetic and Digestive and Kidney Diseases (NIDDK), National Institute of Biomedical Imaging and Bioengineering (NIBIB) under award numbers R01DK125561, R21DK123569 and R21EB029627.

# References