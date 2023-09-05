# Deep learning for detecting MRI artifacts

This repository includes the software used to implement the analysis reported in:

Pizarro R, Assemlal HE, Boopathy S, Jubault T, Antel S, Arnold D, Shmuel A (2023) Detecting artifacts in brain MRI with deep learning and uncertainty estimation: application to large imbalanced databases. 
Journal: Medical Image Analysis.

Please cite our paper if you use the software.

## Introduction
Magnetic resonance imaging (MRI) is increasingly being used to delineate morphological changes underlying neurological disorders. Successfully detecting these changes depends on the MRI data quality. Unfortunately, image artifacts frequently compromise the MRI utility, making it critical to screen the data. Currently, quality assessment requires visual inspection, a time-consuming process that suffers from inter-rater variability. 

Here, we present the first stochastic DL algorithm to generate automated, high-performing MRI artifact detection implemented on a large and imbalanced neuroimaging database. We implemented Monte Carlo dropout in a 3D AlexNet to generate probabilities and epistemic uncertainties. We then developed a method to handle class imbalance, namely data-ramping to transfer the learning by extending the dataset size and the proportion of the artifact-free data instances. 

We used a 34,800 scans (98% clean) dataset. At baseline, we obtained 89.3% testing accuracy (F1 = 0.230). Following the transfer learning (with data-ramping), we obtained 94.9% testing accuracy (F1 = 0.357) outperforming focal cross-entropy (92.9% testing accuracy, F1 = 0.304) incorporated for comparison at handling class imbalance. By implementing epistemic uncertainties, we improved the testing accuracy to 99.5% (F1 = 0.834), outperforming the results obtained in previous comparable studies. In addition, we estimated aleatoric uncertainties by incorporating random flips to the MRI volumes and demonstrated that aleatoric uncertainty can be implemented as part of the pipeline. The methods we introduce enhance the efficiency of managing large databases and the exclusion of artifact images from big data analyses.

We made the implementation openly available here on GitHub, and developed the algorithm in [Python](https://www.python.org) with a [Tensorflow](https://www.tensorflow.org/) backend and compiled it on [Keras](https://keras.io).  Keras is a high-level software package that provides extensive flexibility to easily design and implement DL algorithms.  We empirically selected all the parameters that defined the network architecture, including the number and type of layers, the number of layer nodes, and C, the number of final possible artifacts.  

## Implementation
We have uploaded Python scripts that can be used to reproduce our results.  Unfortunately, due to data management regulations, the data cannot be released to the public. However, we expect our results to be reproducible.  

The repository is organized into two subfolders:
* The 'research' subfolder holds the code for training, validation, and testing that was utilized for the paper's publication.
* The 'production' subfolder houses the finalized inference code along with the model weights.
