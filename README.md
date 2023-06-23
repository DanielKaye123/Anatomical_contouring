# Anatomical_contouring

Stochastic segmentation code is originally from paper / repo:

Monteiro, M., Folgoc, L., Castro, D.C., Pawlowski, N., Marques, B., Kamnitsas, K., van der Wilk, M. and Glocker, B., Stochastic Segmentation Networks: Modelling Spatially Correlated Aleatoric Uncertainty, 2020

Main changes:
- Adding normalisation for cervical cancer. Use Task7 for this (final working version). Zero meaning also works.
- Label smoothing in cross entropy
- Adds code to train the mean seperately from the covariance parameters. This is disabled by default
- Adds sampler code for temeperature scaling. Defaults to 0.001 temperature
- Fixes issue with distributions.py
