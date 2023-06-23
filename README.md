# Anatomical_contouring
Final year project repo.

Stochastic segmentation code is originally from paper / repo:
https://github.com/biomedia-mira/stochastic_segmentation_networks
Monteiro, M., Folgoc, L., Castro, D.C., Pawlowski, N., Marques, B., Kamnitsas, K., van der Wilk, M. and Glocker, B., Stochastic Segmentation Networks: Modelling Spatially Correlated Aleatoric Uncertainty, 2020

Main changes:
- Adding normalisation for cervical cancer. Use Task7 for this (final working version). Zero meaning also works.
- Label smoothing in cross entropy
- Adds code to train the mean seperately from the covariance parameters. This is disabled by default
- Adds sampler code for temeperature scaling. Defaults to 0.001 temperature
- Fixes issue with distributions.py
- Adds /fixes code to plot the graphs in the report.
- Successfully trained model is included under assets folder. 



Repo also includes code for:
- Resampling
- evaluation, under evaluation/eval ->  including code for splitting femoral heads
- Code to generate the nnunet modeles. Code for setting up region based, multi class and esemble models. This is to be used with nnunet repo. https://github.com/MIC-DKFZ/nnUNet/tree/master
- Code for plotting intensities, and also showing the distribution of intensities before/ after label smoothing is applied. White box indicates patch that is drawn
- Finding min max instensities
- Also finding which regions overlap (see parse_data jupyter file)
- Other miscellaneous functions.
