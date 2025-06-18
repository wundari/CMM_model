#############
Data contents
#############

The dataset contains pre-simulated data required for producing the plots in the paper:
1. BEM_canonical: cross-correlation weights based on the binocular energy model (Supplementary Figure 4)
2. checkpoint: pretrained GC-Net
3. CMM: cross-correlation and cross-matching responses to RDSs (Figure 3)
4. epoch_7_iter_22601: GC-Net outputs (Figure 4, 5)
5. MVPA: decoding results (Figure 2, Supplementary Figure 1, 2)
6. rds: generated RDS stimuli used for simulation (general usage)
7. S2N: fMRI percent signal change (Supplementary Figure 3)
8. VTC_extract_smoothed: processed VTC data from Brainvoyager (general usage)
9. VTC_normalized: normalized VTC data (general usage)
10. VTC_stimID: stimulus parameters for the VTC data (general usage)
11. wavelet: wavelet analysis (Supplementary Figure 5)

############
Python codes
############

The corresponding python codes can be cloned from github;
https://github.com/wundari/CMM

See folder_tree_structure.txt for the arrangement of the folders.

Essentially, inside the parent-folder (the folder that contains the cloned repository CMM) has 3 main folders:
1. Codes
2. Data
3. Plots

Move the following dataset folders into the Data folder:
1. BEM_canonical
2. CMM
3. MVPA
4. rds
5. S2N
6. VTC_extract_smoothed
7. VTC_normalized
8. VTC_stimID
9. wavelet

create a directory shift_1.5_median_wrt_left:
Codes/Python/gcnet/results/sceneflow/monkaa/shift_1.5_median_wrt_left

Then, move the following dataset folders into shift_1.5_median_wrt_left directory:
1. checkpoint
2. epoch_7_iter_22601

#######################################################################
The following are the python files for reproducing figures in the paper
#######################################################################

Figure 2: Codes/Python/CMM/MVPA/run_MVPA_Decode_plot_paper.py
Figure 3: Codes/Python/CMM/CMM_Standard/run_CMM_Standard_plot_paper.py
Figure 4b right panel: Codes/Python/gcnet/RDS_analysis/run_rds_analysis_plot_paper.py
Figure 4d, 4e, 5a: Codes/Python/gcnet/Network_dissection/run_AUC_Ratio_plot_paper.py
Figure 5b: Codes/Python/gcnet/Visualization/run_feature_vis_plot_paper.py

Supplementary Figure 1; Codes/Python/CMM/MVPA/run_MVPA_Decode_fixed_vox_percentage_plot_paper.py
Supplementary Figure 2: Codes/Python/CMM/MVPA/run_MVPA_Decode_plot_paper.py
Supplementary Figure 3: Codes/Python/CMM/Signal2Noise/run_s2n_plot_paper.py
Supplementary Figure 4: Codes/Python/CMM/BEM_canonical/run_BEM_canonical_plot_paper.py
Supplementary Figure 5: Codes/Python/gcnet/wavelet/run_wavelet_analysis_plot_paper.py
