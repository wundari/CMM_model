The following are the python files for reproducing figures in the paper:
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

#####################
Folder tree structure
#####################
Parent
└────── Codes
      	└── Python
      	│   ├── CMM
      	│   │   ├── CMM.py
      	│   │   ├── CMM_Standard
      	│   │   ├── Common
      	│   │   ├── DisparityColumn
      	│   │   ├── GLM
      	│   │   ├── MVPA
      	│   │   ├── RDS
      	│   │   ├── RSA
      	│   │   └── Signal2Noise
      	│   └── gcnet
      	│	├── captum
      	│	├── Common
      	│	├── data_handler
      	│	├── engine
      	│	├── Network_dissection
      	│	├── Optimization
      	│	├── python_pfm.py
      	│	├── RDS
      	│	├── RDS_analysis
      	│	├── results
      	│	│   └── sceneflow
      	│	│       └── monkaa
      	│	│           ├── shift_1.5_median_wrt_left
      	│	│           │   ├── checkpoint
      	│	│           │   ├── epoch_7_iter_22601
      	│	│           └── wavelet_power_monkaa.pt
      	│	├── SVM
      	│	├── train_gcnet_pytorch2_sceneflow.py
      	│	├── utils
      	│	├── Visualization
      	│	└── wavelet
      	│
      	Data
      	├── BEM_canonical
      	├── CMM
      	├── MVPA
      	├── rds
      	├── S2N
      	├── VTC_extract_smoothed
      	├── VTC_normalized
      	├── VTC_stimID
      	└── wavelet
