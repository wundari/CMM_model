## 📂 Dataset Contents

This dataset contains pre-simulated data required to generate the plots presented in the accompanying paper. The folder contents are as follows:

1. **`BEM_canonical`** — Cross-correlation weights based on the Binocular Energy Model (see Supplementary Figure 4).
2. **`checkpoint`** — Pretrained GC-Net model.
3. **`CMM`** — Cross-correlation and cross-matching responses to RDSs (see Figure 3).
4. **`epoch_7_iter_22601`** — GC-Net outputs (see Figures 4 & 5).
5. **`MVPA`** — Decoding results (see Figure 2, Supplementary Figures 1 & 2).
6. **`rds`** — Generated RDS stimuli used for simulation (general use).
7. **`S2N`** — fMRI percent signal change (see Supplementary Figure 3).
8. **`VTC_extract_smoothed`** — Smoothed VTC data exported from Brainvoyager (general use).
9. **`VTC_normalized`** — Normalized VTC data (general use).
10. **`VTC_stimID`** — Stimulus parameters corresponding to the VTC data (general use).
11. **`wavelet`** — Wavelet analysis data (see Supplementary Figure 5).

---

## 🐍 Python Code Repository

Refer to `folder_tree_structure.txt` for a complete layout of the folder hierarchy.

The parent folder (which contains the cloned `CMM_model` repository) is organized into three main directories:

* `Codes/`
* `Data/`
* `Plots/`

### Data Setup

Please move the following dataset folders into the `Data/` directory:

* `BEM_canonical`
* `CMM`
* `MVPA`
* `rds`
* `S2N`
* `VTC_extract_smoothed`
* `VTC_normalized`
* `VTC_stimID`
* `wavelet`

### GC-Net Outputs

Create the following directory:

```
Codes/Python/gcnet/results/sceneflow/monkaa/shift_1.5_median_wrt_left
```

Then, move the following folders into that newly created directory:

* `checkpoint`
* `epoch_7_iter_22601`

---

## 📊 Reproducing Figures

Below is a mapping between paper figures and the corresponding Python scripts for reproduction:

### Main Figures

* **Figure 2**:
  `CMM_model/Codes/Python/CMM/MVPA/run_MVPA_Decode_plot_paper.py`

* **Figure 3**:
  `CMM_model/Codes/Python/CMM/CMM_Standard/run_CMM_Standard_plot_paper.py`

* **Figure 4b (right panel)**:
  `CMM_model/Codes/Python/gcnet/RDS_analysis/run_rds_analysis_plot_paper.py`

* **Figures 4d, 4e, 5a**:
  `CMM_model/Codes/Python/gcnet/Network_dissection/run_AUC_Ratio_plot_paper.py`

* **Figure 5b**:
  `CMM_model/Codes/Python/gcnet/Visualization/run_feature_vis_plot_paper.py`

### Supplementary Figures

* **Supplementary Figure 1**:
  `CMM_model/Codes/Python/CMM/MVPA/run_MVPA_Decode_fixed_vox_percentage_plot_paper.py`

* **Supplementary Figure 2**:
  `CMM_model/Codes/Python/CMM/MVPA/run_MVPA_Decode_plot_paper.py`

* **Supplementary Figure 3**:
  `CMM_model/Codes/Python/CMM/Signal2Noise/run_s2n_plot_paper.py`

* **Supplementary Figure 4**:
  `CMM_model/Codes/Python/CMM/BEM_canonical/run_BEM_canonical_plot_paper.py`

* **Supplementary Figure 5**:
  `CMM_model/Codes/Python/gcnet/wavelet/run_wavelet_analysis_plot_paper.py`
