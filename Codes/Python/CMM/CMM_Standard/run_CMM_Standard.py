#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:17:38 2021

    working dir: bw18_005_2/Codes/Python/CMM/CMM_Standard
    
    script for simulating cross-correlation and cross-matching with specific 
    parameters in disparity column model for each brain region

@author: cogni
"""

# %%
from CMM_Standard.CMM_Standard import RDS, RF_CMM, Simulate_CMM
from DisparityColumn.DisparityColumn import DisparityColumn as DispCol

import numpy as np
from timeit import default_timer as timer
from numba import cuda

# %%
# define rds object
# the rds only has 2 disparities: crossed and uncrossed,
# [n_trial, crossed_uncrossed, size_rds, size_rds] = [n_trial, 2, size_rds, size_rds]
n_trial_total = 510  # the number of rds images (trial) used per bootstrap
n_trial_batch = 17  # n_trial_batch, has to be multiple of 8
n_trial_epoch = np.int32(n_trial_total / n_trial_batch)

n_bootstrap = 1000  # 5

rDot = 0.045
dotDens = 0.5
size_rds_bg_deg = 2.5  # 120 pix
size_rds_ct_deg = 1.25
deg_per_pix = 0.02
n_rds = 10240  # ori
rds = RDS(n_rds, rDot, dotDens, size_rds_bg_deg, size_rds_ct_deg, deg_per_pix)

# load rds
rds_type_list = ["ards", "hmrds", "crds"]
rds_type = "ards"
rds.load_rds(rds_type)
# resize rds
new_dim = (96, 96)
rds.resize_rds(new_dim)

# generate disparity column distribution
nVox_to_analyze = 250  # the number of voxels used for the analysis
n_bootstrap_dispCol = n_bootstrap  # similar to scan run (maybe)
noise_dispCol_sigma_list = np.array(
    [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], dtype=np.float32
)
sawtooth_noise_std = noise_dispCol_sigma_list[2]
dispCol = DispCol(sawtooth_noise_std, n_bootstrap_dispCol, nVox_to_analyze)
# generate disparity column map
# [n_bootstrap, nVox, neurons_per_vox]
dispColMap_bootstrap = dispCol.create_dispColMap_vox_bootstrap()

# define rf object
f_batch = np.array([1.0, 2.0, 4.0, 8.0, 16.0]).astype(np.float32)
rf_centroid = np.array([0, 0], dtype=np.float32)

# construct square window RF bootstrap, left
dispCol_left = -dispColMap_bootstrap / 2  # disparity magnitude for RF_left
dispCol_right = dispColMap_bootstrap / 2  # disparity magnitude for RF_left
window_cmm = RF_CMM(rds, f_batch, dispCol_left, rf_centroid)
# window_bootstrap_L = window_cmm.create_window_dispCol_allBootstrap(n_bootstrap_batch)

# instantiate CMM
mtd = "euclidean"  # measured distance method for computing RDM
cmm = Simulate_CMM(
    rds,
    f_batch,
    dispColMap_bootstrap,
    nVox_to_analyze,
    n_trial_epoch,
    n_trial_batch,
    rds_type,
    mtd,
)

# %%
# generate id_rds_bootstrap
# id_rds_allBootstrap = cmm.generate_id_rds_bootstrap()

########################### cmm for ards ######################################
# the number of monocular response
n_rf = nVox_to_analyze * len(f_batch) * cmm.neurons_per_vox
n_mono = n_trial_batch * 2 * n_rf
N = n_trial_epoch * n_mono

monoResp_L_gpu = cuda.to_device(
    np.empty((n_mono, rds.size_rds_bg_pix, rds.size_rds_bg_pix), dtype=np.int8)
)
monoResp_R_gpu = cuda.to_device(
    np.empty((n_mono, rds.size_rds_bg_pix, rds.size_rds_bg_pix), dtype=np.int8)
)
binoResp_gpu = cuda.to_device(
    np.empty((n_mono, rds.size_rds_bg_pix, rds.size_rds_bg_pix), dtype=np.float32)
)
partial_complex_gpu = cuda.to_device(np.empty((n_mono, 4, 4), dtype=np.float32))
complexResp_gpu = cuda.to_device(np.empty(n_mono, dtype=np.float32))

n_rds_single_trial = nVox_to_analyze * len(f_batch) * cmm.neurons_per_vox
n_rds_batch = n_trial_batch * 2 * n_rds_single_trial

for d in range(len(noise_dispCol_sigma_list)):

    sawtooth_noise_std = noise_dispCol_sigma_list[d]

    # generate disparity column map
    dispCol = DispCol(sawtooth_noise_std, n_bootstrap_dispCol, nVox_to_analyze)
    # [n_bootstrap, nVox, neurons_per_vox]
    dispColMap_bootstrap = dispCol.create_dispColMap_vox_bootstrap()

    # construct square window RF bootstrap, left
    dispCol_left = -dispColMap_bootstrap / 2  # disparity magnitude for RF_left
    dispCol_right = dispColMap_bootstrap / 2  # disparity magnitude for RF_left
    window_cmm = RF_CMM(rds, f_batch, dispCol_left, rf_centroid)

    tik = timer()
    for r in range(len(rds_type_list)):

        corrResp = np.zeros((n_bootstrap, N), dtype=np.float32)
        matchResp = np.zeros((n_bootstrap, N), dtype=np.float32)

        # load rds
        rds_type = rds_type_list[r]
        # rds_type = "crds"
        rds.load_rds(rds_type)
        # resize rds
        rds.resize_rds(new_dim)

        dispCol_in_pix = _dispCol_in_pixel(dispCol_left)  # convert to pixel unit
        dispCol_left_vect = _vectorize_disparity(dispCol_in_pix, len(f_batch))

        dispCol_in_pix = _dispCol_in_pixel(dispCol_right)
        dispCol_right_vect = _vectorize_disparity(dispCol_in_pix, len(f_batch))

        for b in range(n_bootstrap):  # iterate over bootstrap

            # generate rf for left eye
            window_allVox_left_cpu = window_cmm.create_window_in_allVox(dispCol_left, b)

            # compute rf_area
            rf_area_cpu = _compute_rf_area(window_allVox_left_cpu)
            rf_area_gpu = cuda.to_device(rf_area_cpu)
            # rf_gpu_left = cuda.to_device(window_allVox_left_cpu)

            # generate rf for right eye
            window_allVox_right_cpu = window_cmm.create_window_in_allVox(
                dispCol_right, b
            )

            ## iterate over rds_trial
            tik = timer()
            for i in range(n_trial_epoch):

                print(
                    "generate rds: {}, epoch:{}/{}, bootstrap:{}/{}".format(
                        rds_type, i + 1, n_trial_epoch, b + 1, n_bootstrap
                    )
                )

                rds_L_CU_trial_cpu, rds_R_CU_trial_cpu = cmm.generate_rds_batch_trial(
                    rds, n_trial_batch
                )

                ########################## monoResp left ###########################
                # load rds_left
                # rds_CU_trial_gpu = cuda.to_device(rds_L_CU_trial_cpu)

                # load disparity in pixel
                # dispCol_vect_gpu = cuda.to_device(dispCol_left_vect[b])

                tx = 8  # number of threads_per_block
                ty = 8
                tz = 4
                tpb = (tx, ty, tz)  # threads per block
                bpg = (
                    np.int32(120 / tx),
                    np.int32(120 / ty),
                    np.int32(n_rf / tz),
                )  # blocks per grid

                print(
                    "compute monocular response left, rds: {}, "
                    "dispColNoise: {:.2f}, epoch:{}/{}, bootstrap:{}/{}".format(
                        rds_type,
                        sawtooth_noise_std,
                        i + 1,
                        n_trial_epoch,
                        b + 1,
                        n_bootstrap,
                    )
                )

                # tik = timer()
                # _monoResp_cuda[bpg, tpb](rds_CU_trial_gpu, rf_gpu_left,
                #                          dispCol_vect_gpu, monoResp_L_gpu)
                _monoResp_cuda[bpg, tpb](
                    cuda.to_device(rds_L_CU_trial_cpu),
                    cuda.to_device(window_allVox_left_cpu),
                    cuda.to_device(dispCol_left_vect[b]),
                    monoResp_L_gpu,
                )
                cuda.synchronize()
                # tok = timer()
                # print(tok - tik)

                # %time monoResp_cpu = monoResp_L_gpu.copy_to_host()
                # monoResp_cpu2 = monoResp_check(rds_L_CU_trial_cpu,
                #                                window_allVox_left_cpu,
                #                                dispCol_left_vect[b])
                # monoResp_cpu3 = monoResp_check_without_shift(rds_L_CU_trial_cpu,
                #                                              window_allVox_left_cpu)
                # # np.allclose(monoResp_cpu, monoResp_cpu2)

                # i=16
                # plt.imshow(monoResp_cpu[i])
                # plt.imshow(monoResp_cpu2[i])
                # plt.imshow(monoResp_cpu3[i])
                # np.allclose(monoResp_cpu[i], monoResp_cpu2[i])

                # plt.imshow(rds_L_CU_trial_cpu[0])
                # plt.imshow(window_allVox_left_cpu[i])

                ########################### monoResp right #########################
                # load rds_right
                # rds_CU_trial_gpu = cuda.to_device(rds_R_CU_trial_cpu)

                # load disparity in pixel
                # dispCol_vect_gpu = cuda.to_device(dispCol_right_vect[b])

                print(
                    "compute monocular response right, rds: {}, "
                    "dispColNoise: {:.2f}, epoch:{}/{}, bootstrap:{}/{}".format(
                        rds_type,
                        sawtooth_noise_std,
                        i + 1,
                        n_trial_epoch,
                        b + 1,
                        n_bootstrap,
                    )
                )
                # monoResp_R_gpu = cuda.to_device(np.zeros((n_mono, 120, 120), dtype=np.int8))
                # _monoResp_cuda[bpg, tpb](rds_CU_trial_gpu, rf_gpu_right,
                #                          dispCol_vect_gpu, monoResp_R_gpu)
                _monoResp_cuda[bpg, tpb](
                    cuda.to_device(rds_R_CU_trial_cpu),
                    cuda.to_device(window_allVox_right_cpu),
                    cuda.to_device(dispCol_right_vect[b]),
                    monoResp_R_gpu,
                )
                cuda.synchronize()

                # %time monoResp_cpu = monoResp_R_gpu.copy_to_host()
                # monoResp_cpu2 = monoResp_check(rds_R_CU_trial_cpu,
                #                                window_allVox_right_cpu,
                #                                dispCol_right_vect[b])
                # monoResp_cpu3 = monoResp_check_without_shift(rds_R_CU_trial_cpu,
                #                                              window_allVox_right_cpu)
                # np.allclose(monoResp_cpu, monoResp_cpu2)

                # i=55
                # plt.imshow(monoResp_cpu[i])
                # plt.imshow(monoResp_cpu2[i])
                # plt.imshow(monoResp_cpu3[i])

                # plt.imshow(rds_R_CU_trial_cpu[0])
                # plt.imshow(window_allVox_right_cpu[i])

                ############################## crossCorrResp ###################################
                print(
                    "compute binocular response for cross-correlation, rds: {}, "
                    "dispColNoise: {:.2f}, epoch:{}/{}, bootstrap:{}/{}".format(
                        rds_type,
                        sawtooth_noise_std,
                        i + 1,
                        n_trial_epoch,
                        b + 1,
                        n_bootstrap,
                    )
                )
                # binoResp_gpu = cuda.to_device(np.zeros((n_mono, 120, 120), dtype=np.float32))
                _binoResp_corr_cuda[bpg, tpb](
                    monoResp_L_gpu, monoResp_R_gpu, binoResp_gpu
                )
                cuda.synchronize()

                tx = 32
                ty = 32
                tz = 1
                tpb = (tx, ty, tz)
                bpg = (4, 4, np.int32(n_rf / tz))

                print(
                    "compute cross-correlation response, rds: {}, "
                    "dispColNoise: {:.2f}, epoch:{}/{}, bootstrap:{}/{}".format(
                        rds_type,
                        sawtooth_noise_std,
                        i + 1,
                        n_trial_epoch,
                        b + 1,
                        n_bootstrap,
                    )
                )
                # partial_gpu = cuda.to_device(np.zeros((n_mono, 4, 4), dtype=np.float32))
                _corr_partial_cuda[bpg, tpb](binoResp_gpu, partial_complex_gpu)
                cuda.synchronize()

                tx = 4
                ty = 4
                tpb = (tx, ty, tz)
                bpg = (1, 1, np.int32(n_rf / tz))

                # %time crossCorrResp_gpu = cuda.to_device(np.zeros(n_mono, dtype=np.float32))
                _corr_final_cuda[bpg, tpb](
                    partial_complex_gpu, rf_area_gpu, complexResp_gpu
                )
                cuda.synchronize()

                # copy to cpu
                print("copy to cpu")
                id_start = i * n_rds_batch
                id_end = id_start + n_rds_batch
                corrResp[b, id_start:id_end] = complexResp_gpu.copy_to_host()

                ############################### crossMatchResp #############################
                print(
                    "compute binocular response for cross-matching, rds: {}, "
                    "dispColNoise: {:.2f}, epoch:{}/{}, bootstrap:{}/{}".format(
                        rds_type,
                        sawtooth_noise_std,
                        i + 1,
                        n_trial_epoch,
                        b + 1,
                        n_bootstrap,
                    )
                )

                tx = 8  # number of threads_per_block
                ty = 8
                tz = 4
                tpb = (tx, ty, tz)  # threads per block
                bpg = (
                    np.int32(120 / tx),
                    np.int32(120 / ty),
                    np.int32(n_rf / tz),
                )  # blocks per grid
                # binoResp_gpu = cuda.to_device(np.zeros((n_mono, 120, 120), dtype=np.float32))
                _binoResp_match_cuda[bpg, tpb](
                    monoResp_L_gpu, monoResp_R_gpu, binoResp_gpu
                )
                cuda.synchronize()

                tx = 32
                ty = 32
                tz = 1
                tpb = (tx, ty, tz)
                bpg = (4, 4, np.int32(n_rf / tz))

                print(
                    "compute cross-matching response, rds: {}, "
                    "dispColNoise: {:.2f}, epoch:{}/{}, bootstrap:{}/{}".format(
                        rds_type,
                        sawtooth_noise_std,
                        i + 1,
                        n_trial_epoch,
                        b + 1,
                        n_bootstrap,
                    )
                )
                # partial_gpu = cuda.to_device(np.zeros((n_mono, 4, 4), dtype=np.float32))
                _match_partial_cuda[bpg, tpb](binoResp_gpu, partial_complex_gpu)
                cuda.synchronize()

                tx = 4
                ty = 4
                tpb = (tx, ty, tz)
                bpg = (1, 1, np.int32(n_rf / tz))

                # %time crossCorrResp_gpu = cuda.to_device(np.zeros(n_mono, dtype=np.float32))
                _match_final_cuda[bpg, tpb](
                    partial_complex_gpu, rf_area_gpu, complexResp_gpu
                )
                cuda.synchronize()

                # copy to cpu
                print("copy to cpu")
                id_start = i * n_rds_batch
                id_end = id_start + n_rds_batch
                matchResp[b, id_start:id_end] = complexResp_gpu.copy_to_host()

            tok = timer()
            print(tok - tik)

        # save data
        # np.save("../../../Data/CMM/corrResp_{}_dispColNoise_{:.2f}"
        #         .format(rds_type, np.round(sawtooth_noise_std, 2)),
        #         corrResp)
        # np.save("../../../Data/CMM/matchResp_{}_dispColNoise_{:.2f}"
        #         .format(rds_type, np.round(sawtooth_noise_std, 2)),
        #         matchResp)
        # np.save("../../../../../../../../media/wundari/Evo4TB/CMM_backup/corrResp_{}_dispColNoise_{:.2f}"
        #         .format(rds_type, np.round(sawtooth_noise_std, 2)),
        #         corrResp)
        # np.save("../../../../../../../../media/wundari/Evo4TB/CMM_backup/matchResp_{}_dispColNoise_{:.2f}"
        #         .format(rds_type, np.round(sawtooth_noise_std, 2)),
        #         matchResp)

    tok = timer()
    print(tok - tik)



# %%
sbjID_mt = []
for sbj in range(len(plot_cmm.sbjID_with_MTlocalizer)):
    sbjID = plot_cmm.sbjID_with_MTlocalizer[sbj]
    idx = plot_cmm.sbjID_all.index(sbjID)
    sbjID_mt.append(idx)
rdm_fmri_mtloc = plot_cmm.rdm_fmri_all[sbjID_mt]  # [sbjID, roi, 6, 6]
rdm_fmri_mtloc_mean = np.mean(rdm_fmri_mtloc, axis=0)

rdm_fmri_mean = np.mean(plot_cmm.rdm_fmri_all, axis=0)
# %%
