#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:43:37 2021

@author: cogni
"""

import numpy as np

from RDS import RDS

import gc
        
# create rds
# dotDens = 0.25 # dot density
# rDot = 0.045 # dot radius in deg
# deg_per_pix = 0.02 # deg per pix

# size_rds_bg_deg = 2.5 # rds size in deg
# size_rds_ct_deg = 1.25 # center rds size in deg

# dot_refresh_rate = 42 # 
# n_block = 24
# dur_per_block = 16
# n_trial = dot_refresh_rate * n_block * dur_per_block
# n_trial = 100
n_epoch = 40
n_batch = 256
rds = RDS(n_epoch, n_batch)

# disparity tuning axis in deg
# step = 1*deg_per_pix
# disp_ct_deg = np.round(np.arange(-0.25, (0.25 + step), step), 2) 
# disp_ct_pix = General._compute_deg2pix(disp_ct_deg) # disparity tuning axis in pix
disp_ct_pix = [0, 0]



## create rds 0-deg disparity (for background)
rds_bg = rds.create_rds_bg_batch()        
np.save("../../../Data/rds/rds_bg", rds_bg)



## create rds 0deg disparity
nEpochs = 40
nBatch = 256
rdsDisp_channels = len(disp_ct_pix)
# dotMatch_list = [-1, 0, 0.5, 1]
dotMatch_list = [1]
# rds_name = ["u", "a", "hm", "c"]
for d in range(len(dotMatch_list)):
    
    dotMatch_ct = dotMatch_list[d]
    
    print("Create rds, dotMatch %s" %str(dotMatch_ct))
    
    rds_left_all = np.zeros((nEpochs*nBatch, rdsDisp_channels, 
                             size_rds_bg_pix, size_rds_bg_pix),
                            dtype=np.float32)
    rds_right_all = np.zeros((nEpochs*nBatch, rdsDisp_channels, 
                              size_rds_bg_pix, size_rds_bg_pix),
                             dtype=np.float32)

    for epoch in range(nEpochs):
        print("make rds, epoch %s" %str(epoch))
        rds_left_batch, rds_right_batch = rds.create_rds_batch(disp_ct_pix,
                                                                dotMatch_ct)
    
        id_start = epoch*nBatch
        id_end = id_start + nBatch
        rds_left_all[id_start:id_end] = rds_left_batch
        rds_right_all[id_start:id_end] = rds_right_batch
        
        # delete rds variables to save memory
        del rds_left_batch, rds_right_batch
        gc.collect()
        
        
    # suffix = rds_name[d]        
    # np.save("%srds_left_all" %(suffix), rds_left_all)
    # np.save("%srds_right_all" %(suffix), rds_right_all)
    
return rds_left_all, rds_right_all