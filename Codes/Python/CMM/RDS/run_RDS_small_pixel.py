#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 23:12:39 2021

    cd /NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM
    
    script for creating rds for simulating disparity computation with 
    cross-correlation and cross-matching computation.
    
    The rds here is used for generating disparity tuning function.

@author: cogni
"""

################### RDS for generating disparity tuning #######################
import numpy as np
from RDS.RDS import RDS

## create rds 0deg disparity
n_rds = 1000

rDot = 0.045
dotDens = 0.50
size_rds_bg_deg = 2.5
size_rds_ct_deg = 1.25
deg_per_pix = 0.02

rds = RDS(n_rds,
          rDot, dotDens, 
          size_rds_bg_deg, 
          size_rds_ct_deg,
          deg_per_pix)


disp_ct_deg = np.linspace(-0.2, 0.2, 21).astype(np.float32)
disp_ct_pix = rds._compute_deg2pix(disp_ct_deg)

rdsDisp_channels = len(disp_ct_pix)
dotMatch_list = [-1, 0, 0.5, 1]
# dotMatch_list = [1]
rds_name = ["u", "a", "hm", "c"]
for d in range(len(dotMatch_list)):
    
    dotMatch_ct = dotMatch_list[d]
        
    print("Create rds, dotMatch %s" %str(dotMatch_ct))
    
    # [n_trials, 2, size_rds_bg, size_rds_bg]
    rds_left_batch, rds_right_batch = rds.create_rds_batch(disp_ct_pix,
                                                           dotMatch_ct)
        
    suffix = rds_name[d]        
    np.save("../../../Data/rds_small_pixel/disp_tuning/rds_left_%s" %(suffix), rds_left_batch)
    np.save("../../../Data/rds_small_pixel/disp_tuning/rds_right_%s" %(suffix), rds_right_batch)
    
    
###############################################################################
################### RDS for generating multivariate analysis ##################
import numpy as np
from RDS.RDS import RDS

## create rds 0deg disparity
n_rds = 1000

rDot = 0.045
dotDens = 0.50
size_rds_bg_deg = 2.5
size_rds_ct_deg = 1.25
deg_per_pix = 0.02

rds = RDS(n_rds,
          rDot, dotDens, 
          size_rds_bg_deg, 
          size_rds_ct_deg,
          deg_per_pix)


disp_ct_deg = np.array([-0.1, 0.1]).astype(np.float32)
disp_ct_pix = rds._compute_deg2pix(disp_ct_deg)

rdsDisp_channels = len(disp_ct_pix)
dotMatch_list = [-1, 0, 0.5, 1]
# dotMatch_list = [1]
rds_name = ["u", "a", "hm", "c"]
for d in range(len(dotMatch_list)):
    
    dotMatch_ct = dotMatch_list[d]
        
    print("Create rds, dotMatch %s" %str(dotMatch_ct))
    
    # [n_trials, 2, size_rds_bg, size_rds_bg]
    rds_left_batch, rds_right_batch = rds.create_rds_batch(disp_ct_pix,
                                                           dotMatch_ct)
        
    suffix = rds_name[d]        
    np.save("../../../Data/rds_small_pixel/multivariate_analysis/rds_left_%s" %(suffix), rds_left_batch)
    np.save("../../../Data/rds_small_pixel/multivariate_analysis/rds_right_%s" %(suffix), rds_right_batch)
    
    
#################### create rds without rds background #########################
import numpy as np
from RDS.RDS import RDS

n_rds = 20000

rDot = 0.045
dotDens = 0.25
size_rds_bg_deg = 1
size_rds_ct_deg = 0.6
deg_per_pix = 0.01

rds = RDS(n_rds,
          rDot, dotDens, 
          size_rds_bg_deg, 
          size_rds_ct_deg,
          deg_per_pix)


disp_ct_deg = np.array([-0.1, 0.1])
disp_ct_pix = rds._compute_deg2pix(disp_ct_deg)

rdsDisp_channels = len(disp_ct_pix)
dotMatch_list = [-1, 0, 0.5, 1]
# dotMatch_list = [1]
rds_name = ["u", "a", "hm", "c"]
for d in range(len(dotMatch_list)):
    
    dotMatch_ct = dotMatch_list[d]
    
    rds_left_crossed = np.zeros((rds.n_trial, 
                                 rds.size_rds_bg_pix, rds.size_rds_bg_pix),
                                dtype= np.int8)
    rds_right_crossed = np.zeros((rds.n_trial, 
                                  rds.size_rds_bg_pix, rds.size_rds_bg_pix),
                                dtype= np.int8)
    rds_left_uncrossed = np.zeros((rds.n_trial, 
                                   rds.size_rds_bg_pix, rds.size_rds_bg_pix),
                                  dtype= np.int8)
    rds_right_uncrossed = np.zeros((rds.n_trial,
                                    rds.size_rds_bg_pix, rds.size_rds_bg_pix),
                                   dtype= np.int8)
    
    # for i in range(rds.n_bootstrap):
        
    print("Create rds, dotMatch %s" %str(dotMatch_ct))
    
    rds_left_batch, rds_right_batch = rds.create_rds_without_bg_batch(disp_ct_pix,
                                                                      dotMatch_ct)
    # a = rds_left_batch[0,0]

    rds_left_crossed = rds_left_batch[:, 0]
    rds_left_uncrossed = rds_left_batch[:, 1]
    
    rds_right_crossed = rds_right_batch[:, 0]
    rds_right_uncrossed = rds_right_batch[:, 1]
        
        
    suffix = rds_name[d]        
    np.save("../../../Data/rds_small_pixel/rds_left_crossed_%s" %(suffix), rds_left_crossed)
    np.save("../../../Data/rds_small_pixel/rds_left_uncrossed_%s" %(suffix), rds_left_uncrossed)
    np.save("../../../Data/rds_small_pixel/rds_right_crossed_%s" %(suffix), rds_right_crossed)
    np.save("../../../Data/rds_small_pixel/rds_right_uncrossed_%s" %(suffix), rds_right_uncrossed)
    
    
##############################################################################
## create 1-D rds

import numpy as np
from RDS.RDS import RDS


n_epoch = 4
n_batch = 256
n_bootstrap = 10

rDot = 0.2
dotDens = 0.35
size_rds_bg_deg = 7
size_rds_ct_deg = 3.5
deg_per_pix = 0.02

rds = RDS(n_epoch, n_batch, n_bootstrap,
          rDot, dotDens, 
          size_rds_bg_deg, 
          size_rds_ct_deg,
          deg_per_pix)


disp_ct_deg = np.array([-0.2, 0.2])
disp_ct_pix = rds._compute_deg2pix(disp_ct_deg)

rdsDisp_channels = len(disp_ct_pix)
dotMatch_list = [-1, 0, 0.5, 1]
# dotMatch_list = [1]
rds_name = ["u", "a", "hm", "c"]
for d in range(len(dotMatch_list)):
    
    dotMatch = dotMatch_list[d]
    
    rds1D_left_crossed = np.zeros((n_bootstrap, rds.n_trial, rds.size_rds_bg_pix),
                                  dtype= np.int8)
    rds1D_right_crossed = np.zeros((n_bootstrap, rds.n_trial, rds.size_rds_bg_pix),
                                  dtype= np.int8)
    rds1D_left_uncrossed = np.zeros((n_bootstrap, rds.n_trial, rds.size_rds_bg_pix),
                                  dtype= np.int8)
    rds1D_right_uncrossed = np.zeros((n_bootstrap, rds.n_trial, rds.size_rds_bg_pix),
                                  dtype= np.int8)
    
    for i in range(rds.n_bootstrap):
    
        print("Create rds, dotMatch {}, i_bootstrap: {}"
              .format(dotMatch, i+1))
        
        rds1D_left_batch, rds1D_right_batch = rds.create_rds1D_batch(disp_ct_pix,
                                                                     dotMatch)
    
        rds1D_left_crossed[i] = rds1D_left_batch[:, 0]
        rds1D_left_uncrossed[i] = rds1D_left_batch[:, 1]
        
        rds1D_right_crossed[i] = rds1D_right_batch[:, 0]
        rds1D_right_uncrossed[i] = rds1D_right_batch[:, 1]
        
            
    suffix = rds_name[d]        
    np.save("../../../Data/rds_small_pixel/rds1D_left_crossed_%s" %(suffix), rds1D_left_crossed)
    np.save("../../../Data/rds_small_pixel/rds1D_left_uncrossed_%s" %(suffix), rds1D_left_uncrossed)
    np.save("../../../Data/rds_small_pixel/rds1D_right_crossed_%s" %(suffix), rds1D_right_crossed)
    np.save("../../../Data/rds_small_pixel/rds1D_right_uncrossed_%s" %(suffix), rds1D_right_uncrossed)


        
