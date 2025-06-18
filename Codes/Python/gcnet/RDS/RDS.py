#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:29:30 2021

@author: cogni
"""

import numpy as np
from joblib import Parallel, delayed
from skimage.draw import disk

from timeit import default_timer as timer
from datetime import datetime

# import random
import sys
sys.path.insert(1, "../Common")

from Common.Common import General

# gen = General()


class RDS(General):
    def __init__(
        self, n_rds_trial, rDot, dotDens, 
        size_rds_bg_pix_height, size_rds_bg_pix_width
    ):

        super().__init__()

        # self.n_epoch = n_epoch
        # self.n_batch = n_batch
        # get the number of rds used for simulation. Total n_rds = 10240
        self.n_trial = n_rds_trial

        # # self.rDot = 0.045 # dot radius
        # # self.dotDens = 0.25 # dot density
        # # self.size_rds_bg_deg = 2.5 # rds size, deg
        # # self.size_rds_ct_deg = 1.25 # center rds size, deg

        ## parameters for 2-D rds
        self.rDot = rDot  # dot radius
        self.dotDens = dotDens  # dot density
        # self.size_rds_bg_deg = size_rds_bg_deg  # rds size, deg
        # self.size_rds_ct_deg = size_rds_ct_deg  # center rds size, deg
        # self.size_rds_bg_pix = self._compute_deg2pix(size_rds_bg_deg)
        self.size_rds_bg_pix_height = size_rds_bg_pix_height
        self.size_rds_bg_pix_width = size_rds_bg_pix_width
        # self.size_rds_ct_pix = self._compute_deg2pix(size_rds_ct_deg)
        # self.size_rds_bg = (self.size_rds_bg_pix, self.size_rds_bg_pix)
        self.size_rds_bg = (self.size_rds_bg_pix_height, 
                            self.size_rds_bg_pix_width)
        self.size_rds_ct = (self.size_rds_bg_pix_height//2,
                            self.size_rds_bg_pix_width//2)

        # disparity tuning axis in deg
        # self.deg_per_pix = 0.02 # deg per pix
        # step = 1*self.deg_per_pix
        # self.disp_ct_deg = np.round(np.arange(-0.25, (0.25 + step), step), 2)
        # self.disp_ct_pix = General._compute_deg2pix(self.disp_ct_deg) # disparity tuning axis in pix


    def _set_dotMatch(self, rds_ct, dotMatch_ct, nDots, rDot_pix):
        """
            set dot match level betweem rds left and right
            
            Inputs:
                - rds_ct: <2D np.array> rds center matrix
                - rds_bg: <2D np.array> rds background matrix
                - dotMatch_ct: <scalar>, dot match level, between 0 and 1.
                                -1 mean uncorrelated RDS
                                0 means anticorrelated RDS
                                0.5 means half-matched RDS
                                1 means correlated RDS
                            
            Outputs:
                rds_ct_left: <2D np.array>, rds for left
                rds_ct_right: <2D np.array>, rds for right
        """

        # find id_dot in rds_ct, excluding gray background
        dotID_ct = np.unique(rds_ct)[np.unique(rds_ct) > 0.5]

        if dotMatch_ct == -1:  # make urds
            nx, ny = np.shape(rds_ct)
            rDot_pix = self._compute_deg2pix(self.rDot)  # dot radius in pixel
            # nDots_ct = np.int32(self.dotDens*np.prod(self.size_rds_bg)/(np.pi*rDot_pix**2))
            nDots_ct = np.int32((nx * ny) / np.prod(self.size_rds_bg) * nDots)

            ## make rds left
            rds_ct_left = np.zeros((nx, ny), dtype=np.int8)
            rds_ct_right = rds_ct_left.copy()

            pos_x_left = np.random.randint(0, nx, nDots_ct).astype(np.int32)
            pos_y_left = np.random.randint(0, ny, nDots_ct).astype(np.int32)
            pos_x_right = np.random.randint(0, nx, nDots_ct).astype(np.int32)
            pos_y_right = np.random.randint(0, ny, nDots_ct).astype(np.int32)
            # distribute white dots
            for d in np.arange(0, np.int(nDots_ct / 2)):
                rr, cc = disk(
                    (pos_x_left[d], pos_y_left[d]), rDot_pix, shape=np.shape(rds_ct)
                )
                rds_ct_left[rr, cc] = 1

                rr, cc = disk(
                    (pos_x_right[d], pos_y_right[d]), rDot_pix, shape=np.shape(rds_ct)
                )
                rds_ct_right[rr, cc] = 1

            # distribute black dots
            for d in np.arange(np.int(nDots_ct / 2) + 1, nDots_ct):
                rr, cc = disk(
                    (pos_x_left[d], pos_y_left[d]), rDot_pix, shape=np.shape(rds_ct)
                )
                rds_ct_left[rr, cc] = -1

                rr, cc = disk(
                    (pos_x_right[d], pos_y_right[d]), rDot_pix, shape=np.shape(rds_ct)
                )
                rds_ct_right[rr, cc] = -1

        elif dotMatch_ct == 0:  # make ards
            rds_ct_left = rds_ct.copy()
            rds_ct_right = rds_ct.copy()

            id_start = 0
            id_end = np.int32(len(dotID_ct) / 2)

            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), -1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), 1, rds_ct_right
            )

            id_start = id_end + 1
            id_end = len(dotID_ct) - 1
            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), 1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), -1, rds_ct_right
            )

        elif (dotMatch_ct > 0) & (dotMatch_ct < 1):
            rds_ct_left = rds_ct.copy()
            rds_ct_right = rds_ct.copy()

            num_dot_to_match = np.int32(dotMatch_ct * len(dotID_ct))
            # always make even number
            if num_dot_to_match % 2 != 0:
                num_dot_to_match = num_dot_to_match - 1

            dotID_to_match = dotID_ct[0:num_dot_to_match]

            # distribute black dots
            id_start = 0
            id_end = np.int32(len(dotID_to_match) / 2)

            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), -1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), -1, rds_ct_right
            )

            # distribute white dots
            id_start = id_end + 1
            id_end = len(dotID_to_match) - 1
            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), 1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), 1, rds_ct_right
            )

            ## set other dots in rds_ct to be unmatched
            id_start = id_end + 1
            # id_end = id_start + np.int((len(dotID_ct)-len(dotID_to_match))/2)
            id_end = np.int32(len(dotID_ct) - 1)
            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), -1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), 1, rds_ct_right
            )

            # for i in range(id_start, id_end):
            #     x = dotID_ct[i]
            #     rds_ct_left[rds_ct_left==x] = 0
            #     rds_ct_right[rds_ct_right==x] = 1

            # id_start = id_end + 1
            # id_end = len(dotID_ct) - 1
            # x0 = dotID_ct[id_start]
            # x1 = dotID_ct[id_end]
            # rds_ct_left = np.where((rds_ct_left>=x0) & (rds_ct_left<=x1), 1,
            #                        rds_ct_left)
            # rds_ct_right = np.where((rds_ct_right>=x0) & (rds_ct_right<=x1), 0,
            #                         rds_ct_right)

            # check other dotID in rds_ct_left and rds_ct_right that hasn't been
            # converted to 0 or 1
            rds_ct_left[rds_ct_left > 1] = 1
            rds_ct_right[rds_ct_right > 1] = 1

        elif dotMatch_ct == 1:  # make crds
            rds_ct_left = rds_ct.copy()
            rds_ct_right = rds_ct.copy()

            # distribute black dots
            id_start = 0
            id_end = np.int32(len(dotID_ct) / 2)

            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), -1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), -1, rds_ct_right
            )

            # distribute white dots
            id_start = id_end + 1
            id_end = len(dotID_ct) - 1
            x0 = dotID_ct[id_start]
            x1 = dotID_ct[id_end]
            rds_ct_left = np.where(
                (rds_ct_left >= x0) & (rds_ct_left <= x1), 1, rds_ct_left
            )
            rds_ct_right = np.where(
                (rds_ct_right >= x0) & (rds_ct_right <= x1), 1, rds_ct_right
            )

        return rds_ct_left, rds_ct_right

    def create_rds_quad(
        self,
        size_rds_bg_deg,
        size_rds_bg_pix,
        quad_dist_deg,
        quad_rad_deg,
        disp_ct_pix,
        dotMatch_ct,
        dotDens,
        rDot,
    ):
        """
            Make a random dot stereogram with 4 circles inside the RDS.
            putting rds_ct to the center of rds_bg.
            rds_bg and rds_ct are a matrix with size_bg and size_ct, respectively:
                0.5 = gray background
                0 = black dot
                1 = white dot
                
            This module creates a set of rds with disparity listed on disp_ct_pix
            for creating a disparity tuning
            
            Inputs:
                - size_rds_bg_deg: <scalar>, size of rds background (diameter) 
                                    in degree,
                                    ex: 14
                            
                - size_rds_bg_pix: <scalar> size of rds background (diameter) 
                                            in pixels,
                                            ex: tool.fxCompute_deg2pix(size_rds_bg_deg)
                                            
                - quad_dist_deg: <scalar> distance between small quadrant-circles (deg)
                                ex: 5
                
                - quad_rad_deg: <scalar> radius of small quadrant-circles (deg)
                                ex: 2
                
                - disp_ct_pix: <np.array>, a list of disparity magnitude of center 
                                            rds (pixel)
                            
                            This variable is a kind of disparity axis in disparity 
                            tuning curve
                            
                            ex: 
                            disp_ct_deg = np.round(np.arange(-0.25, (0.25 + step), step), 2) # disparity mag of center rds in deg
                            disp_ct_pix = tool.fxCompute_deg2pix(disp_ct_deg) # disparity tuning axis in pix
    
                            
                - dotMatch_ct: <scalar>, dot match level of center rds, between 0 and 1.
                                -1 means uncorrelated RDS
                                0 means anticorrelated RDS
                                0.5 means half-matched RDS
                                1 means correlated RDS      
                                
                - dotDens: <scalar> dot density
                
                - rDot: <scalar> dot radius in degree
                
            Outputs:
                rds_left_set, rds_right_set: <[rdsDisp_channels, height, width] np.array>, A pair of rds with which is a 
                                    mixed of rds_bg and rds_ct
                                    
                                rdsDisp_channels: it means the channel (axis) containing
                                        disparity magnitude as given in disp_ct_pix
        """

        # rdsDisp_channels: <scalar>, number of disparity points in disparity tuning function
        rdsDisp_channels = len(disp_ct_pix)

        rDot_pix = self._compute_deg2pix(rDot)  # dot radius in pixel
        # nDots = np.int(dotDens*(size_rds_bg_pix**2)/(np.pi*rDot_pix**2))
        nDots = np.int(dotDens * (size_rds_bg_pix ** 2))
        # always make even number
        if nDots % 2 != 0:
            nDots = nDots - 1

        # allocate memory for rds that follows the format "NHWC" (batch_size, height, width, in_channels)
        rds_left_set = np.zeros(
            (rdsDisp_channels, size_rds_bg_pix, size_rds_bg_pix), dtype="float32"
        )
        rds_right_set = np.zeros(
            (rdsDisp_channels, size_rds_bg_pix, size_rds_bg_pix), dtype="float32"
        )

        for i in range(rdsDisp_channels):

            # create rds matrix for rds background and rds center that has value -1, 0, and 1
            ## make rds background
            rds_bg = 0.5 * np.ones(
                (size_rds_bg_pix, size_rds_bg_pix), dtype="float32"
            )  # for indexing dots
            rds_bg2 = 0.5 * np.ones(
                (size_rds_bg_pix, size_rds_bg_pix), dtype="float32"
            )  # for black and white rds_bg
            pos_x = np.random.randint(0, size_rds_bg_pix, nDots).astype(np.int32)
            pos_y = np.random.randint(0, size_rds_bg_pix, nDots).astype(np.int32)

            for d in np.arange(1, np.int(nDots)):
                rr, cc = disk(
                    (pos_x[d], pos_y[d]), rDot_pix, (size_rds_bg_pix, size_rds_bg_pix)
                )
                rds_bg[rr, cc] = d

                # distribute white dots
                if d <= np.int(nDots / 2):
                    rds_bg2[rr, cc] = 1
                else:  # distribute black dots
                    rds_bg2[rr, cc] = 0

            ## make 4-quadrant mask
            x = np.linspace(
                -size_rds_bg_deg / 2, size_rds_bg_deg / 2, size_rds_bg_pix
            ).astype("float32")
            y = np.linspace(
                -size_rds_bg_deg / 2, size_rds_bg_deg / 2, size_rds_bg_pix
            ).astype("float32")

            xx, yy = np.float32(np.meshgrid(x, y))
            mask = (
                (
                    (xx - quad_dist_deg / 2) ** 2 + (yy - quad_dist_deg / 2) ** 2
                    <= quad_rad_deg ** 2
                )
                + (
                    (xx - quad_dist_deg / 2) ** 2 + (yy + quad_dist_deg / 2) ** 2
                    <= quad_rad_deg ** 2
                )
                + (
                    (xx + quad_dist_deg / 2) ** 2 + (yy - quad_dist_deg / 2) ** 2
                    <= quad_rad_deg ** 2
                )
                + (
                    (xx + quad_dist_deg / 2) ** 2 + (yy + quad_dist_deg / 2) ** 2
                    <= quad_rad_deg ** 2
                )
            )

            # Fill dots in the 4-quadrant circles
            rds_quad = rds_bg * mask

            # set dotMatch level in the 4-quadrant circles (center rds)
            rds_ct_left, rds_ct_right = self._set_dotMatch(
                rds_quad, dotMatch_ct, nDots, rDot_pix
            )

            ## put rds_ct into rds_bg
            # make rds_left
            # shift rds_ct to set disparity magnitude
            rds_shift = np.roll(rds_ct_left, -int(disp_ct_pix[i] / 2))
            mask_shift = np.roll(mask, -int(disp_ct_pix[i] / 2))
            rds_left = rds_bg2 * (~mask_shift) + rds_shift
            rds_left_set[i, :, :] = rds_left

            # make rds_right
            # shift rds_ct to set disparity magnitude
            rds_shift = np.roll(rds_ct_right, int(disp_ct_pix[i] / 2))
            mask_shift = np.roll(mask, int(disp_ct_pix[i] / 2))
            rds_right = rds_bg2 * (~mask_shift) + rds_shift
            rds_right_set[i, :, :] = rds_right

        rds_all = np.zeros(
            (2, rdsDisp_channels, size_rds_bg_pix, size_rds_bg_pix), dtype="float32"
        )
        rds_all[0] = rds_left_set
        rds_all[1] = rds_right_set

        return rds_all

    def create_rds_quad_batch(
        self,
        size_rds_bg_deg,
        size_rds_bg_pix,
        quad_dist_deg,
        quad_rad_deg,
        disp_ct_pix,
        dotMatch_ct,
        dotDens,
        rDot,
        nBatch,
        n_jobs=18,
        backend="loky",
    ):
        """
            Make a batch of random dot stereogram with 4 circles inside the RDS.
            putting rds_ct to the center of rds_bg.
            rds_bg and rds_ct are a matrix with size_bg and size_ct, respectively:
                0.5 = gray background
                0 = black dot
                1 = white dot
                
            This module creates a set of rds with disparity listed on disp_ct_pix
            for creating a disparity tuning
            
            Inputs:
                - size_rds_bg_deg: <scalar>, size of rds background (diameter) 
                                    in degree,
                                    ex: 14
                            
                - size_rds_bg_pix: <scalar> size of rds background (diameter) 
                                            in pixels,
                                            ex: tool.fxCompute_deg2pix(size_rds_bg_deg)
                                            
                - quad_dist_deg: <scalar> distance between small quadrant-circles (deg)
                                ex: 5
                
                - quad_rad_deg: <scalar> radius of small quadrant-circles (deg)
                                ex: 2
                
                - disp_ct_pix: <np.array>, a list of disparity magnitude of center 
                                            rds (pixel)
                            
                            This variable is a kind of disparity axis in disparity 
                            tuning curve
                            
                            ex: 
                            disp_ct_deg = np.round(np.arange(-0.25, (0.25 + step), step), 2) # disparity mag of center rds in deg
                            disp_ct_pix = tool.fxCompute_deg2pix(disp_ct_deg) # disparity tuning axis in pix
    
                            
                - dotMatch_ct: <scalar>, dot match level of center rds, between 0 and 1.
                                -1 means uncorrelated RDS
                                0 means anticorrelated RDS
                                0.5 means half-matched RDS
                                1 means correlated RDS      
                                
                - dotDens: <scalar> dot density
                
                - rDot: <scalar> dot radius in degree
                
            Outputs:
                rds_left_unpack, rds_right_unpack: <[nBatch, rdsDisp_channels, height, width] np.array>, A pair of rds with which is a 
                                    mixed of rds_bg and rds_ct
        """

        rdsDisp_channels = len(disp_ct_pix)

        now = datetime.now()
        time_start = now.strftime("%H:%M:%S")
        t_start = timer()
        rds_batch = []
        rds_batch.append(
            Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(self.create_rds_quad)(
                    size_rds_bg_deg,
                    size_rds_bg_pix,
                    quad_dist_deg,
                    quad_rad_deg,
                    disp_ct_pix,
                    dotMatch_ct,
                    dotDens,
                    rDot,
                )
                for i in range(nBatch)
            )
        )
        t_end = timer()
        now = datetime.now()
        time_end = now.strftime("%H:%M:%S")
        print(time_start, time_end, t_end - t_start)

        # unpack rds_batch
        rds_left_unpack = np.zeros(
            (nBatch, rdsDisp_channels, size_rds_bg_pix, size_rds_bg_pix),
            dtype="float32",
        )
        rds_right_unpack = np.zeros(
            (nBatch, rdsDisp_channels, size_rds_bg_pix, size_rds_bg_pix),
            dtype="float32",
        )
        for i in range(nBatch):
            rds_unpack = rds_batch[0][i]

            rds_left_unpack[i] = rds_unpack[0]
            rds_right_unpack[i] = rds_unpack[1]

        # convert to float32
        rds_left_unpack = rds_left_unpack.astype("float32") - 0.5
        rds_right_unpack = rds_right_unpack.astype("float32") - 0.5

        return rds_left_unpack, rds_right_unpack

    def create_rds(self, disp_ct_pix, dotMatch_ct):
        """
            Make a random dot stereogram by putting rds_ct to the center of rds_bg.
            rds_bg and rds_ct are a matrix with size_bg and size_ct, respectively:
                0.5 = gray background
                0 = black dot
                1 = white dot
                
            This module creates a set of rds with disparity listed on disp_ct_pix            
            
            Inputs:
                - size_rds_bg: <tuple>, size of rds background in pixel, ex: (501,501)
                - size_rds_ct: <tuple> size of rds center in pixel, ex: (251,251)
                - disp_ct_pix: <np.array>, a list of disparity magnitude of center 
                                            rds (pixel)
                            
                            This variable is a kind of disparity axis in disparity 
                            tuning curve
                            
                            ex: 
                            disp_ct_deg = np.round(np.arange(-0.4, 
                                                             (0.4 + deg_per_pix), 
                                                             deg_per_pix), 
                                                   2)
                            disp_ct_pix = cm.fxCompute_deg2pix(disp_ct_deg)
                            
                - dotMatch_ct: <scalar>, dot match level of center rds, between 0 and 1.
                                -1 means uncorrelated RDS
                                0 means anticorrelated RDS
                                0.5 means half-matched RDS
                                1 means correlated RDS      
                                
                - dotDens: <scalar> dot density
                
                - rDot: <scalar> dot radius in degree
                
            Outputs:
                rds_left_set, rds_right_set: <[rdsDisp_channels, height, width] np.array>, 
                        A pair of rds with which is a 
                        mixed of rds_bg and rds_ct
        """

        # rdsDisp_channels: <scalar>, number of disparity points in disparity tuning function
        # disp_ct_pix = (9, -9) # disparity in pixel associated with +-0.2 deg
        rdsDisp_channels = len(disp_ct_pix)

        rDot_pix = self._compute_deg2pix(self.rDot)  # dot radius in pixel
        nDots = np.int32(
            self.dotDens * np.prod(self.size_rds_bg) / (np.pi * rDot_pix ** 2)
        )
        # rDot_pix = rds._compute_deg2pix(rds.rDot) # dot radius in pixel
        # nDots = np.int32(rds.dotDens*np.prod(rds.size_rds_bg)/(np.pi*rDot_pix**2))
        # nDots = np.int32((dotDens*48**2)/(np.pi*rDot_pix**2))
        # nDots = np.int32(self.dotDens*np.prod(self.size_rds_bg))
        # always make even number
        # if nDots%2!=0:
        #     nDots = nDots -1

        # calculate center position in pixel
        center = np.array([self.size_rds_bg[0] / 2, self.size_rds_bg[1] / 2]).astype(
            np.int32
        )
        # center = np.array([rds.size_rds_bg[0]/2,
        #                     rds.size_rds_bg[1]/2]).astype(np.int16)

        # make coordinate for inserting the rds_ct to rds_bg (in pixel)
        # row = np.arange(center[0] - self.size_rds_ct[0]/2,
        #                 center[0] + self.size_rds_ct[0]/2 + 1).astype(np.int16)
        # col = np.arange(center[1] - self.size_rds_ct[1]/2,
        #                 center[1] + self.size_rds_ct[1]/2 + 1).astype(np.int16)
        # size_rds_ct = (26, 26) # rds_center size in pix associated with 0.55deg
        # row = np.arange(center[0]-int(size_rds_ct[0]/2),
        #                 center[0]+int(size_rds_ct[0]/2) + 2).astype(np.int16)
        # col = np.arange(center[1]-int(size_rds_ct[1]/2),
        #                 center[1]+int(size_rds_ct[1]/2) + 2).astype(np.int16)

        # allocate memory for rds that follows the format "NHWC" (batch_size, height, width, in_channels)
        rds_left_set = np.zeros(
            (rdsDisp_channels, self.size_rds_bg[0], self.size_rds_bg[1]), 
            dtype=np.int32
        )
        rds_right_set = np.zeros(
            (rdsDisp_channels, self.size_rds_bg[0], self.size_rds_bg[1]), 
            dtype=np.int32
        )

        for i in range(rdsDisp_channels):
            # create rds matrix for rds background and rds center that has pixel value -1, 0, and 1
            ## make rds background
            rds_bg = np.zeros(self.size_rds_bg, dtype=np.int32)  # for indexing dots
            rds_bg2 = rds_bg.copy()  # for black and white rds_bg
            pos_x = np.random.randint(0, self.size_rds_bg[0], nDots).astype(np.int32)
            pos_y = np.random.randint(0, self.size_rds_bg[1], nDots).astype(np.int32)
            # rds_bg = np.zeros(rds.size_rds_bg, dtype=np.int8) # for indexing dots
            # rds_bg2 = rds_bg.copy() # for black and white rds_bg
            # pos_x = np.random.randint(0, rds.size_rds_bg[0], nDots).astype(np.int16)
            # pos_y = np.random.randint(0, rds.size_rds_bg[1], nDots).astype(np.int16)

            for d in np.arange(nDots):
                rr, cc = disk((pos_x[d], pos_y[d]), rDot_pix, 
                              shape=self.size_rds_bg)
                # rr, cc = disk((pos_x[d], pos_y[d]), rDot_pix,
                #               shape=(48,48))
                rds_bg[rr, cc] = d

                # distribute white dots
                if d <= np.int32(nDots / 2):
                    rds_bg2[rr, cc] = 1
                else:  # distribute black dots
                    rds_bg2[rr, cc] = -1

            ## make rds center
            row_start = np.int32(center[0] - self.size_rds_ct[0] / 2)
            row_end = np.int32(row_start + self.size_rds_ct[0] + 1)
            col_start = np.int32(center[1] - self.size_rds_ct[1] / 2)
            col_end = np.int32(col_start + self.size_rds_ct[1] + 1)
            # id_start = center[0] - np.int(size_rds_ct[0]/2)
            # id_end = center[0] + np.int(size_rds_ct[0]/2) + 1
            rds_ct = rds_bg[row_start:row_end, col_start:col_end]

            # set dotMatch level
            rds_ct_left, rds_ct_right = self._set_dotMatch(
                rds_ct, dotMatch_ct, nDots, rDot_pix
            )

            ## put rds_ct into rds_bg
            # make rds_left
            id_row_start = (center[0] - rds_ct_left.shape[0] / 2).astype(np.int32)
            id_row_end = (id_row_start + rds_ct_left.shape[0]).astype(np.int32)
            id_col_start = (
                center[1] - rds_ct_left.shape[1] / 2 - disp_ct_pix[i] / 2
            ).astype(np.int32)
            id_col_end = (id_col_start + rds_ct_left.shape[1]).astype(np.int32)
            # pos_left = col - np.int16(disp_ct_pix[i]/2) # make dot index with disparity for rds_left
            rds_left = rds_bg2.copy()
            # rds_left[row[0]:row[-1], pos_left[0]:pos_left[-1]] = rds_ct_left # set disparity magnitude
            rds_left[id_row_start:id_row_end, id_col_start:id_col_end] = rds_ct_left
            rds_left_set[i, :, :] = rds_left

            # make rds_right
            id_row_start = (center[0] - rds_ct_right.shape[0] / 2).astype(np.int32)
            id_row_end = (id_row_start + rds_ct_right.shape[0]).astype(np.int32)
            id_col_start = (
                center[1] - rds_ct_right.shape[1] / 2 + disp_ct_pix[i] / 2
            ).astype(np.int32)
            id_col_end = (id_col_start + rds_ct_right.shape[1]).astype(np.int32)
            # pos_right = col + np.int16(disp_ct_pix[i]/2) # make dot index with disparity for rds_right
            rds_right = rds_bg2.copy()
            # rds_right[row[0]:row[-1], pos_right[0]:pos_right[-1]] = rds_ct_right # set disparity magnitude
            rds_right[id_row_start:id_row_end, id_col_start:id_col_end] = rds_ct_right
            rds_right_set[i, :, :] = rds_right

        rds_all = np.zeros(
            (2, rdsDisp_channels, self.size_rds_bg[0], self.size_rds_bg[1]),
            dtype=np.int32,
        )

        rds_all[0] = rds_left_set
        rds_all[1] = rds_right_set

        return rds_all

    def create_rds_batch(self, disp_ct_pix, dotMatch_ct):
        """
            Make nBatch of random dot stereogram obtained from fxCreate_rds
            
            rds_bg and rds_ct are a matrix with size_bg and size_ct, respectively:
                0.5 = gray background
                0 = black dot
                1 = white dot
                
            This module creates a set of rds with disparity listed on disp_ct_pix
            
            Inputs:
                - size_rds_bg: <tuple>, size of rds background, ex: (501,501)
                - size_rds_ct: <tuple> size of rds center, ex: (251,251)
                - disp_ct_pix: <np.array>, a list of disparity magnitude of center 
                                            rds (pixel)
                            
                            This variable is a kind of disparity axis in disparity 
                            tuning curve
                            
                            ex: 
                            disp_ct_deg = np.round(np.arange(-0.4, 
                                                             (0.4 + deg_per_pix), 
                                                             deg_per_pix), 
                                                   2)
                            disp_ct_pix = cm.fxCompute_deg2pix(disp_ct_deg)
                            
                - dotMatch_ct: <scalar>, dot match level of center rds, between 0 and 1.
                                -1 means uncorrelated RDS
                                0 means anticorrelated RDS
                                0.5 means half-matched RDS
                                1 means correlated RDS      
                                
                - dotDens: <scalar> dot density
                
                - rDot: <scalar> dot radius in degree
                
                - nBatch: <scalar> number of batch size (ex: 1000)
                
                - n_workers: <scalar>: number of cpu
                
            Outputs:
                rds_left_unpack, rds_right_unpack: <[nBatch, rdsDisp_channels, height, width] np.array>,
                                nBatch pair of rds with which are a mixed of rds_bg and rds_ct
        """

        #    nBatch = 10
        rdsDisp_channels = len(disp_ct_pix)
        nx = self.size_rds_bg[0]
        ny = self.size_rds_bg[1]

        now = datetime.now()
        time_start = now.strftime("%H:%M:%S")
        t_start = timer()
        rds_batch = []
        rds_batch.append(
            Parallel(n_jobs=-1)(
                delayed(self.create_rds)(disp_ct_pix, dotMatch_ct)
                for i in range(self.n_trial)
            )
        )
        t_end = timer()
        now = datetime.now()
        time_end = now.strftime("%H:%M:%S")
        print(time_start, time_end, t_end - t_start)

        # unpack rds_batch
        rds_left_unpack = np.zeros(
            (self.n_trial, rdsDisp_channels, nx, ny), dtype=np.int32
        )
        rds_right_unpack = np.zeros(
            (self.n_trial, rdsDisp_channels, nx, ny), dtype=np.int32
        )
        for i in range(self.n_trial):
            rds_unpack = rds_batch[0][i]

            rds_left_unpack[i] = rds_unpack[0]
            rds_right_unpack[i] = rds_unpack[1]

        return rds_left_unpack, rds_right_unpack

    def create_rds_without_bg(self, disp_ct_pix, dotMatch_ct):
        """
            Make a single plane of random dot stereogram (without background RDS).
            it means that the whole dots in RDS are shifted to set the disparity.
            
            The pixel values are as follow:
                0 = gray background
                -1 = black dot
                1 = white dot                            
                                
            Outputs:
                rds_all: <[2, len(disp_ct_pix), size_rds_bg, size_rds_bg] np.array>, 
                        A pair of rds with which is a 
                        mixed of rds_bg and rds_ct
        """

        # rdsDisp_channels: <scalar>, number of disparity points in disparity tuning function
        # rdsDisp_channels = len(disp_ct_pix)

        rDot_pix = self._compute_deg2pix(self.rDot)  # dot radius in pixel
        nDots = np.int32(
            self.dotDens * np.prod(self.size_rds_bg) / (np.pi * rDot_pix ** 2)
        )
        # always make even number
        # if nDots%2!=0:
        #     nDots = nDots -1

        # rdsDisp_channels: <scalar>, number of disparity points in disparity tuning function
        rdsDisp_channels = len(disp_ct_pix)

        # allocate memory for rds that follows the format "NHWC" (batch_size, height, width, in_channels)
        rds_left_set = np.zeros(
            (rdsDisp_channels, self.size_rds_bg[0], self.size_rds_bg[1]), dtype=np.int32
        )
        rds_right_set = np.zeros(
            (rdsDisp_channels, self.size_rds_bg[0], self.size_rds_bg[1]), dtype=np.int32
        )

        for d in range(rdsDisp_channels):  # iterate over crossed-uncrossed disparity

            ## create rds matrix for rds background and rds center that has pixel value -1, 0, and 1
            # make rds background
            rds_bg = np.zeros(self.size_rds_bg, dtype=np.int32)  # for indexing dots
            rds_bg2 = rds_bg.copy()  # for black and white rds_bg
            pos_x = np.random.randint(0, self.size_rds_bg[0], nDots).astype(np.int32)
            pos_y = np.random.randint(0, self.size_rds_bg[1], nDots).astype(np.int32)
            # rds_bg = np.zeros((48, 48), dtype=np.int8) # for indexing dots
            # rds_bg2 = rds_bg.copy() # for black and white rds_bg
            # pos_x = np.random.randint(0, rds.size_rds_bg[0], nDots).astype(np.int32)
            # pos_y = np.random.randint(0, rds.size_rds_bg[1], nDots).astype(np.int32)

            for i_dot in np.arange(nDots):  # iterate over dot ID
                rr, cc = disk(
                    (pos_x[i_dot], pos_y[i_dot]), rDot_pix, shape=self.size_rds_bg
                )
                # rr, cc = disk((pos_x[d], pos_y[d]), rDot_pix,
                #               shape=(48,48))
                rds_bg[rr, cc] = i_dot

                # distribute white dots
                if i_dot <= np.int32(nDots / 2):
                    rds_bg2[rr, cc] = 1
                else:  # distribute black dots
                    rds_bg2[rr, cc] = -1

            ## set dotMatch level
            rds_bg_left, rds_bg_right = self._set_dotMatch(
                rds_bg, dotMatch_ct, nDots, rDot_pix
            )
            # rds_bg_left, rds_bg_right = rds._set_dotMatch(rds_bg,
            #                                                dotMatch_ct,
            #                                                nDots,
            #                                                rDot_pix)

            ## make rds_left: shift all pixels to the left
            rds_left = np.roll(
                rds_bg_left, -np.int32(disp_ct_pix[d] / 2), axis=1
            )  # set disparity magnitude
            rds_left_set[d, :, :] = rds_left

            ## make rds_right: shift all pixels to the right
            rds_right = np.roll(
                rds_bg_right, np.int32(disp_ct_pix[d] / 2), axis=1
            )  # set disparity magnitude
            rds_right_set[d, :, :] = rds_right

        ## alocate array to store the left and right rds images
        rds_all = np.zeros(
            (2, rdsDisp_channels, self.size_rds_bg[0], self.size_rds_bg[1]),
            dtype=np.int32,
        )

        rds_all[0] = rds_left_set
        rds_all[1] = rds_right_set

        return rds_all

    def create_rds_without_bg_batch(self, disp_ct_pix, dotMatch_ct):
        """
            Make nBatch of random dot stereogram obtained from fxCreate_rds
            
            rds_bg and rds_ct are a matrix with size_bg and size_ct, respectively:
                0.5 = gray background
                0 = black dot
                1 = white dot
                
            This module creates a set of rds with disparity listed on disp_ct_pix
            
            
            Inputs:
                - size_rds_bg: <tuple>, size of rds background, ex: (501,501)
                - size_rds_ct: <tuple> size of rds center, ex: (251,251)
                - disp_ct_pix: <np.array>, a list of disparity magnitude of center 
                                            rds (pixel)
                            
                            This variable is a kind of disparity axis in disparity 
                            tuning curve
                            
                            ex: 
                            disp_ct_deg = np.round(np.arange(-0.4, 
                                                             (0.4 + deg_per_pix), 
                                                             deg_per_pix), 
                                                   2)
                            disp_ct_pix = cm.fxCompute_deg2pix(disp_ct_deg)
                            
                - dotMatch_ct: <scalar>, dot match level of center rds, between 0 and 1.
                                -1 means uncorrelated RDS
                                0 means anticorrelated RDS
                                0.5 means half-matched RDS
                                1 means correlated RDS      
                                
                - dotDens: <scalar> dot density
                
                - rDot: <scalar> dot radius in degree
                
                - nBatch: <scalar> number of batch size (ex: 1000)
                
                - n_workers: <scalar>: number of cpu
                
            Outputs:
                rds_left_unpack: <[n_trials, len(disp_ct_pix), 
                                 size_rds_bg, size_rds_bg] np.array>,
                                n_trials pair of rds whose whole pixels are shifted
                                
                rds_right_unpack: <[n_trials, len(disp_ct_pix), 
                                 size_rds_bg, size_rds_bg] np.array>,
                                n_trials pair of rds whose whole pixels are shifted
        """

        #    nBatch = 10
        rdsDisp_channels = len(disp_ct_pix)
        nx = self.size_rds_bg[0]
        ny = self.size_rds_bg[1]

        now = datetime.now()
        time_start = now.strftime("%H:%M:%S")
        t_start = timer()
        rds_batch = []
        rds_batch.append(
            Parallel(n_jobs=-1)(
                delayed(self.create_rds_without_bg)(disp_ct_pix, dotMatch_ct)
                for i in range(self.n_trial)
            )
        )

        # rds_batch.append(Parallel(n_jobs=-1)
        #                         (delayed(rds.create_rds_without_bg)
        #                          (disp_ct_pix, dotMatch_ct)
        #                          for i in range(rds.n_trial)))

        t_end = timer()
        now = datetime.now()
        time_end = now.strftime("%H:%M:%S")
        print(time_start, time_end, t_end - t_start)

        # unpack rds_batch
        rds_left_unpack = np.zeros(
            (self.n_trial, rdsDisp_channels, nx, ny), dtype=np.int32
        )
        rds_right_unpack = np.zeros(
            (self.n_trial, rdsDisp_channels, nx, ny), dtype=np.int32
        )
        for i in range(self.n_trial):
            rds_unpack = rds_batch[0][i]

            rds_left_unpack[i] = rds_unpack[0]
            rds_right_unpack[i] = rds_unpack[1]

        return rds_left_unpack, rds_right_unpack
    

    def load_rds(self, rds_type):
        """
        
        load rds that has crossed and uncrossed disparity

        Args:
            rds_type (str): type of rds: "ards", "hmrds", "crds".

        Returns:
            rds dimension:
            <[n_trial, crossed_uncrossed, size_rds, size_rds] np.array>
            
            ex: [10240, 2, 120, 120]
            
        """
        print("load rds: {}".format(rds_type))
        self.L = np.zeros(
            (self.n_trial, 2, self.size_rds_bg_pix, self.size_rds_bg_pix), dtype=np.int32
        )
        self.R = np.zeros(
            (self.n_trial, 2, self.size_rds_bg_pix, self.size_rds_bg_pix), dtype=np.int32
        )
        # self.rds_bg = np.zeros((n_trial, 2, self.size_rds_bg_pix, self.size_rds_bg_pix),
        #                        dtype=np.float32)

        if rds_type == "ards":
            # if disp_type=="crossed":
            temp = np.load("../../../../Data/rds/ards_L_crossed.npy")

            # generate n_trial of random integers
            rdx_idx = np.random.randint(0, temp.shape[0], self.n_trial)

            # temp = np.load("../../../Data/rds_small_pixel/rds_left_crossed_a.npy")
            self.L[:, 0] = temp[rdx_idx]

            temp = np.load("../../../../Data/rds/ards_L_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_left_uncrossed_a.npy")
            self.L[:, 1] = temp[rdx_idx]

            temp = np.load("../../../../Data/rds/ards_R_crossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_crossed_a.npy")
            self.R[:, 0] = temp[rdx_idx]

            temp = np.load("../../../../Data/rds/ards_R_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_uncrossed_a.npy")
            self.R[:, 1] = temp[rdx_idx]

        elif rds_type == "hmrds":
            # if disp_type=="crossed":
            temp = np.load("../../../../Data/rds/hmrds_L_crossed.npy")

            # generate n_trial of random integers
            rdx_idx = np.random.randint(0, temp.shape[0], self.n_trial)

            # temp = np.load("../../../Data/rds_small_pixel/rds_left_crossed_hm.npy")
            self.L[:, 0] = temp[rdx_idx]

            temp = np.load("../../../../Data/rds/hmrds_L_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_left_uncrossed_hm.npy")
            self.L[:, 1] = temp[rdx_idx]

            temp = np.load("../../../../Data/rds/hmrds_R_crossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_crossed_hm.npy")
            self.R[:, 0] = temp[rdx_idx]

            temp = np.load("../../../../Data/rds/hmrds_R_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_uncrossed_hm.npy")
            self.R[:, 1] = temp[rdx_idx]

        elif rds_type == "crds":
            # if disp_type=="crossed":
            temp = np.load("../../../../Data/rds/crds_L_crossed.npy")

            # generate n_trial of random integers
            rdx_idx = np.random.randint(0, temp.shape[0], self.n_trial)

            # temp = np.load("../../../Data/rds_small_pixel/rds_left_crossed_c.npy")
            self.L[:, 0] = temp[rdx_idx]

            temp = np.load("../../../../Data/rds/crds_L_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_left_uncrossed_c.npy")
            self.L[:, 1] = temp[rdx_idx]

            temp = np.load("../../../../Data/rds/crds_R_crossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_crossed_c.npy")
            self.R[:, 0] = temp[rdx_idx]

            temp = np.load("../../../../Data/rds/crds_R_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_uncrossed_c.npy")
            self.R[:, 1] = temp[rdx_idx]

        self.rds_type = rds_type

        ## load urds
        print("load urds")
        self.u_L = np.zeros(
            (self.n_trial, 2, self.size_rds_bg_pix, self.size_rds_bg_pix), dtype=np.int32
        )
        self.u_R = np.zeros(
            (self.n_trial, 2, self.size_rds_bg_pix, self.size_rds_bg_pix), dtype=np.int32
        )

        temp = np.load("../../../../Data/rds/urds_L_crossed.npy")

        # generate n_trial of random integers
        rdx_idx = np.random.randint(0, temp.shape[0], self.n_trial)

        # temp = np.load("../../../Data/rds_small_pixel/rds_left_crossed_u.npy")
        self.u_L[:, 0] = temp[rdx_idx]

        temp = np.load("../../../../Data/rds/urds_L_uncrossed.npy")
        # temp = np.load("../../../Data/rds_small_pixel/rds_left_uncrossed_u.npy")
        self.u_L[:, 1] = temp[rdx_idx]

        temp = np.load("../../../../Data/rds/urds_R_crossed.npy")
        # temp = np.load("../../../Data/rds_small_pixel/rds_right_crossed_u.npy")
        self.u_R[:, 0] = temp[rdx_idx]

        temp = np.load("../../../../Data/rds/urds_R_uncrossed.npy")
        # temp = np.load("../../../Data/rds_small_pixel/rds_right_uncrossed_u.npy")
        self.u_R[:, 1] = temp[rdx_idx]

        ## load rds_background
        # print("load rds_bg")
        # temp = np.load("../../../Data/rds/rds_bg.npy")
        # self.rds_bg[:, 0] = temp[0:n_trial]

    def load_rds_disp_tuning(self, rds_type):
        """
        
        load rds for generating disparity tuning function

        Args:
            rds_type (str): type of rds: "ards", "hmrds", "crds".

        Returns:
            rds dimension:
            <[n_trial, len(disp_ct_deg), size_rds, size_rds] np.array>
            
            disp_ct_deg = np.linspace(-0.2, 0.2, 21).astype(np.float32) -> 11 disparity mag
            
            ex: [1000, 21, 120, 120]
            
        """
        print("load rds: {}".format(rds_type))
        self.L = np.zeros(
            (self.n_trial, 21, self.size_rds_bg_pix, self.size_rds_bg_pix),
            dtype=np.int8,
        )
        self.R = np.zeros(
            (self.n_trial, 21, self.size_rds_bg_pix, self.size_rds_bg_pix),
            dtype=np.int8,
        )

        ## allocate array for uncorrelated dots
        self.u_L = np.zeros(
            (self.n_trial, 21, self.size_rds_bg_pix, self.size_rds_bg_pix),
            dtype=np.int8,
        )
        self.u_R = np.zeros(
            (self.n_trial, 21, self.size_rds_bg_pix, self.size_rds_bg_pix),
            dtype=np.int8,
        )

        if rds_type == "ards":
            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_left_a.npy")
            self.L[0 : self.n_trial] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_right_a.npy")
            self.R[0 : self.n_trial] = temp[0 : self.n_trial]

        elif rds_type == "hmrds":
            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_left_hm.npy")
            self.L[0 : self.n_trial] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_right_hm.npy")
            self.R[0 : self.n_trial] = temp[0 : self.n_trial]

        elif rds_type == "crds":
            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_left_c.npy")
            self.L[0 : self.n_trial] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_right_c.npy")
            self.R[0 : self.n_trial] = temp[0 : self.n_trial]

        ## load uncorrelated dots
        temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_left_u.npy")
        self.u_L[0 : self.n_trial] = temp[0 : self.n_trial]

        temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_right_u.npy")
        self.u_R[0 : self.n_trial] = temp[0 : self.n_trial]

        self.rds_type = rds_type

    def load_rds_multivariate_analysis(self, rds_type):
        """
        
        load rds for generating disparity tuning function

        Args:
            rds_type (str): type of rds: "ards", "hmrds", "crds".

        Returns:
            rds dimension:
            <[n_trial, len(disp_ct_deg), size_rds, size_rds] np.array>
            
            disp_ct_deg = np.array([-0.1, 0.1]) -> 2 disparity mag.
            
            ex: [1000, 2, 120, 120]
            
        """
        print("load rds: {}".format(rds_type))
        # self.L = np.zeros((self.n_trial, 4,
        #                    self.size_rds_bg_pix, self.size_rds_bg_pix),
        #                   dtype=np.int8)
        # self.R = np.zeros((self.n_trial, 4,
        #                    self.size_rds_bg_pix, self.size_rds_bg_pix),
        #                   dtype=np.int8)

        if rds_type == "ards":
            temp = np.load(
                "../../../Data/rds_small_pixel/multivariate_analysis/rds_left_a.npy"
            )
            self.L = temp[0 : self.n_trial]

            temp = np.load(
                "../../../Data/rds_small_pixel/multivariate_analysis/rds_right_a.npy"
            )
            self.R = temp[0 : self.n_trial]

        elif rds_type == "hmrds":
            temp = np.load(
                "../../../Data/rds_small_pixel/multivariate_analysis/rds_left_hm.npy"
            )
            self.L = temp[0 : self.n_trial]

            temp = np.load(
                "../../../Data/rds_small_pixel/multivariate_analysis/rds_right_hm.npy"
            )
            self.R = temp[0 : self.n_trial]

        elif rds_type == "crds":
            temp = np.load(
                "../../../Data/rds_small_pixel/multivariate_analysis/rds_left_c.npy"
            )
            self.L = temp[0 : self.n_trial]

            temp = np.load(
                "../../../Data/rds_small_pixel/multivariate_analysis/rds_right_c.npy"
            )
            self.R = temp[0 : self.n_trial]

        ## load uncorrelated dots
        temp = np.load(
            "../../../Data/rds_small_pixel/multivariate_analysis/rds_left_u.npy"
        )
        self.u_L = temp[0 : self.n_trial]

        temp = np.load(
            "../../../Data/rds_small_pixel/multivariate_analysis/rds_right_u.npy"
        )
        self.u_R = temp[0 : self.n_trial]

        self.rds_type = rds_type

    def set_rds(self, rds_type_new):

        print("set new rds: {}".format(rds_type_new))
        self.L = np.zeros(
            (self.n_trial, 2, self.size_rds_bg_pix, self.size_rds_bg_pix), dtype=np.int32
        )
        self.R = np.zeros(
            (self.n_trial, 2, self.size_rds_bg_pix, self.size_rds_bg_pix), dtype=np.int32
        )
        if rds_type_new == "ards":
            # if disp_type=="crossed":
            temp = np.load("../../../Data/rds/ards_L_crossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_left_crossed_a.npy")
            self.L[:, 0] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds/ards_L_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_left_uncrossed_a.npy")
            self.L[:, 1] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds/ards_R_crossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_crossed_a.npy")
            self.R[:, 0] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds/ards_R_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_uncrossed_a.npy")
            self.R[:, 1] = temp[0 : self.n_trial]

        elif rds_type_new == "hmrds":
            # if disp_type=="crossed":
            temp = np.load("../../../Data/rds/hmrds_L_crossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_left_crossed_hm.npy")
            self.L[:, 0] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds/hmrds_L_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_left_uncrossed_hm.npy")
            self.L[:, 1] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds/hmrds_R_crossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_crossed_hm.npy")
            self.R[:, 0] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds/hmrds_R_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_uncrossed_hm.npy")
            self.R[:, 1] = temp[0 : self.n_trial]

        elif rds_type_new == "crds":
            # if disp_type=="crossed":
            temp = np.load("../../../Data/rds/crds_L_crossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_left_crossed_c.npy")
            self.L[:, 0] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds/crds_L_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_left_uncrossed_c.npy")
            self.L[:, 1] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds/crds_R_crossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_crossed_c.npy")
            self.R[:, 0] = temp[0 : self.n_trial]

            temp = np.load("../../../Data/rds/crds_R_uncrossed.npy")
            # temp = np.load("../../../Data/rds_small_pixel/rds_right_uncrossed_c.npy")
            self.R[:, 1] = temp[0 : self.n_trial]

        self.rds_type = rds_type_new

    def set_rds_disp_tuning(self, rds_type_new):
        """
        
        set rds for generating disparity tuning function

        Args:
            rds_type (str): type of rds: "ards", "hmrds", "crds".

        Returns:
            rds dimension:
            <[n_trial, len(disp_ct_deg), size_rds, size_rds] np.array>
            
            disp_ct_deg = np.linspace(-0.2, 0.2, 11).astype(np.float32) -> 11 disparity mag
            
            ex: [1000, 11, 120, 120]
            
        """

        print("set new rds: {}".format(rds_type_new))

        self.L = np.zeros(
            (self.n_trial, 21, self.size_rds_bg_pix, self.size_rds_bg_pix),
            dtype=np.int8,
        )
        self.R = np.zeros(
            (self.n_trial, 21, self.size_rds_bg_pix, self.size_rds_bg_pix),
            dtype=np.int8,
        )
        if rds_type_new == "ards":
            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_left_a.npy")
            self.L[:, :, 0] = temp[:, 0 : self.n_trial]

            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_right_a.npy")
            self.R[:, :, 0] = temp[:, 0 : self.n_trial]

        elif rds_type_new == "hmrds":
            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_left_hm.npy")
            self.L[:, :, 0] = temp[:, 0 : self.n_trial]

            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_right_hm.npy")
            self.R[:, :, 1] = temp[:, 0 : self.n_trial]

        elif rds_type_new == "crds":
            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_left_c.npy")
            self.L[:, :, 0] = temp[:, 0 : self.n_trial]

            temp = np.load("../../../Data/rds_small_pixel/disp_tuning/rds_right_c.npy")
            self.R[:, :, 1] = temp[:, 0 : self.n_trial]

        self.rds_type = rds_type_new
