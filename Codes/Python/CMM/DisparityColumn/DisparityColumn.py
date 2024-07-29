#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:44:19 2021

    cd /NVME/fmri_data_processing/bw18_005_2/Codes/Python/CMM/DisparityColumn
    
    script for generating disparity columnar structure.
    ref: Ban, nature neuroscience 2011

@author: cogni
"""

import numpy as np
from timeit import default_timer as timer
from scipy import signal
from joblib import Parallel, delayed

from numba import njit, prange


@njit
def _create_sawtooth(col_cycle, x):

    step = int(x.shape[0] / col_cycle)
    y = np.zeros(x.shape[0], dtype=np.float32)
    for i in range(col_cycle):

        id_start = i * step
        id_end = id_start + step

        x0 = x[id_start]
        y[id_start:id_end] = 6 * (x[id_start:id_end] - x0) - 1

    y[-1] = -1

    return y


@njit
def _create_dispColMap(
    col_cycle, delta_dispPref, dispColMap_amp, vox_res, nVox, std_sawtooth_noise
):
    """

    Create the distribution of disparity-selective column cells based
    on sawtooth distribution

    :param
        - dispColMap_amp: <scalar>, disparity amplitude of dispColMap
                            distribution, (ex: 0.4 degree)
        - delta_dispPref: <scalar>, # distance between two disparity
                        tuning in the column
                        For ex: delta_dispPref = 0.05
        - col_cycle: <scalar>, length of one cycle, mm/cycle.
                        For ex: 3 mm/cycle (Ban, nature neuroscience 2011, sup p.7)
        - vox_res: <scalar> voxel resolution in mm (ex: 2 mm)
        - nVox: <scalar>, number of voxels to construct column map

    :return:
        - delta_x: <scalar>, the distance (mm) from one disparity tuning to
                    the next within disparity column.
                essentially, it is the distance required to sample disparity
                tuning in the disparity column map

        - dispColMap: <[x, disparity_preference] np.array> disparity column map
                    representing the distribution of disparity preferrence
                    of disparity neurons in a columnar structure.
                The 1st column: the cell position in the column (mm),
                the 2nd column: disparity selectivity magnitude (deg).


    """

    n = 100
    samp_rate = col_cycle * n  # number of points per 1 mm

    # The disparity distance between each disparity tuning in the column is
    # determined to be delta_disp (deg)

    # calculate the distance in mm from one disparity tuning to the next.
    # self.delta_x = self.std_disp/(self.col_cycle* 2*self.dispColMap_amp) # mm
    delta_x = (delta_dispPref * col_cycle) / (2 * dispColMap_amp)  # mm

    # the number of points per delta_x -> It means that we get disparity preference every delta_point
    # delta_point = np.int32(delta_x * samp_rate)

    # generate 1 mm of sawtooth signal
    x = np.linspace(0, 1, samp_rate)
    # one_sawtooth = dispColMap_amp * signal.sawtooth(2*np.pi * col_cycle * x)
    one_sawtooth = dispColMap_amp * _create_sawtooth(col_cycle, x)
    # plt.plot(x, one_sawtooth)

    # calculate the total length (mm) of the disparity column
    adjust_length = col_cycle * n  # for randomness
    colMap_length = vox_res * nVox + adjust_length  # total length of column map, mm

    # generate sawtooth signal to fill the disparity column
    # sawtooth_all = np.tile(one_sawtooth, (colMap_length))
    sawtooth_all = (
        one_sawtooth.repeat(colMap_length).reshape((-1, colMap_length)).T.flatten()
    )
    # plt.plot(sawtooth_all[0:1200])

    # add gaussian noise
    noise = np.random.normal(0, std_sawtooth_noise, len(sawtooth_all))
    disp_real = sawtooth_all + noise
    # plt.plot(disp_real[0:1200])

    # make coordinate
    x = np.linspace(0, colMap_length, len(sawtooth_all))

    # make a matrix containing the coordinate and disparity preference in the column map
    dispColMap = np.hstack((x.reshape(-1, 1), disp_real.reshape(-1, 1)))

    return delta_x, dispColMap


@njit(parallel=True)
def _create_dispColMap_vox_bootstrap(
    col_cycle,
    delta_dispPref,
    dispColMap_amp,
    vox_res,
    nVox,
    std_sawtooth_noise,
    n_bootstrap,
):
    """
    create n_bootstrap of voxelized disparity column map.
    It creates n_bootstrap disparity column map

    Args:
        dispColMap_amp (TYPE): DESCRIPTION.
        std_disp (TYPE): DESCRIPTION.
        std_dispTuning_noise (TYPE): DESCRIPTION.
        col_cycle (TYPE): DESCRIPTION.
        vox_res (TYPE): DESCRIPTION.
        nVox (TYPE): DESCRIPTION.
        n_bootstrap (TYPE): DESCRIPTION.

    Returns:
        dispColMap_bootstrap ([n_bootstrap, nVox, neurons_per_vox] np.array):
                contains n_bootstrap of [nVox, neurons_per_vox] disparity column map
                obtained from fxCreate_voxelizedDispCol.

                neurons_per_vox here represents the number of neurons in a voxel
                of the disparity column map

                each value represents the magnitude of disparity
                selective neurons in the columns.


    """
    # n_bootstrap = 100

    delta_x, dispColMap = _create_dispColMap(
        col_cycle, delta_dispPref, dispColMap_amp, vox_res, nVox, std_sawtooth_noise
    )

    # find the distance from one neuron to the next in dispColMap associated with delta_x
    # d = np.abs(dispColMap[:, 0] - delta_x)
    # d_min = np.min(d)
    # delta_x_neuron = np.where(d==d_min)[0][0] -> not working in njit
    delta_x_neuron = 56

    neurons_per_vox = int(vox_res / delta_x)  # number of neurons per voxel
    dispColMap_bootstrap = np.zeros(
        (n_bootstrap, nVox, neurons_per_vox), dtype=np.float32
    )

    for i in prange(n_bootstrap):

        delta_x, dispColMap = _create_dispColMap(
            col_cycle, delta_dispPref, dispColMap_amp, vox_res, nVox, std_sawtooth_noise
        )
        # dispColMap_vox = _voxelize_dispColMap(delta_x, dispColMap)

        # voxel position jitter
        # vox_res = 2
        # std_vox_jitter = 0.25*vox_res # according to Kamitani & Tong, nature 2005

        # x = dispColMap[:, 0]
        y = dispColMap[:, 1]  # get the disparity selectivity of column cell (deg)

        # preallocate voxelized column cell
        # neurons_per_vox = np.int32(vox_res/delta_x) # number of neurons per voxel

        # dispColMap_vox = np.zeros((nVox, neurons_per_vox),
        #                           dtype=np.float32)
        # x_vox = np.zeros((self.nVox, neurons_per_vox),
        #                  dtype=np.float32)

        # id_sample_all = np.zeros((nVox, nTuning))

        for v in range(nVox):
            for n in range(neurons_per_vox):

                # jitter = 0
                # jitter = np.random.normal(0, self.std_vox_jitter)
                # jitter = np.random.choice([-1, 0, 1]) # jitter in step
                # if v==0:
                #     jitter = np.abs(jitter)

                id_sample = int((v * neurons_per_vox + n) * delta_x_neuron)
                # x_vox[v-1, t] = x_sample

                # id_sample_all[v-1, t] = id_sample
                # get disparity preference at id_sample
                # dispColMap_vox[v, n] = idx

                dispColMap_bootstrap[i, v, n] = y[id_sample]

    return dispColMap_bootstrap


# tik = timer()
# a = _create_dispColMap_vox_bootstrap(col_cycle, delta_dispPref, dispColMap_amp,
#                                       vox_res, nVox, std_sawtooth_noise,
#                                       n_bootstrap)
# tok = timer()
# print(tok-tik)

# @njit(parallel=True)
# def prange_ok_result_whole_arr(x):
#     n = x.shape[0]
#     y = np.zeros((4,4))

#     a = np.arange(6, 10)
#     for i in prange(4):
#         for j in range(4):
#             d = i*x[i] + a[j]
#             y[i, j] = d
#     return y


# prange_ok_result_whole_arr(x)


class DisparityColumn:
    """
    Create discparity column distribution

    ref: Ban, nature neuroscience 2011

    """

    def __init__(self, std_sawtooth_noise, n_bootstrap, nVox):

        self.dispColMap_amp = 0.4  # max disparity amplitude, degree (ori)
        # self.dispColMap_amp = 0.2  # max disparity amplitude, degree
        self.delta_dispPref = (
            0.05  # distance between two disparity tuning in the column, deg (ori)
        )
        # self.delta_dispPref = 0.02 # distance between two disparity tuning in the column, deg
        # self.std_sawtooth_noise = 0.05 # gaussian noise for jittering the sawtooth distribution
        self.std_sawtooth_noise = std_sawtooth_noise

        self.vox_res = 2  # voxel resolution in mm
        # self.std_vox_jitter = 0.25*self.vox_res # jitter voxel position due to head motion, according to Kamitani & Tong, nature 2005

        self.nVox = nVox  # number of voxel
        self.col_cycle = (
            3  # column cycle, 3 mm/cycle (Ban, nature neuroscience 2011, sup p.7)
        )
        self.n_bootstrap = n_bootstrap

    def create_dispColMap(self):
        """

        Create the distribution of disparity-selective column cells based
        on sawtooth distribution

        :param
            - dispColMap_amp: <scalar>, disparity amplitude of dispColMap
                                distribution, (ex: 0.4 degree)
            - delta_dispPref: <scalar>, # distance between two disparity
                            tuning in the column
                            For ex: delta_dispPref = 0.05
            - col_cycle: <scalar>, length of one cycle, mm/cycle.
                            For ex: 3 mm/cycle (Ban, nature neuroscience 2011, sup p.7)
            - vox_res: <scalar> voxel resolution in mm (ex: 2 mm)
            - nVox: <scalar>, number of voxels to construct column map

        :return:
            - delta_x: <scalar>, the distance (mm) from one disparity tuning to
                        the next within disparity column.
                    essentially, it is the distance required to sample disparity
                    tuning in the disparity column map

            - dispColMap: <[x, disparity_preference] np.array> disparity column map
                        representing the distribution of disparity preferrence
                        of disparity neurons in a columnar structure.
                    The 1st column: the cell position in the column (mm),
                    the 2nd column: disparity selectivity magnitude (deg).


        """

        n = 100
        samp_rate = self.col_cycle * n  # number of points per 1 mm

        # The disparity distance between each disparity tuning in the column is
        # determined to be delta_disp (deg)

        # calculate the distance in mm from one disparity tuning to the next.
        # self.delta_x = self.std_disp/(self.col_cycle* 2*self.dispColMap_amp) # mm
        delta_x = (self.delta_dispPref * self.col_cycle) / (
            2 * self.dispColMap_amp
        )  # mm

        # the number of points per delta_x -> It means that we get disparity preference every delta_point
        # delta_point = np.int32(delta_x * samp_rate)

        # generate 1 mm of sawtooth signal
        x = np.linspace(0, 1, samp_rate)
        one_sawtooth = self.dispColMap_amp * signal.sawtooth(
            2 * np.pi * self.col_cycle * x
        )
        # plt.plot(x, one_sawtooth)

        # calculate the total length (mm) of the disparity column
        adjust_length = self.col_cycle * n  # for randomness
        colMap_length = (
            self.vox_res * self.nVox + adjust_length
        )  # total length of column map, mm

        # generate sawtooth signal to fill the disparity column
        sawtooth_all = np.tile(one_sawtooth, (colMap_length))
        # plt.plot(sawtooth_all[0:1200])

        # add gaussian noise
        noise = np.random.normal(0, self.std_sawtooth_noise, len(sawtooth_all))
        disp_real = sawtooth_all + noise
        # plt.plot(disp_real[0:1200])

        # make coordinate
        x = np.linspace(0, colMap_length, len(sawtooth_all))

        # make a matrix containing the coordinate and disparity preference in the column map
        dispColMap = np.hstack((x.reshape(-1, 1), disp_real.reshape(-1, 1)))

        return delta_x, dispColMap

    def voxelize_dispColMap(self, delta_x, dispColMap):
        """
        voxelize the generated disparity column map obtained from create_dispColMap.
        The idea is to create a box to sample the column map and return the
        sampled column map in voxels

        :param:
            - dispColMap: <[x, disparity_preference] np.array> disparity column map
                        representing the distribution of disparity preferrence
                        of disparity neurons in a columnar structure.

                    The first column represents the cell position in mm, and the
                    second column represents disparity selectivity magnitude in deg.

            - nTuning: <scalar>, the number of sampling points per vox in position axis, sampPoint/vox.
                    essentially, it is the number of tuning function in each voxel
                    of disparity column map

            - nVox: <scalar>, the number of voxel needed to sample colMap

        :return:
            - dispColMap_vox: <[vox_number, disparity_preference] np.array>,
                            sampled dispColMap (voxelized disparity column).
                        Each row represents a group of disparity neuron whose
                        disparity preference magnitude

        """

        # np.random.seed(None)

        # voxel position jitter
        # vox_res = 2
        # std_vox_jitter = 0.25*vox_res # according to Kamitani & Tong, nature 2005

        x = dispColMap[:, 0]
        y = dispColMap[:, 1]  # get the disparity selectivity of column cell (deg)

        # preallocate voxelized column cell
        neurons_per_vox = int(self.vox_res / delta_x)  # number of neurons per voxel
        self.neurons_per_vox = neurons_per_vox

        dispColMap_vox = np.zeros((self.nVox, neurons_per_vox), dtype=np.float32)
        # x_vox = np.zeros((self.nVox, neurons_per_vox),
        #                  dtype=np.float32)

        # id_sample_all = np.zeros((nVox, nTuning))

        for v in range(1, self.nVox + 1):

            for t in range(neurons_per_vox):
                # jitter = 0
                # jitter = np.random.normal(0, self.std_vox_jitter)
                # jitter = np.random.choice([-1, 0, 1]) # jitter in step
                # if v==0:
                #     jitter = np.abs(jitter)

                # x_sample = (v*neurons_per_vox + t) * delta_x + jitter
                x_sample = (v * neurons_per_vox + t) * delta_x
                # x_vox[v-1, t] = x_sample

                # find the closest coordinate in dispColMap associated with x_sample
                d = np.abs(x - x_sample)
                id_sample = np.where(d == np.min(d))[0][0]

                # id_sample_all[v-1, t] = id_sample
                # get disparity preference at id_sample
                dispColMap_vox[v - 1, t] = y[id_sample]

        return dispColMap_vox

    def create_dispColMap_vox(self):
        """

        A wrapper to create voxelized disparity column map


        Args:
            dispColMap_amp (TYPE): DESCRIPTION.
            std_disp (TYPE): DESCRIPTION.
            std_dispTuning_noise (TYPE): DESCRIPTION.
            col_cycle (TYPE): DESCRIPTION.
            vox_res (TYPE): DESCRIPTION.
            nVox (TYPE): DESCRIPTION.

        Returns:
            dispColMap_vox ([nVox, nTuning]): voxelized disparity column map.
                nTuning here represents the number of tuning functions in a voxel
                of the disparity column map


        """

        delta_x, dispColMap = self.create_dispColMap()
        dispColMap_vox = self.voxelize_dispColMap(delta_x, dispColMap)

        return dispColMap_vox

    # def create_dispColMap_vox_bootstrap(self):
    #     '''
    #     create n_bootstrap of voxelized disparity column map.
    #     It creates n_bootstrap disparity column map

    #     Args:
    #         dispColMap_amp (TYPE): DESCRIPTION.
    #         std_disp (TYPE): DESCRIPTION.
    #         std_dispTuning_noise (TYPE): DESCRIPTION.
    #         col_cycle (TYPE): DESCRIPTION.
    #         vox_res (TYPE): DESCRIPTION.
    #         nVox (TYPE): DESCRIPTION.
    #         n_bootstrap (TYPE): DESCRIPTION.

    #     Returns:
    #         dispColMap_bootstrap ([n_bootstrap, nVox, neurons_per_vox] np.array):
    #                 contains n_bootstrap of [nVox, neurons_per_vox] disparity column map
    #                 obtained from fxCreate_voxelizedDispCol.

    #                 neurons_per_vox here represents the number of neurons in a voxel
    #                 of the disparity column map

    #                 each value represents the magnitude of disparity
    #                 selective neurons in the columns.

    #     '''
    #     # n_bootstrap = 100

    #     temp = []
    #     t_start = timer()
    #     temp.append(Parallel(n_jobs=-1)
    #                 (delayed(self.create_dispColMap_vox)
    #                   ()
    #                   for i in range(self.n_bootstrap)))

    #     t_end = timer()
    #     print(t_end - t_start)

    #     # unpack
    #     neurons_per_vox = temp[0][0].shape[1]
    #     self.neurons_per_vox = neurons_per_vox
    #     dispColMap_bootstrap = np.zeros((self.n_bootstrap, self.nVox, neurons_per_vox),
    #                                     dtype=np.float32)
    #     for i in range(self.n_bootstrap):
    #         dispColMap_bootstrap[i] = temp[0][i]

    #     return dispColMap_bootstrap

    def create_dispColMap_vox_bootstrap(self):
        """
        create n_bootstrap of voxelized disparity column map.
        It creates n_bootstrap disparity column map

        Args:
            dispColMap_amp (TYPE): DESCRIPTION.
            std_disp (TYPE): DESCRIPTION.
            std_dispTuning_noise (TYPE): DESCRIPTION.
            col_cycle (TYPE): DESCRIPTION.
            vox_res (TYPE): DESCRIPTION.
            nVox (TYPE): DESCRIPTION.
            n_bootstrap (TYPE): DESCRIPTION.

        Returns:
            dispColMap_bootstrap ([n_bootstrap, nVox, neurons_per_vox] np.array):
                    contains n_bootstrap of [nVox, neurons_per_vox] disparity column map
                    obtained from fxCreate_voxelizedDispCol.

                    neurons_per_vox here represents the number of neurons in a voxel
                    of the disparity column map

                    each value represents the magnitude of disparity
                    selective neurons in the columns.


        """
        # n_bootstrap = 100

        dispColMap_bootstrap = _create_dispColMap_vox_bootstrap(
            self.col_cycle,
            self.delta_dispPref,
            self.dispColMap_amp,
            self.vox_res,
            self.nVox,
            self.std_sawtooth_noise,
            self.n_bootstrap,
        )

        return dispColMap_bootstrap
