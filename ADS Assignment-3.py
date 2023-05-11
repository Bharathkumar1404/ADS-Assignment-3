# -*- coding: utf-8 -*-
"""
Created on Fri May 12 02:47:21 2023

@author: SSD
"""


#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import cluster_tools as ct
import errors as err
import importlib


def read_data(file):
    '''
    read_data willread the data create dataframe from the given file

    Parameters
    ----------
    file : STR
        File or location.

    Returns
    -------
    data : pandas.DataFrame
        DataFrame created from given file.

    '''
    data = pd.read_csv(file, skiprows=4)
    data = data.set_index('Country Name', drop=True)
    data = data.loc[:, '1990':'2021']

    return data


def transpose_data(data):
    '''
    transpose_data it will create transpose of given dataframe.

    Parameters
    ----------
    data  : pandas.DataFrame
        DataFrame for which transpose to be found.

    Returns
    -------
    data_tr : pandas.DataFrame
        Transposed DataFrame of given DataFrame.

    '''
    data_tr = data.transpose()

    return data_tr
