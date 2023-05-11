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

def correlation_and_scattermatrix(data):
    '''
    correlation_and_scattermatrix plots correlation matrix and scatter plots
    of the data.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame for which analysis will be done.

    Returns
    -------
    None.

    '''
    corr = data.corr()
    print(corr)
    plt.figure(figsize=(8,8))
    plt.matshow(corr, cmap='gist_rainbow')

    # xticks and yticks for corr matrix
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns,rotation=0)
    plt.title('Correlation heat map of labour force')
    plt.colorbar()
    plt.show()

    pd.plotting.scatter_matrix(data, figsize=(10,10), s=5, alpha=1)
    plt.show()

    return


def cluster_number(data, data_normalised):
    '''
    cluster_number calculates the best number of clusters based on silhouette
    score

    Parameters
    ----------
    data : pandas.DataFrame
        Actual data.
    data_normalised : pandas.DataFrame
        Normalised data.

    Returns
    -------
    INT
        Best cluster number.

    '''

    clusters = []
    scores = []
    # loop over number of clusters
    for ncluster in range(2, 12):

        # Setting up clusters over number of clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Cluster fitting
        kmeans.fit(data_normalised)
        lab = kmeans.labels_

        # Silhoutte score over number of clusters
        print(ncluster, skmet.silhouette_score(data, lab))

        clusters.append(ncluster)
        scores.append(skmet.silhouette_score(data, lab))

    clusters = np.array(clusters)
    scores = np.array(scores)

    best_ncluster = clusters[scores == np.max(scores)]
    print()
    print('best n clusters', best_ncluster[0])

    return best_ncluster[0]


def clusters_and_centers(data, ncluster, y1, y2):
    '''
    clusters_and_centers will plot clusters and its centers for given data

    Parameters
    ----------
    data : pandas.DataFrame
        Data for which clusters and centers will be plotted.
    ncluster : INT
        Number of clusters.
    y1 : INT
        Column 1
    y2 : INT
        Column 2

    Returns
    -------
    data : pandas.DataFrame
        Data with cluster labels column added.
    cen : array
        Cluster Centers.

    '''
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(data)

    labels = kmeans.labels_
    data['labels'] = labels
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    cen = np.array(cen)
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    df1 = plt.cm.get_cmap('tab10')
    df2 = plt.scatter(data[y1], data[y2], 10, labels, marker="o", cmap=df1)
    plt.scatter(xcen, ycen, 45, "k", marker="s")
    plt.xlabel(f"labor force({y1})")
    plt.ylabel(f"labor force({y2})")
    plt.legend(*df2.legend_elements(), title='clusters')
    plt.title('Clusters of labor force in 1970 and 2020')
    plt.show()

    print()
    print(cen)

    return data, cen


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 
    and growth rate g"""

    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f

