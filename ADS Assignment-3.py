# -*- coding: utf-8 -*-

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
    read_data it will reads the data and create dataframe from the given file.

    Parameters
    ----------
    filepath : STR
        File or location.

    Returns
    -------
    data : pandas.DataFrame
        DataFrame created from given file.

    '''
    data = pd.read_csv(file, skiprows=4)
    data = data.set_index('Country Name', drop=True)
    data = data.loc[:, '1970':'2008']

    return data


def transpose_data(data):
    '''
    transpose_data it will create transpose of given dataframe

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
    of coloumns in the given data.

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
    plt.figure(figsize=(10,10))
    plt.matshow(corr, cmap='gist_rainbow')

    # xticks and yticks for corr matrix
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns,rotation=0)
    plt.title('Correlation over Agriculture methane emission')
    plt.colorbar()
    plt.show()

    pd.plotting.scatter_matrix(data, figsize=(12,12), s=5, alpha=1)
    plt.show()

    return


def cluster_num(data, data_normalised):
    '''
    cluster_number evalutes the best number of clusters based on silhouette
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
    for ncluster in range(2, 15):

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
    clusters_and_centers will show the plot of clusters and its centers 
    for given data

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

    cm = plt.cm.get_cmap('tab10')
    sc = plt.scatter(data[y1], data[y2], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="s")

    plt.xlabel(f"Methane emission({y1})")
    plt.ylabel(f"Methane emission({y2})")
    plt.legend(*sc.legend_elements(), title='clusters')
    plt.title('Clusters of Countries over Agriculture methane 1970 and 2008')
    plt.show()

    print()
    print(cen)

    return data, cen


def logistic_value(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 
    and growth rate g"""

    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f


def forecast(data, country, start_year, end_year):
    '''
    forecast will analyse data and optimize to forecast emission of selected 
    country

    Parameters
    ----------
    data : pandas.DataFrame
        Data for which forecasting analysis is performed.
    country : STR
        Country for which forecasting is performed.
    start_year : INT
        Starting year for forecasting.
    end_year : INT
        Ending year for forecasting.

    Returns
    -------
    None.

    '''
    data = data.loc[:, country]
    data = data.dropna(axis=0)

    data_emission = pd.DataFrame()

    data_emission['Year'] = pd.DataFrame(data.index)
    data_emission['METHANE'] = pd.DataFrame(data.values)
    data_emission["Year"] = pd.to_numeric(data_emission["Year"])
    importlib.reload(opt)

    param, covar = opt.curve_fit(logistic_value, data_emission["Year"], 
                   data_emission["METHANE"],p0=(1.2e12, 0.03, 1970.0)
                   ,maxfev=10000)

    sigma = np.sqrt(np.diag(covar))

    year = np.arange(start_year, end_year)
    forecast_emission = logistic_value(year, *param)
    low, up = err.err_ranges(year, logistic_value, param, sigma)
    plt.figure()
    plt.plot(data_emission["Year"], data_emission["METHANE"], label="METHANE")
    plt.plot(year, forecast_emission, label="forecast", color='k')
    plt.fill_between(year, low, forecast_emission, color="pink", alpha=0.7)
    plt.fill_between(year, forecast_emission, up, color="pink", alpha=0.7)
    plt.xlabel("year")
    plt.ylabel("Methane emission (kilo metric tons)")
    plt.legend(loc='upper right')
    plt.title(f'Methane forecast_emission for {country}')
    plt.savefig(f'{country}.png', bbox_inches='tight', dpi=300)
    plt.show()

    emission2030 = logistic_value(2030, *param)/10000

    low, up = err.err_ranges(2030, logistic_value, param, sigma)
    sig = np.abs(up-low)/(2.0 * 10000)
    print()
    print("Emission 2030", emission2030*10000, "+/-", sig*10000)


#Reading GDP per capita Data
methane_emission = read_data("Agricultural methane emissions.csv")
print(methane_emission.describe())

#Finding transpose of GDP per capita Data
methane_emission_tr = transpose_data(methane_emission)
print(methane_emission_tr.head())

#Selecting years for which correlation is done for further analysis
methane_emission_years = methane_emission[[ "1970","1975","1980","1985","1990"
                                           ,"2000", "2008"]]
print(methane_emission_years.describe())

correlation_and_scattermatrix(methane_emission_years)
year1 = "1970"
year2 = "2008"

# Extracting columns for clustering
emission_clustering = methane_emission_years[[year1, year2]]
emission_clustering = emission_clustering.dropna()

# Normalising data and storing minimum and maximum
df_norm, df_min, df_max = ct.scaler(emission_clustering)

print()
print("Number of Clusters and Scores")
ncluster = cluster_num(emission_clustering, df_norm)

df_norm, cen = clusters_and_centers(df_norm, ncluster, year1, year2)

#Applying backscaling to get actual cluster centers
scen = ct.backscale(cen, df_min, df_max)
print('scen\n', scen) 

emission_clustering, scen = clusters_and_centers(emission_clustering, 
                            ncluster, year1, year2)

'''
We can see some difference in actual cluster centers and 
backscaled cluster centers.
'''

print()
print('Countries in last cluster')
print(emission_clustering[emission_clustering['labels'] == ncluster-1].
      index.values)
