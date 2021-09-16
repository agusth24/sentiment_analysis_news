# -*- coding: utf-8 -*-
"""
@Title: Clustering Stock on based return and volatile
@author: Agus Tri Haryono
@Description: Clustring Indonesia stock on based return and volatile.
"""

# download raw data
import requests
import pandas
import pathlib

def donwload_url(url,location=''):
    print ("download start! ", url)
    response = requests.get(url)
    if response.ok:
        file = open(location, "wb+") # write, binary, allow creation
        file.write(response.content)
        file.close()
        print ("download file location: ", location)
    else:
        print("Failed to get the file")

DownloadLocation = str(pathlib.Path().absolute())

# download all emiten csv
url="https://raw.githubusercontent.com/wildangunawan/Dataset-Saham-IDX/master/List Emiten/all.csv"
donwload_url(url,DownloadLocation + '/all.csv')

df = pandas.read_csv('all.csv')
print(df)

for index,row in df.iterrows():
    print(row['code'])
    url = "https://raw.githubusercontent.com/wildangunawan/Dataset-Saham-IDX/master/Saham/Semua/" + row['code'] + ".csv"
    donwload_url(url,DownloadLocation + '/raw/' + row['code'] + ".csv")
    
# prepare dataset YTD return and YTD Volatile
import pandas as pd
import numpy as np
import pathlib
import math
stocks = []    
DownloadLocation = str(pathlib.Path().absolute())

#row['code'] = "NICL"
#emiten = pd.read_csv(DownloadLocation + "/raw/" + row['code'] + ".csv")
#mask = (emiten['date'].isna() | emiten['previous'].isna() | emiten['close'].isna() | emiten['change'].isna()) | (emiten['date'].str.len() < 10)
#emiten = emiten.drop(emiten[mask].index)
#emiten['code'] = row['code']
#emiten['date'] = pd.to_datetime(emiten['date'])
#emiten['periode'] = emiten['date'].dt.strftime('%Y%m')
#emiten['tahun'] = emiten['date'].dt.strftime('%Y')
#emiten = emiten[emiten['tahun'] == '2021'].filter(['code','date','periode','tahun','previous','close','change'])
#emiten['return'] = np.log(emiten['close']/emiten['previous']) # return stock per day
#m_return = emiten['return'].mean() # mean / average return
#emiten['d_return'] = (emiten['return'] - m_return) ** 2 # deviation return
#volatile = math.sqrt(emiten['d_return'].sum()/(len(emiten.index)-1)) # volatile return
#emiten = [row['code'],emiten['return'].mean(),volatile]
jum_data = []
jum_data_filter = []
emitens = pd.read_csv(DownloadLocation + '/all.csv')
for index,row in emitens.iterrows():
    emiten = pd.read_csv(DownloadLocation + "/raw/" + row['code'] + ".csv")
    jum_data.append(emiten)
    mask = (emiten['date'].isna() | emiten['previous'].isna() | emiten['close'].isna() | emiten['change'].isna()) | (emiten['date'].str.len() < 10)
    emiten = emiten.drop(emiten[mask].index)
    emiten['code'] = row['code']
    emiten['date'] = pd.to_datetime(emiten['date'])
    emiten['periode'] = emiten['date'].dt.strftime('%Y%m')
    emiten['tahun'] = emiten['date'].dt.strftime('%Y')
    emiten_filter = emiten[emiten['tahun'] == '2021'].filter(['code','date','periode','tahun','previous','close','change'])
    jum_data_filter.append(emiten_filter)
    emiten_filter['return'] = np.log(emiten_filter['close']/emiten_filter['previous']) # return stock per day
    m_return = emiten_filter['return'].mean() # mean / average return
    emiten_filter['d_return'] = (emiten_filter['return'] - m_return) ** 2 # deviation return
    volatile = math.sqrt(emiten_filter['d_return'].sum()/(len(emiten_filter.index)-1)) # volatile return
    emiten_d = [row['code'],m_return,volatile]
    stocks.append(emiten_d)
emiten_s = pd.DataFrame(stocks, columns = ['code', 'mean_ytd_return', 'volatile_ytd_return'])
mask = emiten_s['mean_ytd_return'].isna() | emiten_s['volatile_ytd_return'].isna()
emiten_s = emiten_s.drop(emiten_s[mask].index)
emiten_s.to_csv(DownloadLocation + '/dataset.csv')

jumdata = pd.concat(jum_data)
jumdata_filter = pd.concat(jum_data_filter)

# clustering K-Means
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# plot first data before clustering
emiten_s[emiten_s['volatile_ytd_return'].isna()].index
plt.scatter(emiten_s['mean_ytd_return'], emiten_s['volatile_ytd_return'], marker='.')
plt.title('Emiten Stock Return and Volatile')
plt.xlabel('Mean Return')
plt.ylabel('Volatile')
plt.show()

# search k with sum squared error, elbow method
sse_kmeans = []
X = emiten_s.filter(['mean_ytd_return','volatile_ytd_return'])
for i in range(1,11): 
     kmeans = KMeans(n_clusters=i, init ='k-means++')
     kmeans.fit(X)
     sse_kmeans.append(kmeans.inertia_)
plt.plot(range(1,11),sse_kmeans)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

k_means = KMeans(init = "k-means++", n_clusters = 4)
k_means.fit(X)
k_means_labels = k_means.labels_
emiten_s["clus_km"] = k_means_labels
k_mean_mean = emiten_s.groupby('clus_km').mean()
k_mean_count = emiten_s.groupby('clus_km').count()

centers = k_means.cluster_centers_
labels = np.unique(k_means_labels)

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(10, 6))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
for k, col in zip(labels, colors):
    # Define the centroid, or cluster center.
    cluster_center = centers[k]
    
    # Plots the datapoints with color col.
    plt.plot(emiten_s[emiten_s['clus_km']==k]['mean_ytd_return'], emiten_s[emiten_s['clus_km']==k]['volatile_ytd_return'], '2', c=col)
    
    # Plots the centroids with specified color, but with a darker outline
    # plt.plot(cluster_center[0], cluster_center[1], '*', c=col,  markeredgecolor='k', markersize=8)
    plt.text(cluster_center[0], cluster_center[1], str(k), color="green", fontsize=12)

plt.title('KMeans')
plt.xlabel('Mean Return')
plt.ylabel('Volatile')
plt.show()

# clustering Agglomerative 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy 
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import silhouette_score

emiten_hir = emiten_s[emiten_s['clus_km']==1].reset_index(drop=True)
plt.scatter(emiten_hir['mean_ytd_return'], emiten_hir['volatile_ytd_return'], marker='.')
plt.title('Emiten Stock Return and Volatile')
plt.xlabel('Mean Return')
plt.ylabel('Volatile')
plt.show()

# dendogram
Y = emiten_hir.filter(['mean_ytd_return','volatile_ytd_return'])
fig = plt.figure(figsize=(15, 10))
dendrogram = hierarchy.dendrogram(hierarchy.linkage(Y, method  = "average"))
plt.title('Dendrogram')
plt.xlabel('Emiten')
plt.ylabel('Euclidean distances')
plt.show()

# run agglomerative clustering
agglom = AgglomerativeClustering(n_clusters = 3, linkage = 'average', affinity='euclidean')
result = agglom.fit_predict(Y)
cluster_labels = agglom.labels_
labels_agglo = np.unique(cluster_labels)

# calculate silhoutte evaluation
clf = NearestCentroid()
clf.fit(Y, result)
print("Centroids:")
print(clf.centroids_)
cluster_center_agglo = clf.centroids_
silhouette_avg = silhouette_score(Y, cluster_labels)

emiten_hir["agglo_res"] = result
agglom_mean = emiten_hir.groupby('agglo_res').mean()
agglom_count = emiten_hir.groupby('agglo_res').count()

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(10, 6))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(cluster_labels))))
for k, col in zip(labels_agglo, colors):
    # Define the centroid, or cluster center.
    cluster_center = cluster_center_agglo[k]
    
    # Plots the datapoints with color col.
    plt.plot(emiten_hir[emiten_hir['agglo_res']==k]['mean_ytd_return'], emiten_hir[emiten_hir['agglo_res']==k]['volatile_ytd_return'], '.', c=col)
    
    # Plots the centroids with specified color, but with a darker outline
    # plt.plot(cluster_center[0], cluster_center[1], '*', c=col,  markeredgecolor='k', markersize=8)
    plt.text(cluster_center[0], cluster_center[1], str(k), color="green", fontsize=12)

plt.title('Agglomerative')
plt.xlabel('Mean Return')
plt.ylabel('Volatile')
plt.show()

# evaluation graph silouhette
sse_agglo = []
for i in range(2,11): 
    agglom = AgglomerativeClustering(n_clusters = i, linkage = 'average', affinity='euclidean')
    result = agglom.fit_predict(Y)
    cluster_labels = agglom.labels_
    labels_agglo = np.unique(cluster_labels)
    
    clf = NearestCentroid()
    clf.fit(Y, result)
    print("Centroids:")
    print(clf.centroids_)
    cluster_center_agglo = clf.centroids_
    silhouette_avg = silhouette_score(Y, cluster_labels)
    sse_agglo.append(silhouette_avg)
    
plt.plot(range(2,11),sse_agglo)
plt.title('The Silhouette Graph')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()