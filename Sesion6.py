#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:46:04 2020

@author: isaaclera
"""

from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, centroid



## GENERAMOS DATOS ALEATORIOS PARA DIFERENTES SERIES
x = np.random.randint(low=3,high=100, size=40)
print("X: %s"%x)

y = np.random.randint(low=3,high=100, size=40)
print("Y: %s"%y)
print(x[0])
##
# =============================================================================
# DATOS similares valores y MISMAS UNIDADES 
# =============================================================================

# En este caso ambas series X,Y podrían tener similares valores y MISMAS UNIDADES
Data = {'x': x,
        'y': y
       }
  
df = DataFrame(Data,columns=['x','y'])
  



kmeans = KMeans(n_clusters=17).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

# =============================================================================
# ¿PERO REALMENTE 4 grupos son los suficienteS?
# DENDROGRAMA!!!
# =============================================================================
#https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html

centr = centroid(df)
labelList = range(len(x))

plt.figure(figsize=(10, 7))
dendrogram(centr,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Dendrograma using centroid method')
plt.show()


# =============================================================================
# ALERTA! Hay más técnicas de clustering
# =============================================================================

linked = linkage(df, 'single')
labelList = range(len(x))
plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Dendrograma using linkage method')
plt.show()


# =============================================================================
# DATOS Con diferentes unidades --- SIN NORMALIZAR
# =============================================================================
# Cambiamos los valores de un eje 

y = np.random.uniform(low=0,high=2,size=40)
print("Y: %s"%y)


Data = {'x': x,
        'y': y
       }
  
df = DataFrame(Data,columns=['x','y'])
  
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.title('Clustering without normalization')
plt.show()


# =============================================================================
# DATOS Con diferentes unidades --- NORMALIZANDO POR RANGO
# =============================================================================
xr = (x-x.min())/(float(x.max()-x.min()))
yr = (y-y.min())/(y.max()-y.min())
print("Xr: %s"%xr)
print("Yr: %s"%yr)

Data = {'x': xr,
        'y': yr
       }
  
df = DataFrame(Data,columns=['x','y'])
  
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.title('Clustering using range normalization')
plt.show()


# =============================================================================
# DATOS Con diferentes unidades --- NORMALIZANDO POR Z-SCORE
# =============================================================================
xz = stats.zscore(x)
yz = stats.zscore(y)
print("Xz: %s"%xz)
print("Yz: %s"%yz)

Data = {'x': xz,
        'y': yz
       }
  
df = DataFrame(Data,columns=['x','y'])
  
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.title('Clustering using Z-score normalization')
plt.show()

