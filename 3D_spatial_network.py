import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('3D_spatial_network.csv')

from sklearn.preprocessing import StandardScaler,normalize
scaler=StandardScaler()
scaler.fit(dataset)
x=scaler.fit_transform(dataset)
x=normalize(x)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(x)
x_pca=pca.transform(x)

from sklearn.cluster import KMeans
k=wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x_pca)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10,7))
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.plot(range(1,11),wcss)

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x_pca)

plt.figure(figsize=(10,7))
plt.title('Clsutering')
plt.scatter(x_pca[y_kmeans==0,0],x_pca[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(x_pca[y_kmeans==1,0],x_pca[y_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(x_pca[y_kmeans==2,0],x_pca[y_kmeans==2,1],s=100,c='green',label='Cluster3')
plt.scatter(x_pca[y_kmeans==3,0],x_pca[y_kmeans==3,1],s=100,c='magenta',label='Cluster4')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=400,c='yellow',label='Centroid')
plt.legend()
plt.show()
