import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('cricket.csv')
dataset=dataset.drop(['PLAYER'],axis='columns')
dataset=dataset.replace(to_replace='-',value='0')

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(dataset)
x=scaler.fit_transform(dataset)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(x)
x=pca.transform(x)

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,7))
plt.plot(range(1,11),wcss)
plt.show()

kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x)

plt.figure(figsize=(10,7))
plt.title('Clusters')
plt.xlabel('X1')
plt.ylabel('Y1')
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='yellow',label='Cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='Cluster3')
#plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='blue',label='Cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=400,c='magenta',label='Centroid')
plt.legend()
plt.show()

