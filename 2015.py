import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('2015.csv')
dataset=pd.DataFrame(dataset.iloc[:,3:].values)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(dataset)
x=scaler.fit_transform(dataset)

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,7))
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS')
plt.plot(range(1,11),wcss)
plt.show()

kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x)

plt.figure(figsize=(10,7))
plt.title('Clustering')
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],c='red',s=100,label='Cluster1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],c='green',s=100,label='Cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],c='blue',s=100,label='Cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow',s=400,label='Centroids')
plt.legend()
plt.show()
