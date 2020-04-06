import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('College.csv')

#Label Encoder
from sklearn.preprocessing import LabelEncoder
lbl_private=LabelEncoder()
dataset['Private']=lbl_private.fit_transform(dataset['Private'])

#Feature Engineering
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(dataset)
scaled_data=scaler.fit_transform(dataset)

#Principal Component Analysis to shrink our features
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)

#Implementing kmeans to calculate wcss
from sklearn.cluster import KMeans
wcss=[]

for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x_pca)
    wcss.append(kmeans.inertia_)

#plotting the elbow method results
plt.plot(range(1,10),wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

#now as n_cluster=3, so implementing kmeans according to that
kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x_pca)

#plotting the three clusters and there respective centroids
plt.scatter(x_pca[y_kmeans==0,0],x_pca[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(x_pca[y_kmeans==1,0],x_pca[y_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(x_pca[y_kmeans==2,0],x_pca[y_kmeans==2,1],s=100,c='cyan',label='Cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='magenta',label='Centroids')
plt.title('Cluster of College')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
