import numpy as np
import pandas as pd

dataset=pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')
dataset=dataset.drop(['Product_Code','MIN','MAX'],axis='columns')

x=dataset.iloc[:,52:].values

from sklearn.cluster import KMeans
wcss=[]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
x=scaler.fit_transform(x)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(x)
x_pca=pca.fit_transform(x)

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x_pca)
    wcss.append(kmeans.inertia_)
    
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.plot(range(1,11),wcss)
plt.show()

kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x_pca)

plt.figure(figsize=(10,10))
plt.title('Clustering')
plt.scatter(x_pca[y_kmeans==0,0],x_pca[y_kmeans==0,1],c='red',s=100,label='Cluster1')
plt.scatter(x_pca[y_kmeans==1,0],x_pca[y_kmeans==1,1],s=100,c='green',label='Cluster2')
plt.scatter(x_pca[y_kmeans==2,0],x_pca[y_kmeans==2,1],s=100,c='magenta',label='Cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=400,c='yellow',label='Centroid')
plt.legend()
plt.show()
