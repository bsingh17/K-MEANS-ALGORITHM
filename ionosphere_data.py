import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('ionosphere_data.csv')

from sklearn.preprocessing import LabelEncoder
lbl_a=LabelEncoder()
dataset['column_a']=lbl_a.fit_transform(dataset['column_a'])
lbl_b=LabelEncoder()
dataset['column_b']=lbl_b.fit_transform(dataset['column_b'])
lbl_ai=LabelEncoder()
dataset['column_ai']=lbl_ai.fit_transform(dataset['column_ai'])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(dataset)
scaled_data=scaler.fit_transform(dataset)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x_pca)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number Of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x_pca)

plt.scatter(x_pca[y_kmeans==0,0],x_pca[y_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(x_pca[y_kmeans==1,0],x_pca[y_kmeans==1,1],s=100,c='cyan',label='Cluster2')
plt.scatter(x_pca[y_kmeans==2,0],x_pca[y_kmeans==2,1],s=100,c='green',label='Cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroid')
plt.title('Cluster of Ionosphere')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()