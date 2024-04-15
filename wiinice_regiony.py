#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:10:45 2023

@author: patryk
"""
import geopandas as gpd
import libpysal as ps
from esda.moran import Moran_Local
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from shapely.geometry import MultiPolygon

gminy = gpd.read_file('shp/gminy.shp')

w_weights = ps.weights.Queen.from_dataframe(gminy)
moran_loc = Moran_Local(gminy['winnice'], w_weights)

# Local Moran's I value
local_moran_i = moran_loc.Is
gminy['Local_Moran_I'] = local_moran_i

# Create a function to classify values into four categories
def classify_moran_i(value, threshold):
    if value >= threshold:
        return 'HH'
    else:
        return 'LL'

# Define a custom threshold for HIGH values
hh_threshold = 2

# Create a new column in the GeoDataFrame with the classification
gminy['Moran_Class'] = [classify_moran_i(i, hh_threshold) for i in local_moran_i]

# Update classification for LH and HL based on Moran's I values
for i in range(len(gminy)):
    if local_moran_i[i] < 0 and gminy['Moran_Class'][i] == 'HH':
        gminy['Moran_Class'][i] = 'LH'
    elif local_moran_i[i] < 0 and gminy['Moran_Class'][i] == 'LL':
        gminy['Moran_Class'][i] = 'HL'

# Create a choropleth map of the classification
fig, ax = plt.subplots(figsize=(8, 8))
gminy.plot(column='Moran_Class', cmap='coolwarm', ax=ax, legend=True)
ax.set_title('Local Moran\'s I Classification (Threshold = {})'.format(hh_threshold))

plt.show()

##################################

gminy_hh = gminy[gminy['Moran_Class'] == 'HH']

# Create a GeoDataFrame with just the centroids
gminy_hh['Centroid'] = gminy_hh.centroid
centroidy = gminy_hh.copy()
centroidy = centroidy.set_geometry('Centroid')

# Extract the centroid coordinates
coords = centroidy.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()

num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=0)

# Fit the model results to the centroid coordinates
centroidy['KMeans_Cluster'] = kmeans.fit_predict(coords)

wojewodztwa = gpd.read_file('shp/wojewodztwa.shp')

fig, ax = plt.subplots(figsize=(8, 8))
wojewodztwa.plot(ax=ax, color='lightgray', edgecolor='black')
centroidy.plot(column='KMeans_Cluster', cmap='viridis', ax=ax, legend=True)

ax.set_title('K-Means Clustering - Klastry gmin o liczbie winnic >= 2')

# Add labels for each cluster
for cluster_id in range(num_clusters):
    cluster_centroid = centroidy[centroidy['KMeans_Cluster'] == cluster_id].geometry.centroid.iloc[0]
    ax.annotate(f'Cluster {cluster_id}', xy=(cluster_centroid.x, cluster_centroid.y), color='black', fontsize=12)

plt.show()


##################################

gminy_hh['KMeans_Cluster'] = centroidy['KMeans_Cluster']

fig, ax = plt.subplots(figsize=(12, 10))  
wojewodztwa.plot(ax=ax, color='lightgray', edgecolor='black')

unique_clusters = np.sort(gminy_hh['KMeans_Cluster'].unique())

# Define custom cluster names and colors
cluster_info = {
    0: {'name': 'Podkarpacie', 'color': 'tab:orange'},
    1: {'name': 'Lubuskie', 'color': 'tab:purple'},
    2: {'name': 'Dolny Śląsk', 'color': 'tab:green'},
    3: {'name': 'Małopolski Przełim Wisły & Sandomierz', 'color': 'tab:pink'},
    4: {'name': 'Małopolska', 'color': 'tab:red'},
    5: {'name': 'Zachodnie Pomorze', 'color': 'tab:brown'},
}

# Map cluster labels to the colors used in the plot
cluster_colors_dict = {label: info['color'] for label, info in cluster_info.items()}

# Plot clusters with specified colors
cluster_plot = gminy_hh.plot(column='KMeans_Cluster', ax=ax, legend=False, color=[cluster_colors_dict[c] for c in gminy_hh['KMeans_Cluster']])

# Create a custom legend with corresponding colors
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=info['color'], markersize=10, label=f"{info['name']}")
                   for label, info in cluster_info.items()]

ax.legend(handles=legend_elements, loc='lower left', frameon=False, fontsize='large')

ax.set_title('Polskie klastry winiarskie', size=30)
arrow = FancyArrowPatch((0.92, 0.92), (0.92, 0.97), color='black', arrowstyle='->', mutation_scale=30, transform=ax.transAxes)
ax.add_patch(arrow)

plt.show()

##################################

# Map cluster labels to cluster names
gminy_hh['Cluster_Name'] = gminy_hh['KMeans_Cluster'].map({label: info['name'] for label, info in cluster_info.items()})

# Specify the path where you want to save the new shapefile
output_shapefile_path = 'regiony_wino_pl.shp'

if 'Cluster_Name' in gminy_hh.columns:
    # Dissolve based on 'Cluster_Name' and save the GeoDataFrame to a new shapefile
    regiony = gminy_hh.dissolve(by='Cluster_Name', aggfunc='first')

    # Reset the index to make 'Cluster_Name' a regular column
    regiony = regiony.reset_index()

    regiony[['KMeans_Cluster', 'Cluster_Name', 'geometry']].to_file(output_shapefile_path, encoding='utf-8')
else:
    print("Error: 'Cluster_Name' column not found in GeoDataFrame.")

##################################

output_csv_path = 'regiony_wino_pl.csv'

gminy_hh[['JPT_NAZWA_', 'JPT_KOD_JE', 'Cluster_Name']].to_csv(output_csv_path, encoding='utf-8', index=False)


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Data
X = coords

# Range of clusters to test
k_values = range(2, 21)
silhouette_scores = []

# Calculate silhouette score for each value of k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotting
plt.plot(range(2, 21), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')
plt.xticks(range(2, 21))  # Adjust x-axis ticks to start from 2
plt.show()
