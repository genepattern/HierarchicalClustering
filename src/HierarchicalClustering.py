"""
Created on 2017-08-15 by Edwin F. Juarez.

This module will grab a .gct file to perform hierarchical clustering on the columns.

Pre-release. This module should only be used for testing purposes.
"""

print("Disclaimer: This is a pre-release version.")
print("This module should only be used for testing purposes.")

import os
import sys
tasklib_path = os.path.dirname(os.path.realpath(sys.argv[0]))
# sys.path.append(tasklib_path + "/ccalnoir")
import matplotlib as mpl
mpl.use('Agg')
# import pandas as pd
# import numpy as np
import scipy
import seaborn as sns
# from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
# from time import time
# import cuzcatlan as cusca
sns.set_style("white")
from hc_functions import *

gct_name, distance_metric, output_distances, cluster_by_rows = parse_inputs(sys.argv)
data, data_df, plot_labels, row_lables = parse_data(gct_name)
model = AgglomerativeClustering(linkage='average', n_clusters=2, affinity=str2func[distance_metric])
model.fit(data)

fig = plt.figure(dpi=300)
order_of_columns = plot_dendrogram(model, dist=str2dist[distance_metric], labels=plot_labels)

# scipy.cluster.hierarchy.linkage(col_distance_matrix, method='average')
# plt.clf()
# sns.clustermap(data_df[order_of_columns], metric=cusca.mydist, cmap="viridis", method='average')
# plt.savefig('clustermap.png')

# Creating outputs.
print("Creating outputs now.")
plt.savefig('sample_cluster.png', dpi=300, bbox_inches='tight')

cusca.list2cls(model.labels_, name_of_out='column_labels.cls')

order_of_rows = data_df.index.values
# Independent clustering by rows
if cluster_by_rows:
    model_T = AgglomerativeClustering(linkage='average', n_clusters=2, affinity=str2func[distance_metric])
    model_T.fit(np.transpose(data))
    fig.clf()
    # order_of_rows = plot_dendrogram(row_model, dist=str2dist[distance_metric], labels=row_lables)
    order_of_rows = two_plot_two_dendrogram(model_T, dist=str2dist[distance_metric], labels=row_lables)
    # plt.savefig('sample_cluster.png', dpi=300, bbox_inches='tight')
    # two_plot_two_dendrogram(model, top=int(np.floor(len(data_df)/2)), col_order=order_of_columns)
    plt.savefig('sample_cluster_2.png', dpi=300, bbox_inches='tight')
    cusca.list2cls(model_T.labels_, name_of_out='row_labels.cls')

plot_heatmap(data_df, top=int(np.floor(len(data_df)/2)), col_order=order_of_columns, row_order=order_of_rows)


if output_distances:
    row_distance_matrix = str2affinity_func[distance_metric](data)
    # col_distance_matrix = str2affinity_func[distance_metric](np.transpose(data))
    dist_file = open('pairwise_distances.csv', 'w')
    dist_file.write('labels,')
    dist_file.write(",".join(model.labels_.astype(str))+"\n")
    dist_file.write('samples,')
    dist_file.write(",".join(list(data_df))+"\n")
    i = 0
    for row in row_distance_matrix:
        dist_file.write('distances row='+str(i)+","+",".join(row.astype(str)) + "\n")
        i += 1
