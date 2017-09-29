"""
Created on 2017-08-15 by Edwin F. Juarez.

This module will grab a .gct file to perform hierarchical clustering on the columns.

Pre-release. This module should only be used for testing purposes.
"""

print("Disclaimer: This is a pre-release version.")
print("This module should only be used for testing purposes.")

print("*Expecto installer*")
import pip

def install(package):
    pip.main(['install', package])

# # Example
# install('sklearn')
# print("[a beautifull installer appeared]")
# print("...sklearn installed successfully!")
# print("Trying to install sklearn again.")
# install('cuzcatlan')


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

gct_name, distance_metric, output_distances, row_distance_metric, clustering_method, output_base_name = parse_inputs(sys.argv)
data, data_df, plot_labels, row_lables, full_gct = parse_data(gct_name)
model = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
                                affinity=str2func[distance_metric])
model.fit(data)

##fig = plt.figure(dpi=300)
##order_of_columns = plot_dendrogram(model, dist=str2dist[distance_metric], labels=plot_labels)

# scipy.cluster.hierarchy.linkage(col_distance_matrix, method='average')
# plt.clf()
# sns.clustermap(data_df[order_of_columns], metric=cusca.mydist, cmap="viridis", method='average')
# plt.savefig('clustermap.png')

# Creating outputs.
##print("Creating outputs now.")
##plt.savefig(output_base_name+'_sample_cluster.png', dpi=300, bbox_inches='tight')

##cusca.list2cls(model.labels_, name_of_out=output_base_name+'_column_labels.cls')

##order_of_rows = data_df.index.values

# Independent clustering by rows
# if row_distance_metric:
#     model_T = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
#                                       affinity=str2func[distance_metric])
#     model_T.fit(np.transpose(data))
#     fig.clf()
#     # order_of_rows = plot_dendrogram(row_model, dist=str2dist[distance_metric], labels=row_lables)
#     order_of_rows = two_plot_two_dendrogram(model_T, dist=str2dist[distance_metric], labels=row_lables)
#     # plt.savefig('sample_cluster.png', dpi=300, bbox_inches='tight')
#     # two_plot_two_dendrogram(model, top=int(np.floor(len(data_df)/2)), col_order=order_of_columns)
#     plt.savefig(output_base_name+'_sample_cluster_2.png', dpi=300, bbox_inches='tight')
#     cusca.list2cls(model_T.labels_, name_of_out=output_base_name+'_row_labels.cls')

##plot_heatmap(data_df, top=int(np.floor(len(data_df)/2)), col_order=order_of_columns, row_order=order_of_rows)


if output_distances:
    row_distance_matrix = str2affinity_func[distance_metric](data)
    # col_distance_matrix = str2affinity_func[distance_metric](np.transpose(data))
    dist_file = open(output_base_name+'_pairwise_distances.csv', 'w')
    dist_file.write('labels,')
    dist_file.write(",".join(model.labels_.astype(str))+"\n")
    dist_file.write('samples,')
    dist_file.write(",".join(list(data_df))+"\n")
    i = 0
    for row in row_distance_matrix:
        dist_file.write('distances row='+str(i)+","+",".join(row.astype(str)) + "\n")
        i += 1

# Create outputs compatible with HCV
# print("making the tree")
tree = make_tree(model, data)
# print(tree)

# my_affinity_generic(data, str2func[distance_metric])
row_distance_matrix = my_affinity_generic(data, str2dist[distance_metric])
col_distance_matrix = my_affinity_generic(np.transpose(data), str2dist[distance_metric])

AID = make_atr(tree, col_distance_matrix, file_name='test.atr')
make_cdt(data=full_gct, name='test.cdt', atr_companion=True, gtr_companion=False, AID=AID)
