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
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
# from time import time
# import cuzcatlan as cusca
sns.set_style("white")
from hc_functions import *

gct_name, col_distance_metric, output_distances, row_distance_metric, \
clustering_method, output_base_name = parse_inputs(sys.argv)

data, data_df, col_labels, row_labels, full_gct = parse_data(gct_name)
order_of_columns = list(data_df)
order_of_rows = list(data_df.index)

data_transpose = np.transpose(data)

atr_companion = False
gtr_companion = False

AID = None
GID = None

if col_distance_metric != 'No_column_clustering':
    atr_companion = True
    col_model = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
                                        affinity=str2func[col_distance_metric], compute_full_tree=True)

    col_model.fit(data_transpose)
    col_tree = make_tree(col_model)
    order_of_columns = order_leaves(col_model, tree=col_tree, data=data_transpose,
                                    dist=str2similarity[col_distance_metric], labels=col_labels, reverse=True)

    AID = make_atr(col_tree, file_name='test.atr', data=data, dist=str2similarity[col_distance_metric])

# if (row_distance_metric != 'No row clustering') and (row_distance_metric != 'No_row_clustering'):
if row_distance_metric != 'No_row_clustering':
    gtr_companion = True
    row_model = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
                                        affinity=str2func[row_distance_metric])
    row_model.fit(data)
    # y_col = row_model.fit_predict(np.transpose(data))
    # print(y_col)
    row_tree = make_tree(row_model)
    # order_of_rows = plot_dendrogram(row_model, tree=row_tree, data=np.transpose(data), title='rows.png', axis=0,
    #                                 dist=str2dist[row_distance_metric], labels=row_labels, count_sort='ascending')
    order_of_rows = order_leaves(row_model, tree=row_tree, data=data,
                                 dist=str2similarity[row_distance_metric], labels=row_labels)
    GID = make_gtr(row_tree, data=data, file_name='test.gtr', dist=str2similarity[row_distance_metric])
    # exit(order_of_rows)

if output_distances:
    #TODO: check wich col or row was selected, or both
    row_distance_matrix = str2affinity_func[row_distance_metric](data)
    # col_distance_matrix = str2affinity_func[col_distance_metric](np.transpose(data))
    dist_file = open(output_base_name+'_pairwise_distances.csv', 'w')
    dist_file.write('labels,')
    dist_file.write(",".join(col_model.labels_.astype(str))+"\n")
    dist_file.write('samples,')
    dist_file.write(",".join(list(data_df))+"\n")
    i = 0
    for row in row_distance_matrix:
        dist_file.write('distances row='+str(i)+","+",".join(row.astype(str)) + "\n")
        i += 1

# if (row_distance_metric != 'No row clustering') and (row_distance_metric != 'No_row_clustering'):
    # print(data)
    # print(col_tree)
    # print(row_tree)
    # print(np.transpose(data))
    # gtr_companion = True
    # GID = make_gtr(row_tree, data=data, file_name='test.gtr')
# print(['Description']+order_of_columns)
# full_gct = full_gct[['Description']+order_of_columns]  # Reordering the columns
# print(list(full_gct))

make_cdt(data=full_gct, name='test.cdt', atr_companion=atr_companion, gtr_companion=gtr_companion,
         AID=AID, order_of_columns=order_of_columns, GID=GID, order_of_rows=order_of_rows)
