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

print(col_distance_metric, (col_distance_metric != 'No column clustering'), (col_distance_metric != 'No_column_clustering'))

if (col_distance_metric != 'No column clustering') and (col_distance_metric != 'No_column_clustering'):
    col_model = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
                                    affinity=str2func[col_distance_metric], compute_full_tree=True)
    # col_model.fit(data)
    y = col_model.fit_predict(data)
    print(y)
    order_of_columns = plot_dendrogram(col_model, dist=str2dist[col_distance_metric], labels=col_labels)
    # order_of_columns = plot_dendrogram(col_model, dist=str2dist[col_distance_metric], labels=col_labels)


if (row_distance_metric != 'No row clustering') and (row_distance_metric != 'No_row_clustering'):
    row_model = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
                                    affinity=str2func[row_distance_metric])
    # row_model.fit(np.transpose(data))
    y_col = row_model.fit_predict(np.transpose(data))
    print(y_col)
    order_of_rows = plot_dendrogram(row_model, dist=str2dist[row_distance_metric], labels=row_labels)

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

# Create outputs compatible with HCV
# print("making the tree")

# print(tree)


# row_distance_matrix = my_affinity_generic(data, str2dist[col_distance_metric])
# col_distance_matrix = my_affinity_generic(np.transpose(data), str2dist[col_distance_metric])

atr_companion = False
gtr_companion = False

AID = None
GID = None
if (col_distance_metric != 'No column clustering') and (col_distance_metric != 'No_column_clustering'):
    col_tree = make_tree(col_model, data)
    atr_companion = True

    order_of_data_columns = [full_gct.columns.get_loc(col) for col in order_of_columns]
    # print(col_tree)

    # print(data)

    AID = make_atr(col_tree, file_name='test.atr', data=data, order_of_data_columns=order_of_data_columns)

    # print(get_children(col_tree, leaves_are_self_children=True))
    # print(list_children_single_node(6, col_tree))




    # Start of what seems to work

    # X = data
    # Z = []
    # # should really call this cluster dict
    # node_dict = {}
    # n_samples = len(data)
    # agg_cluster = col_model
    # leaf_count = col_model.n_leaves_
    # from sklearn import metrics
    #
    # print(X[0])
    #
    #
    # def get_all_children(k, verbose=False):
    #     i, j = agg_cluster.children_[k]
    #
    #     if k in node_dict:
    #         return node_dict[k]['children']
    #
    #     if i < leaf_count:
    #         left = [i]
    #     else:
    #         # read the AgglomerativeClustering doc. to see why I select i-n_samples
    #         left = get_all_children(i - n_samples)
    #
    #     if j < leaf_count:
    #         right = [j]
    #     else:
    #         right = get_all_children(j - n_samples)
    #
    #     if verbose:
    #         print(k, i, j, left, right)
    #     temp = map(lambda ii: X[ii], left)
    #     left_pos = np.mean(list(temp), axis=0)
    #     temp = map(lambda ii: X[ii], right)
    #     right_pos = np.mean(list(temp), axis=0)
    #
    #     # this assumes that agg_cluster used euclidean distances
    #     dist = metrics.pairwise_distances([left_pos, right_pos], metric='euclidean')[0, 1]
    #
    #     all_children = [x for y in [left, right] for x in y]
    #     pos = np.mean(list(map(lambda ii: X[ii], all_children)), axis=0)
    #
    #     # store the results to speed up any additional or recursive evaluations
    #     node_dict[k] = {'top_child': [i, j], 'children': all_children, 'pos': pos, 'dist': dist,
    #                     'node_i': k + n_samples}
    #     return all_children
    #     # return node_di|ct
    #
    #
    # for k, x in enumerate(agg_cluster.children_):
    #     get_all_children(k, verbose=True)
    #
    # # Every row in the linkage matrix has the format [idx1, idx2, distance, sample_count].
    # Z = [[v['top_child'][0], v['top_child'][1], v['dist'], len(v['children'])] for k, v in node_dict.items()]
    # # create a version with log scaled distances for easier visualization
    # # Z_log = [[v['top_child'][0], v['top_child'][1], np.log(1.0 + v['dist']), len(v['children'])] for k, v in
    # #          node_dict.items()]
    #
    # from scipy.cluster import hierarchy
    #
    # # plt.figure()
    # # dn = hierarchy.dendrogram(Z, p=4, truncate_mode='level')
    # # plt.savefig("DELETE_ME_2.png")
    #
    # atr = [[k+leaf_count, v['top_child'][0], v['top_child'][1], v['dist']] for k, v in node_dict.items()]
    #
    # for line in atr:
    #     # translate_tree(what, length, g_or_a)
    #     print([translate_tree(i, 4, 'atr') for i in line])

    # END of What works




    # print("WHAT WHAT")
    # node_dict = {}
    #
    # for k, x in enumerate(col_model.children_):
    #     _, node_dict = get_children_recursively(k, col_model, node_dict, leaf_count, n_samples, data,
    #                                             verbose=True, left=None, right=None)
    # atr = [[k + leaf_count, v['top_child'][0], v['top_child'][1], v['dist']] for k, v in node_dict.items()]
    #
    # for line in atr:
    #     # translate_tree(what, length, g_or_a)
    #     print([translate_tree(i, 4, 'atr') for i in line])





if (row_distance_metric != 'No row clustering') and (row_distance_metric != 'No_row_clustering'):
    row_tree = make_tree(row_model, data)
    # print(data)
    # print(col_tree)
    print(row_tree)
    # print(np.transpose(data))
    gtr_companion = True
    GID = make_gtr(row_tree, data=data, file_name='test.gtr')
# print(['Description']+order_of_columns)
# full_gct = full_gct[['Description']+order_of_columns]  # Reordering the columns
# print(list(full_gct))

# GID = ['GID_'+element for element in order_of_rows]

make_cdt(data=full_gct, name='test.cdt', atr_companion=atr_companion, gtr_companion=gtr_companion,
         AID=AID, order_of_columns=order_of_columns, GID=GID, order_of_rows=order_of_rows)
