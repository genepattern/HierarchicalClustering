"""
Created on 2017-08-15 by Edwin F. Juarez.

This module will grab a .gct file to perform hierarchical clustering on the columns.
"""
from timeit import default_timer as timer
beginning_of_time = timer()
import os
from hc_functions import *
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering  # We are not using this anymore
import matplotlib as mpl
import humanfriendly
import datetime
from inspect import currentframe
tasklib_path = os.path.dirname(os.path.realpath(sys.argv[0]))
# mpl.use('Agg')
sns.set_style("white")
import fastcluster
from scipy.spatial.distance import pdist

DEBUG = True


def get_linenumber():
    # https://stackoverflow.com/questions/3056048/filename-and-line-number-of-python-script
    cf = currentframe()
    return cf.f_back.f_lineno


def log(text, line_number='?', debug=True):
    if debug:
        print(':::::', datetime.datetime.now().time(), line_number, "===> ", text)
        sys.stdout.flush()


log("About to parse the inputs", get_linenumber(), DEBUG)

# Parse the inputs -- This is using my depreciated "parse_inputs" approach
# TODO: use python's argparse instead (see "download_from_gdc" for that)
# TODO: here is the git for that: https://github.com/genepattern/download_from_gdc
gct_name, col_distance_metric, output_distances, row_distance_metric, clustering_method, output_base_name, \
    row_normalization, col_normalization, row_centering, col_centering = parse_inputs(sys.argv)

log("About to parse data", get_linenumber(), DEBUG)

# Parse the data, i.e., read the GCT file and create different objects
og_data, og_data_df, data, data_df, col_labels, row_labels, og_full_gct, new_full_gct = \
    parse_data(gct_name, row_normalization, col_normalization, row_centering, col_centering)
order_of_columns = list(data_df)
order_of_rows = list(data_df.index)

data_transpose = np.transpose(data)

# Flags to be used when creating the CDT file
atr_companion = False
gtr_companion = False

# AID = None
# GID = None

log("About to cluster columns", get_linenumber(), DEBUG)

if col_distance_metric != 'No_column_clustering':
    atr_companion = True

    # TO Be DELETED
    # # Set Sklearn's clustering model parameters
    # col_model = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
    #                                     affinity=str2func[col_distance_metric])
    # # fit Sklearn's clustering model
    # col_model.fit(data_transpose)
    # col_tree = make_tree(col_model)
    # order_of_columns = order_leaves(col_model, tree=col_tree, data=data_transpose,
    #                                 dist=str2similarity[col_distance_metric], labels=col_labels, reverse=True)

    #  # fastcluster
    D = pdist(data_transpose, metric=pdist_dict[col_distance_metric])
    # print(row_distance_metric, pdist_dict[row_distance_metric])

    Z = fastcluster.linkage(D, method=linkage_dic[clustering_method])
    numeric_order_of_columns, R = two_plot_2_dendrogram(Z=Z, num_clust=2, no_plot=True)
    # order_of_rows = [row_labels[int(i)] for i in numeric_order_of_rows]  # Getting label names from order of rows
    # order_of_rows = row_labels[numeric_order_of_rows]  # Getting label names from order of rows
    order_of_columns = [col_labels[i] for i in numeric_order_of_columns]

    col_tree = make_tree(Z, scipy=True, n_leaves=len(order_of_columns))

    log("About to write atr file", get_linenumber(), DEBUG)
    # Create atr file
    make_atr(col_tree, file_name=output_base_name+'.atr', data=data,
             dist=str2similarity[col_distance_metric], clustering_method=linkage_dic[clustering_method])

log("About to cluster rows", get_linenumber(), DEBUG)
if row_distance_metric != 'No_row_clustering':
    gtr_companion = True

    # TO Be DELETED
    # # Set Sklearn's clustering model parameters
    # row_model = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
    #                                     affinity=str2func[row_distance_metric])
    #
    # # fit Sklearn's clustering model
    # row_model.fit(data)
    # row_tree = make_tree(row_model, scipy=False)
    # order_of_rows = order_leaves(row_model, tree=row_tree, data=data,
    #                              dist=str2similarity[row_distance_metric], labels=row_labels)

    #  # fastcluster
    D = pdist(data, metric=pdist_dict[row_distance_metric])
    # print(row_distance_metric, pdist_dict[row_distance_metric])

    Z = fastcluster.linkage(D, method=linkage_dic[clustering_method])
    numeric_order_of_rows, R = two_plot_2_dendrogram(Z=Z, num_clust=2, no_plot=True)
    # order_of_rows = [row_labels[int(i)] for i in numeric_order_of_rows]  # Getting label names from order of rows
    # order_of_rows = row_labels[numeric_order_of_rows]  # Getting label names from order of rows
    order_of_rows = [row_labels[i] for i in numeric_order_of_rows]

    row_tree = make_tree(Z, scipy=True, n_leaves=len(order_of_rows))

    log("About to write gtr file", get_linenumber(), DEBUG)
    # Create gtr file
    make_gtr(row_tree, data=data, file_name=output_base_name+'.gtr', dist=str2similarity[row_distance_metric])

log("About to create distance matrix", get_linenumber(), DEBUG)
# Possibly create a distances file
if output_distances:
    row_distance_matrix = str2affinity_func[row_distance_metric](data)
    # col_distance_matrix = str2affinity_func[col_distance_metric](np.transpose(data))  # This is "never" useful info.
    log("About to write the row_distance matrix", get_linenumber(), DEBUG)
    dist_file = open(output_base_name+'_pairwise_distances.csv', 'w')
    dist_file.write('labels,')
    dist_file.write(",".join(col_model.labels_.astype(str))+"\n")
    dist_file.write('samples,')
    dist_file.write(",".join(list(data_df))+"\n")
    i = 0
    for row in row_distance_matrix:
        dist_file.write('distances row='+str(i)+","+",".join(row.astype(str)) + "\n")
        i += 1

log("About to make cdt file", get_linenumber(), DEBUG)
# Make the cdt file
make_cdt(data=new_full_gct, name=output_base_name+'.cdt', atr_companion=atr_companion, gtr_companion=gtr_companion,
         order_of_columns=order_of_columns, order_of_rows=order_of_rows)
end_of_time = timer()
spanned = end_of_time - beginning_of_time
print("We are done! Wall time elapsed:", humanfriendly.format_timespan(spanned))
print("###", spanned)
