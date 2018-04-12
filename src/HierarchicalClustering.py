"""
Created on 2017-08-15 by Edwin F. Juarez.

This module will grab a .gct file to perform hierarchical clustering on the columns.
"""
from timeit import default_timer as timer
beginning_of_time = timer()
import os
from hc_functions import *
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import matplotlib as mpl
import humanfriendly
tasklib_path = os.path.dirname(os.path.realpath(sys.argv[0]))
# mpl.use('Agg')"
sns.set_style("white")

# Parse the inputs -- This is using my depreciated "parse_inputs" approach
# TODO: use python's argparse instead (see "download_from_gdc" for that)
# TODO: here is the git for that: https://github.com/genepattern/download_from_gdc
gct_name, col_distance_metric, output_distances, row_distance_metric, clustering_method, output_base_name, \
    row_normalization, col_normalization, row_centering, col_centering = parse_inputs(sys.argv)

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

if col_distance_metric != 'No_column_clustering':
    atr_companion = True

    # Set Sklearn's clustering model parameters
    col_model = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
                                        affinity=str2func[col_distance_metric])
    # fit Sklearn's clustering model
    col_model.fit(data_transpose)
    col_tree = make_tree(col_model)
    order_of_columns = order_leaves(col_model, tree=col_tree, data=data_transpose,
                                    dist=str2similarity[col_distance_metric], labels=col_labels, reverse=True)
    # Create atr file
    make_atr(col_tree, file_name=output_base_name+'.atr', data=data,
             dist=str2similarity[col_distance_metric], clustering_method=linkage_dic[clustering_method])

if row_distance_metric != 'No_row_clustering':
    gtr_companion = True

    # Set Sklearn's clustering model parameters
    row_model = AgglomerativeClustering(linkage=linkage_dic[clustering_method], n_clusters=2,
                                        affinity=str2func[row_distance_metric])
    # fit Sklearn's clustering model
    row_model.fit(data)
    row_tree = make_tree(row_model)
    order_of_rows = order_leaves(row_model, tree=row_tree, data=data,
                                 dist=str2similarity[row_distance_metric], labels=row_labels)
    # Create gtr file
    make_gtr(row_tree, data=data, file_name=output_base_name+'.gtr', dist=str2similarity[row_distance_metric])

# Possibly create a distances file
if output_distances:
    row_distance_matrix = str2affinity_func[row_distance_metric](data)
    # col_distance_matrix = str2affinity_func[col_distance_metric](np.transpose(data))  # This is "never" useful info.
    dist_file = open(output_base_name+'_pairwise_distances.csv', 'w')
    dist_file.write('labels,')
    dist_file.write(",".join(col_model.labels_.astype(str))+"\n")
    dist_file.write('samples,')
    dist_file.write(",".join(list(data_df))+"\n")
    i = 0
    for row in row_distance_matrix:
        dist_file.write('distances row='+str(i)+","+",".join(row.astype(str)) + "\n")
        i += 1

# Make the cdt file
make_cdt(data=new_full_gct, name=output_base_name+'.cdt', atr_companion=atr_companion, gtr_companion=gtr_companion,
         order_of_columns=order_of_columns, order_of_rows=order_of_rows)
end_of_time = timer()
print("We are done! Wall time elapsed:", humanfriendly.format_timespan(end_of_time - beginning_of_time))
