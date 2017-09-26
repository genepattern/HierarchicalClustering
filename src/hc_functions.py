import sys
import cuzcatlan as cusca
import numpy as np
from statistics import mode
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_inputs(args=sys.argv):
    # inp = []
    # inp = args
    # Error handling:
    arg_n = len(args)
    if arg_n == 1:
        sys.exit("Not enough parameters files were provided. This module needs a GCT file to work.")
    elif arg_n == 2:
        gct_name = args[1]
        distance_metric = 'euclidean'
        output_distances = False
        cluster_by_rows = False
        clustering_method = 'Pairwise average-linkage'
        output_base_name = 'HC'
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tdistance_metric = euclidean (default value)")
        print("\toutput_distances =", output_distances, "(default: not computing it and creating a file)")
        print("\tcluster_by_rows =", cluster_by_rows, "(default: not clustering by rows)")
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC)")
    elif arg_n == 3:
        gct_name = args[1]
        distance_metric = args[2]
        output_distances = False
        cluster_by_rows = False
        clustering_method = 'Pairwise average-linkage'
        output_base_name = 'HC'
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tdistance_metric =", distance_metric)
        print("\toutput_distances =", output_distances, "(default: not computing it and creating a file)")
        print("\tcluster_by_rows =", cluster_by_rows, "(default: not clustering by rows)")
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC)")
    elif arg_n == 4:
        gct_name = args[1]
        distance_metric = args[2]
        output_distances = args[3]
        cluster_by_rows = False
        clustering_method = 'Pairwise average-linkage'
        output_base_name = 'HC'
        if (output_distances == 'False') or (output_distances == 'F')\
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tdistance_metric =", distance_metric)
        print("\toutput_distances =", output_distances)
        print("\tcluster_by_rows =", cluster_by_rows, "(default: not clustering by rows)")
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC)")
    elif arg_n == 5:
        gct_name = args[1]
        distance_metric = args[2]
        output_distances = args[3]
        cluster_by_rows = args[4]
        clustering_method = 'Pairwise average-linkage'
        output_base_name = 'HC'
        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True
        if (cluster_by_rows == 'False') or (cluster_by_rows == 'F') \
                or (cluster_by_rows == 'false') or (cluster_by_rows == 'f'):
            cluster_by_rows = False
        else:
            cluster_by_rows = True
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tdistance_metric =", distance_metric)
        print("\toutput_distances =", output_distances)
        print("\tcluster_by_rows =", cluster_by_rows)
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC)")
    elif arg_n == 6:
        gct_name = args[1]
        distance_metric = args[2]
        output_distances = args[3]
        cluster_by_rows = args[4]
        clustering_method = args[5]
        if clustering_method not in linkage_dic.keys():
            exit("Clustering method chosen not supported. This should not have happened.")

        if (linkage_dic[clustering_method] == 'ward') and (distance_metric != 'average'):
            exit("When choosing 'Pairwise ward-linkage' the distance metric *must* be 'average' ")

        output_base_name = 'HC'
        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True
        if (cluster_by_rows == 'False') or (cluster_by_rows == 'F') \
                or (cluster_by_rows == 'false') or (cluster_by_rows == 'f'):
            cluster_by_rows = False
        else:
            cluster_by_rows = True
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tdistance_metric =", distance_metric)
        print("\toutput_distances =", output_distances)
        print("\tcluster_by_rows =", cluster_by_rows)
        print("\tclustering_method =", clustering_method)
        print("\toutput_base_name =", output_base_name, "(default: HC)")
    elif arg_n == 7:
        gct_name = args[1]
        distance_metric = args[2]
        output_distances = args[3]
        cluster_by_rows = args[4]
        clustering_method = args[5]
        output_base_name = args[6]
        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True
        if (cluster_by_rows == 'False') or (cluster_by_rows == 'F') \
                or (cluster_by_rows == 'false') or (cluster_by_rows == 'f'):
            cluster_by_rows = False
        else:
            cluster_by_rows = True
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tdistance_metric =", distance_metric)
        print("\toutput_distances =", output_distances)
        print("\tcluster_by_rows =", cluster_by_rows)
        print("\tclustering_method =", clustering_method)
        print("\toutput_base_name =", output_base_name)
    else:
        sys.exit("Too many inputs. This module needs only a GCT file to work, "
                 "plus an optional input choosing between Pearson Correlation or Information Coefficient.")

    return gct_name, distance_metric, output_distances, cluster_by_rows, clustering_method, output_base_name


def plot_dendrogram(model, dist=cusca.mydist, **kwargs):
    #modified from https://github.com/scikit-learn/scikit-learn/pull/3464/files
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    distance = cusca.dendodist(children, dist)
    if all(value == 0 for value in distance):
        # If all distances are zero, then use uniform distance
        distance = np.arange(len(distance))

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # Plot the corresponding dendrogram
    R = dendrogram(linkage_matrix, color_threshold=0, **kwargs)
    [label.set_rotation(90) for label in plt.gca().get_xticklabels()]
    order_of_columns = R['ivl']
    # print(order_of_columns)

    plt.gca().get_yaxis().set_visible(False)
    return order_of_columns


def two_plot_two_dendrogram(model, dist=cusca.mydist, **kwargs):
    #modified from https://github.com/scikit-learn/scikit-learn/pull/3464/files
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    distance = cusca.dendodist(children, dist)
    if all(value == 0 for value in distance):
        # If all distances are zero, then use uniform distance
        distance = np.arange(len(distance))

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # Plot the corresponding dendrogram
    R = dendrogram(linkage_matrix, color_threshold=0, orientation='left', **kwargs)
    # [label.set_rotation(90) for label in plt.gca().get_xticklabels()]
    order_of_rows = R['ivl']
    # print(order_of_columns)
    plt.gca().get_xaxis().set_visible(False)

    return list(reversed(order_of_rows))


def my_affinity_i(M):
    return np.array([[cusca.information_coefficient(a, b) for a in M]for b in M])


def my_affinity_ai(M):
    return np.array([[cusca.absolute_information_coefficient(a, b) for a in M]for b in M])


def my_affinity_p(M):
    return np.array([[cusca.custom_pearson(a, b) for a in M]for b in M])


def my_affinity_s(M):
    return np.array([[cusca.custom_spearman(a, b) for a in M]for b in M])


def my_affinity_k(M):
    return np.array([[cusca.custom_kendall_tau(a, b) for a in M]for b in M])


def my_affinity_ap(M):
    return np.array([[cusca.absolute_pearson(a, b) for a in M]for b in M])


def my_affinity_u(M):
    return np.array([[cusca.uncentered_pearson(a, b) for a in M]for b in M])


def my_affinity_au(M):
    return np.array([[cusca.absolute_uncentered_pearson(a, b) for a in M]for b in M])


def my_affinity_l1(M):
    return np.array([[pairwise.paired_manhattan_distances(a, b) for a in M]for b in M])


def my_affinity_l2(M):
    return np.array([[pairwise.paired_euclidean_distances(a, b) for a in M]for b in M])


def my_affinity_m(M):
    return np.array([[pairwise.paired_manhattan_distances(a, b) for a in M]for b in M])


def my_affinity_c(M):
    return np.array([[pairwise.paired_cosine_distances(a, b) for a in M]for b in M])


def my_affinity_e(M):
    global dist_matrix
    dist_matrix = np.array([[cusca.mydist(a, b) for a in M]for b in M])
    return dist_matrix


def count_diff(x):
    count = 0
    compare = x[0]
    for i in x:
        if i != compare:
            count += 1
    return count


def count_mislabels(labels, true_labels):
    # 2017-08-17: I will make the assumption that clusters have only 2 values.
    # clusters = np.unique(true_labels)
    # mislabels = 0
    # for curr_clust in clusters:
    #     print("for label", curr_clust)
    #     print("\t", labels[(true_labels == curr_clust)])
    #     compare_to = mode(labels[(true_labels == curr_clust)])
    #     print("\tcompare to:", compare_to, "mislables: ", np.count_nonzero(labels[(true_labels == curr_clust)] != compare_to))
    #     mislabels += np.count_nonzero(labels[(true_labels == curr_clust)] != compare_to)

    set_a = labels[true_labels == 0]
    set_b = labels[true_labels == 1]

    if len(set_a) <= len(set_b):
        shorter = set_a
        longer = set_b
    else:
        shorter = set_b
        longer = set_a

    long_mode = mode(longer)  # this what the label of the longer cluster should be.
    short_mode = 1 if long_mode == 0 else 0  # Choose the other value for the label of the shorter cluster

    # start with the longer vector:
    # print("The long set is", longer, "it has", np.count_nonzero(longer != long_mode), 'mislabels.')
    # print("The short set is", shorter, "it has", np.count_nonzero(shorter != short_mode), 'mislabels.')

    # np.count_nonzero(longer != long_mode) + np.count_nonzero(shorter != short_mode)

    return np.count_nonzero(longer != long_mode) + np.count_nonzero(shorter != short_mode)


def plot_heatmap(df, col_order, row_order, top=5, title_text='differentially expressed genes per phenotype'):
    if not(len(col_order), len(list(df))):
        exit("Number of columns in dataframe do not match the columns provided for ordering.")
    if not(len(row_order), len(df)):
        exit("Number of rows in dataframe do not match the columns provided for ordering.")
    # print(list(df), col_order)
    df = df[col_order]
    df = df.reindex(row_order)

    plt.clf()
    sns.heatmap(df.iloc[np.r_[0:top, -top:0], :], cmap='viridis')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Top {} {}'.format(top, title_text))
    plt.ylabel('Genes')
    plt.xlabel('Sample')
    plt.savefig('heatmap.png', dpi=300, bbox_inches="tight")


def parse_data(gct_name):
    data_df = pd.read_csv(gct_name, sep='\t', skiprows=2)
    data_df.set_index(data_df['Name'], inplace=True)
    data_df.drop(['Name', 'Description'], axis=1, inplace=True)
    plot_short_labels = [item[1] + "{:02d}".format(i) for item, i in zip(list(data_df), range(len(list(data_df))))]
    data_df.columns = plot_short_labels
    plot_labels = list(data_df)
    data = data_df.as_matrix().T
    row_labels = data_df.index.values
    return data, data_df, plot_labels, row_labels


str2func = {
    'custom_euclidean': my_affinity_e,
    'uncentered_pearson': my_affinity_u,
    'absolute_uncentered_pearson': my_affinity_au,
    'information_coefficient': my_affinity_i,
    'pearson': my_affinity_p,
    'spearman': my_affinity_s,
    'kendall': my_affinity_k,
    'absolute_pearson': my_affinity_ap,
    'l1': 'l1',
    'l2': 'l2',
    'manhattan': 'manhattan',
    'cosine': 'cosine',
    'euclidean': 'euclidean',
}


str2affinity_func = {
    'custom_euclidean': my_affinity_e,
    'uncentered_pearson': my_affinity_u,
    'absolute_uncentered_pearson': my_affinity_au,
    'information_coefficient': my_affinity_i,
    'pearson': my_affinity_p,
    'spearman': my_affinity_s,
    'kendall': my_affinity_k,
    'absolute_pearson': my_affinity_ap,
    'l1': my_affinity_l1,
    'l2': my_affinity_l2,
    'manhattan': my_affinity_m,
    'cosine': my_affinity_c,
    'euclidean': my_affinity_e,
}

str2dist = {
    'custom_euclidean': cusca.mydist,
    'uncentered_pearson': cusca.uncentered_pearson,
    'absolute_uncentered_pearson': cusca.absolute_pearson,
    'information_coefficient': cusca.information_coefficient,
    'pearson': cusca.custom_pearson,
    'spearman': cusca.custom_spearman,
    'kendall': cusca.custom_kendall_tau,
    'absolute_pearson': cusca.absolute_pearson,
    'l1': pairwise.paired_manhattan_distances,
    'l2': pairwise.paired_euclidean_distances,
    'manhattan': pairwise.paired_manhattan_distances,
    'cosine': pairwise.paired_cosine_distances,
    'euclidean': pairwise.paired_euclidean_distances,
}

linkage_dic = {
    'Pairwise average-linkage': 'average',
    'Pairwise complete-linkage': 'complete',
    'Pairwise ward-linkage': 'ward',
}