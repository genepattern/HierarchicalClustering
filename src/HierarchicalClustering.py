"""
Created on 2017-08-15 by Edwin F. Juarez.

This module will grab a .gct file to perform hierarchical clustering on the columns.
"""

# TODO: Create a function that parses the inputs.

import os
import sys
tasklib_path = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(tasklib_path + "/ccalnoir")
import matplotlib as mpl
mpl.use('Agg')
import ccalnoir as ccal
from ccalnoir.mathematics.information import information_coefficient
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from time import time
from statistics import mode
import cuzcatlan as cusca
sns.set_style("white")

def parse_inputs(args=sys.argv):
    # inp = []
    # inp = args
    # Error handling:
    arg_n = len(sys.argv)
    if arg_n == 1:
        sys.exit("Not enough parameters files were provided. This module needs a GCT file to work.")
    elif arg_n == 2:
        gct_name = sys.argv[1]
        distance_metric = 'euclidean'
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tdistance_metric = euclidean (default value)")
    elif arg_n == 3:
        gct_name = sys.argv[1]
        distance_metric = sys.argv[2]
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tdistance_metric =", distance_metric)
        print()
    else:
        sys.exit("Too many inputs. This module needs only a GCT file to work, "
                 "plus an optional input choosing between Pearson Correlation or Information Coefficient.")

    return gct_name, distance_metric

gct_name, distance_metric = parse_inputs()


def plot_dendrogram(model, **kwargs):
    #modified from https://github.com/scikit-learn/scikit-learn/pull/3464/files
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    distance = dendodist(children)
    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # Plot the corresponding dendrogram
    R = dendrogram(linkage_matrix, **kwargs)
    [label.set_rotation(90) for label in plt.gca().get_xticklabels()]
    order_of_columns = R['ivl']
    # print(order_of_columns)

    plt.gca().get_yaxis().set_visible(False)
    return order_of_columns


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


def count_diff(x):
    count = 0
    compare = x[0]
    for i in x:
        if i != compare:
            count += 1
    return count


# a custom function that just computes Euclidean distance
def mydist(p1, p2):
    diff = p1 - p2
    return np.vdot(diff, diff) ** 0.5


def my_affinity_e(M):
    global dist_matrix
    dist_matrix = np.array([[mydist(a, b) for a in M]for b in M])
    return dist_matrix


def dendodist(V):
    dists = np.array([mydist(a[0],a[1]) for a in V])
    return np.cumsum(dists)

# func_dic = {
#     my_affinity_e: "custom_eucledian",
#     #my_affinity_i: "custom_ic",
#     my_affinity_p: "custom_pearson",
#     'l1': 'l1',
#     'l2': 'l2',
#     'manhattan': 'manhattan',
#     'cosine': 'cosine',
#     'euclidean': 'euclidean',
# }

str2func = {
    'custom_euclidean': my_affinity_e,
    'uncentered_pearson': my_affinity_u,
    'absolute_uncentered_pearson': my_affinity_au,
    #'custom_ic': my_affinity_i,
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

data_df = pd.read_csv("../data/test_dataset.gct", sep='\t', skiprows=2)
# plot_labels = [item[:3] for item in list(df1)]
data_df.set_index(data_df['Name'], inplace=True)
data_df.drop(['Name', 'Description'], axis=1, inplace=True)
plot_short_labels = [item[1]+"{:02d}".format(i) for item, i in zip(list(data_df), range(len(list(data_df))))]
data_df.columns = plot_short_labels

plot_labels = list(data_df)
# df = pd.read_csv("../data/test_dataset.gct", sep='\t', skiprows=2)
# df.drop(['Name', 'Description'], axis=1, inplace=True)
data = data_df.as_matrix().T

# Reading the cls for the
df = pd.read_csv("../data/test_dataset.cls", sep=' ', skiprows=2, header=None)
new_labels = np.asarray(df.as_matrix().T)
new_labels = new_labels.reshape(new_labels.shape[0],)

# af_to_use = my_affinity_p
# for distance_metric in str2func.keys():
model = AgglomerativeClustering(linkage='average', n_clusters=2, affinity=str2func[distance_metric])
# model = AgglomerativeClustering(linkage='average', n_clusters=2, affinity='euclidean')
# clustering = AgglomerativeClustering(linkage='average', n_clusters=2, affinity=my_affinity_e)
t0 = time()
model.fit(data)
print("{:>16s}: {:3.2g}s\t: {} mislabels".format(
    distance_metric, time() - t0, count_mislabels(model.labels_, new_labels)))
# print(new_labels)
# print(model.labels_)
# print(np.sum(new_labels == model.labels_), '"Real errors"')

# print(new_labels)
# print(clustering.labels_)
# import itertools
# X = data
# model = clustering
# ii = itertools.count(X.shape[0])
# print([{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in model.children_])

# print(model.children_)


fig = plt.figure(dpi=300)
# order_of_columns = plot_dendrogram(model, labels=plot_short_labels)
order_of_columns = plot_dendrogram(model, labels=plot_labels)

# plt.show()
plt.savefig('sample_cluster.png', dpi=300, bbox_inches='tight')


def plot_heatmap(df, col_order, top=5, title_text='differentially expressed genes per phenotype'):
    if not(len(col_order), len(list(df))):
        exit("Number of columns in dataframe do not match the columns provided for ordering.")
    # print(list(df), col_order)
    df = df[col_order]
    plt.clf()
    sns.heatmap(df.iloc[np.r_[0:top, -top:0], :], cmap='coolwarm')
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title('Top {} {}'.format(top, title_text))
    plt.ylabel('Genes')
    plt.xlabel('Sample')
    plt.savefig('heatmap.png', dpi=300, bbox_inches="tight")

np.floor(len(data_df)/2)
plot_heatmap(data_df, top=int(np.floor(len(data_df)/2)), col_order=order_of_columns)


# Creating outputs.
cusca.list2cls(model.labels_, name_of_out='labels.cls')

# print(model.labels_)
# print(list(data_df))
