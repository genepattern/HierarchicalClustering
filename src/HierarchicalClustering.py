"""
Created on 2017-08-15 by Edwin F. Juarez.

This module will grab a .gct file to perform hierarchical clustering on the columns.
"""

# TODO: turn name of files into inputs, just like in DiffEx.
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
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from time import time

def custom_pearson(x, y):
    return scipy.stats.pearsonr(x, y)[0]


def myaffintyp(M):
    return np.array([[custom_pearson(a, b) for a in M]for b in M])


def count_mislabels(labels, true):
    return count_diff(labels[:21]) + count_diff(labels[21:])


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


def myaffintye(M):
    global dist_matrix
    dist_matrix = np.array([[mydist(a, b) for a in M]for b in M])
    return dist_matrix


def dendodist(V):
    dists = np.array([mydist(a[0],a[1]) for a in V])
    return np.cumsum(dists)

func_dic = {
    myaffintye: "custom_eucledian",
    #myaffintyi: "custom_ic",
    myaffintyp: "custom_pearson",
    'l1': 'l1',
    'l2': 'l2',
    'manhattan': 'manhattan',
    'cosine': 'cosine',
}

df1 = pd.read_csv("../data/test_dataset.gct", sep='\t', skiprows=2)
# plot_labels = [item[:3] for item in list(df1)]
plot_short_labels = [item[1] for item in list(df1)]
plot_labels = list(df1)
df = pd.read_csv("../data/test_dataset.gct", sep='\t', skiprows=2)
df.drop(['Name', 'Description'], axis=1, inplace=True)
data = df.as_matrix().T

# Reading the cls for the
df = pd.read_csv("../data/test_dataset.cls", sep=' ', skiprows=2, header=None)
new_labels = np.asarray(df.as_matrix().T)
new_labels = new_labels.reshape(new_labels.shape[0],)

# clustering = AgglomerativeClustering(linkage='average', n_clusters=2, affinity=myaffintyp)
# clustering = AgglomerativeClustering(linkage='average', n_clusters=2, affinity='euclidean')
clustering = AgglomerativeClustering(linkage='average', n_clusters=2, affinity=myaffintye)
t0 = time()
clustering.fit(data)
# TODO: improve the count_mislabels function. E.g., it outputs the wrong value when all labels are the same
# todo: also, the index for separating the lables is currently hardcoded. I should replace this function with one that
# todo: reads the correct lables and knows how many elements in a row should appear, saves this as a vector...
# todo: and checks that the created lables follow the same pattern. This will scale for more than two lables.
print("{} : {:3.2g}s : {} mislabels".format(
    func_dic[myaffintyp], time() - t0, count_mislabels(clustering.labels_, new_labels)))

# print(new_labels)
# print(clustering.labels_)

# import itertools
# X = data
model = clustering
# ii = itertools.count(X.shape[0])
# [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in model.children_]

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
#     print(children)
#     distance = np.arange(children.shape[0])
    distance = dendodist(children)
#     print(distance)

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
# plot_dendrogram(model, labels=plot_labels)

fig = plt.figure(dpi=300)
# plot_dendrogram(model, labels=plot_short_labels)
plot_dendrogram(model, labels=plot_labels)
[label.set_rotation(90) for label in plt.gca().get_xticklabels()]
# plt.show()
plt.savefig('sample_cluster.png', dpi=300, bbox_inches='tight')