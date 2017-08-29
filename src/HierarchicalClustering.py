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
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from time import time
import cuzcatlan as cusca
sns.set_style("white")
from hc_functions import *

gct_name, distance_metric, output_distances = parse_inputs(sys.argv)
data, data_df, plot_labels = parse_data(gct_name)
model = AgglomerativeClustering(linkage='average', n_clusters=2, affinity=str2func[distance_metric])

# print(plot_labels)

model.fit(data)

# TO MOVE
# END TO MOVE

fig = plt.figure(dpi=300)
order_of_columns = plot_dendrogram(model, dist=str2dist[distance_metric], labels=plot_labels)
plt.savefig('sample_cluster.png', dpi=300, bbox_inches='tight')
plot_heatmap(data_df, top=int(np.floor(len(data_df)/2)), col_order=order_of_columns)

# Creating outputs.
print("Creating outputs now.")
cusca.list2cls(model.labels_, name_of_out='labels.cls')

if output_distances:
    dist_file = open('pairwise_distances.csv', 'w')
    dist_file.write('labels,')
    dist_file.write(",".join(model.labels_.astype(str))+"\n")
    dist_file.write('samples,')
    dist_file.write(",".join(list(data_df))+"\n")
    i = 0
    for row in str2affinity_func[distance_metric](data):
        dist_file.write('distances row='+str(i)+","+",".join(row.astype(str)) + "\n")
        i += 1
