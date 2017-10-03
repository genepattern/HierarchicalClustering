import sys
import cuzcatlan as cusca
import numpy as np
from statistics import mode
from sklearn.metrics import pairwise
from sklearn import metrics

from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from sklearn.cluster import AgglomerativeClustering
import scipy
import itertools

def parse_inputs(args=sys.argv):
    # inp = []
    # inp = args
    # Error handling:
    arg_n = len(args)
    if arg_n == 1:
        sys.exit("Not enough parameters files were provided. This module needs a GCT file to work.")
    elif arg_n == 2:
        gct_name = args[1]
        col_distance_metric = 'euclidean'
        output_distances = False
        row_distance_metric = 'No row clustering'
        clustering_method = 'Pairwise average-linkage'
        output_base_name = 'HC'
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric = euclidean (default value)")
        print("\toutput_distances =", output_distances, "(default: not computing it and creating a file)")
        print("\trow_distance_metric =", row_distance_metric, "(default: No row clustering)")
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC)")
    elif arg_n == 3:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = False
        row_distance_metric = 'No row clustering'
        clustering_method = 'Pairwise average-linkage'
        output_base_name = 'HC'
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances, "(default: not computing it and creating a file)")
        print("\trow_distance_metric =", row_distance_metric, "(default: No row clustering)")
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC)")
    elif arg_n == 4:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = 'No row clustering'
        clustering_method = 'Pairwise average-linkage'
        output_base_name = 'HC'
        if (output_distances == 'False') or (output_distances == 'F')\
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True
        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric, "(default: No row clustering)")
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC)")
    elif arg_n == 5:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = args[4]
        clustering_method = 'Pairwise average-linkage'
        # clustering_method = 'Pairwise complete-linkage'
        output_base_name = 'HC'
        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True
        # if (row_distance_metric == 'False') or (row_distance_metric == 'F') \
        #         or (row_distance_metric == 'false') or (row_distance_metric == 'f')\
        #         or (row_distance_metric == 'No row clustering'):
        #     row_distance_metric = False

        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric)
        print("\tclustering_method =", clustering_method, "(default: Pairwise average-linkage)")
        print("\toutput_base_name =", output_base_name, "(default: HC)")
    elif arg_n == 6:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = args[4]
        clustering_method = args[5]
        if clustering_method not in linkage_dic.keys():
            exit("Clustering method chosen not supported. This should not have happened.")

        if (linkage_dic[clustering_method] == 'ward') and (col_distance_metric != 'average'):
            exit("When choosing 'Pairwise ward-linkage' the distance metric *must* be 'average' ")

        output_base_name = 'HC'
        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True
        # if (row_distance_metric == 'False') or (row_distance_metric == 'F') \
        #         or (row_distance_metric == 'false') or (row_distance_metric == 'f')\
        #         or (row_distance_metric == 'No row clustering'):
        #     row_distance_metric = False

        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric)
        print("\tclustering_method =", clustering_method)
        print("\toutput_base_name =", output_base_name, "(default: HC)")
    elif arg_n == 7:
        gct_name = args[1]
        col_distance_metric = args[2]
        output_distances = args[3]
        row_distance_metric = args[4]
        clustering_method = args[5]
        output_base_name = args[6]
        if (output_distances == 'False') or (output_distances == 'F') \
                or (output_distances == 'false') or (output_distances == 'f'):
            output_distances = False
        else:
            output_distances = True
        # if (row_distance_metric == 'False') or (row_distance_metric == 'F') \
        #         or (row_distance_metric == 'false') or (row_distance_metric == 'f')\
        #         or (row_distance_metric == 'No row clustering'):
        #     row_distance_metric = False

        print("Using:")
        print("\tgct_name =", gct_name)
        print("\tcol_distance_metric =", col_distance_metric)
        print("\toutput_distances =", output_distances)
        print("\trow_distance_metric =", row_distance_metric)
        print("\tclustering_method =", clustering_method)
        print("\toutput_base_name =", output_base_name)
    else:
        sys.exit("Too many inputs. This module needs only a GCT file to work, "
                 "plus an optional input choosing between Pearson Correlation or Information Coefficient.")

    return gct_name, col_distance_metric, output_distances, row_distance_metric, clustering_method, output_base_name


def plot_dendrogram(model, data, tree, axis, dist=cusca.mydist, title='no_title.png', **kwargs):
    plt.clf()
    #modified from https://github.com/scikit-learn/scikit-learn/pull/3464/files
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    #TODO: Fix this cusca.mydist
    # distance = cusca.dendodist(children, euclidian_similarity)
    # distance = cusca.dendodist(children, dist)

    distance = np.cumsum(better_dendodist(children, dist, tree, data, axis=axis))

    # print(distance)
    if all(value == 0 for value in distance):
        # If all distances are zero, then use uniform distance
        distance = np.arange(len(distance))

    print(distance)
    # print(np.cumsum(distance))

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
    plt.savefig(title, dpi=300)
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


def my_affinity_generic(M, metric):
    return np.array([np.array([metric(a, b) for a in M])for b in M])


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
    full_gct = data_df.copy()
    full_gct.drop(['Name'], axis=1, inplace=True)
    data_df.drop(['Name', 'Description'], axis=1, inplace=True)
    # plot_short_labels = [item[1] + "{:02d}".format(i) for item, i in zip(list(data_df), range(len(list(data_df))))]
    # data_df.columns = plot_short_labels
    plot_labels = list(full_gct.drop(['Description'], axis=1, inplace=False))
    data = data_df.as_matrix().T
    row_labels = data_df.index.values
    return data, data_df, plot_labels, row_labels, full_gct


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


def make_tree(model, data=None):
    """
    Modified from:
    https://stackoverflow.com/questions/27386641/how-to-traverse-a-tree-from-sklearn-agglomerativeclustering
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    import itertools

    X = np.concatenate([np.random.randn(3, 10), np.random.randn(2, 10) + 100])
    model = AgglomerativeClustering(linkage="average", affinity="cosine")
    model.fit(X)

    ii = itertools.count(X.shape[0])
    [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in model.children_]

    ---

    You can also do dict(enumerate(model.children_, model.n_leaves_))
    which will give you a dictionary where the each key is the ID of a node
    and the value is the pair of IDs of its children. – user76284

    :param model:
    :return: a dictionary where the each key is the ID of a node and the value is the pair of IDs of its children.
    """
    # ii = itertools.count(data.shape[0])  # Setting the counter at the number of leaves.
    # tree = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in model.children_]
    # print(tree)
    # return tree

    return dict(enumerate(model.children_, model.n_leaves_))
    # return dict(enumerate(model.children_, 1))


def make_cdt(data, AID, order_of_columns, GID, order_of_rows, name='test.cdt', atr_companion=True, gtr_companion=False):
    # TODO: if order_of_columns == None, then do arange(len(list(data)))
    # TODO: if order_of_rows == None, then do arange(len(list(data)))

    data.index.name = "ID"
    data.rename(columns={'Description': 'Name'}, inplace=True)

    temp = np.ones(len(data))
    data.insert(loc=1, column='GWEIGHT', value=temp)  # adding an extra column

    # These three lines add a row
    data.loc['EWEIGHT'] = list(np.ones(len(list(data))))
    newIndex = ['EWEIGHT'] + [ind for ind in data.index if ind != 'EWEIGHT']
    data = data.reindex(index=newIndex)

    if atr_companion:
        new_AID = ['', '']
        for element in np.sort(np.unique(AID)):
            if 'NODE' in element:
                pass
            else:
                new_AID.append(element)

        data.loc['AID'] = new_AID
        newIndex = ['AID'] + [ind for ind in data.index if ind != 'AID']
        data = data.reindex(index=newIndex)
        data = data[['Name', 'GWEIGHT']+order_of_columns]
    if gtr_companion:
        new_GID = ['']
        if atr_companion:
            new_GID = ['AID', 'EWEIGHT']  # This is to make sure we fit the CDT format
        for element in np.sort(np.unique(GID)):
            if 'NODE' in element:
                # print(element, 'GTR delete')
                pass
            else:
                new_GID.append(element)


        data.insert(loc=0, column='GID', value=new_GID)  # adding an extra column
        data.insert(loc=0, column=data.index.name, value=data.index)  # Making the index a column

        # reorder to match dendogram
        temp = ['AID', 'EWEIGHT'] + order_of_rows
        data = data.loc[temp]
        # print(list(data.index))
        # print(data['GID'])
        # print(data['Name'])

        # Making the 'GID' the index -- for printing purposes
        data.index = data['GID']
        data.index.name = 'GID'
        data.drop(['GID'], axis=1, inplace=True)
        # print(list(data.index))

    # The first three lines need to be written separately due to a quirk in the CDT file format:


    # print(data.to_csv(sep='\t', index=True, header=True))
    f = open(name, 'w')
    f.write(data.to_csv(sep='\t', index=True, header=True))
    f.close()
    return


def make_atr(dic, data, order_of_data_columns, file_name='test.atr'):
    # print('Current ATR:')
    val = len(dic)
    max_val = len(dic)
    AID = []
    val -= 2

    # compute distances
    distance_dic = {}
    for key, value in dic.items():
        # print(value[0], value[1])
        val = centroid_distances(value[0], value[1], tree=dic, data=data, axis=1, distance=cusca.mydist)
        # val = centroid_distances(value[0], value[1], tree=dic, data=data, axis=1, distance=norm_euclidian)
        distance_dic[key] = val

    for key, value in distance_dic.items():
        distance_dic[key] = 1/(1+distance_dic[key])

    f = open(file_name, 'w')
    # norm = my_affinity_generic(np.transpose(data), cusca.mydist).max()
    for key, value in dic.items():
        # print(value[0], value[1])
        # dist = str(centroid_distances(value[0], value[1], tree=dic, data=data, axis=1, distance=cusca.mydist))
        # dist = str(centroid_distances(value[0], value[1], tree=dic, data=data, axis=1, distance=cusca.custom_pearson))
        elements = [translate_tree(key, max_val, 'atr'), translate_tree(value[0], max_val, 'atr'),
                    translate_tree(value[1], max_val, 'atr'), str(distance_dic[key])]
        # print('\t', '\t'.join(elements))
        AID.append(translate_tree(value[0], max_val, 'atr'))
        AID.append(translate_tree(value[1], max_val, 'atr'))
        f.write('\t'.join(elements) + '\n')
    f.close()

    return AID

def translate_tree(what, length, g_or_a):
    if 'a' in g_or_a:
        if what <= length:
            translation = 'ARRAY'+str(what)+'X'
        else:
            translation = 'NODE' + str(what-length) + 'X'
    elif 'g' in g_or_a:
        if what <= length:
            translation = 'GENE'+str(what)+'X'
        else:
            translation = 'NODE' + str(what-length) + 'X'
    else:
        translation = []
        print('This function does not support g_or_a=', g_or_a)
    return translation


def make_gtr(dic, data, file_name='test.gtr'):
    val = len(dic)
    max_val = len(dic)
    GID = []
    val -= 2

    # compute distances
    distance_dic = {}
    for key, value in dic.items():
        # print(value[0], value[1])
        val = centroid_distances(value[0], value[1], tree=dic, data=data, axis=0, distance=cusca.mydist)
        # val = centroid_distances(value[0], value[1], tree=dic, data=data, axis=0, distance=norm_euclidian)
        distance_dic[key] = val

    # norm = max(distance_dic.values())

    for key, value in distance_dic.items():
        distance_dic[key] = 1/(1+distance_dic[key])

    # norm = min(distance_dic.values())

    # for key, value in distance_dic.items():
    #     distance_dic[key] = distance_dic[key]/norm

    f = open(file_name, 'w')
    for key, value in dic.items():
        # elements = ['NODE'+str(key)+'X', 'ARRY'+str(value[0])+'X', 'ARRY'+str(value[1])+'X', str(val/len(dic))]

        # dist = str(val/len(dic))

        # dist = str(val)
        # dist = str(1)
        elements = [translate_tree(key, max_val, 'gtr'), translate_tree(value[0], max_val, 'gtr'),
                    translate_tree(value[1], max_val, 'gtr'), str(distance_dic[key])]
        # elements = [str(single_element) for single_element in elements]
        # print('\t'.join(elements))
        GID.append(translate_tree(value[0], max_val, 'gtr'))
        GID.append(translate_tree(value[1], max_val, 'gtr'))
        f.write('\t'.join(elements)+'\n')
        val -= 1
    f.close()
    # print(GID)
    return GID


# def get_children_recursively(k, model, node_dict, leaf_count, n_samples, data, verbose=False, left=None, right=None):
#     # print(k)
#     i, j = model.children_[k]
#
#     if k in node_dict:
#         return node_dict[k]['children']
#
#     if i < leaf_count:
#         # print("i if")
#         left = [i]
#     else:
#         # print("i else")
#         # read the AgglomerativeClustering doc. to see why I select i-n_samples
#         left, node_dict = get_children_recursively(i - n_samples, model, node_dict,
#                                                    leaf_count, n_samples, data, verbose, left, right)
#
#     if j < leaf_count:
#         # print("j if")
#         right = [j]
#     else:
#         # print("j else")
#         right, node_dict = get_children_recursively(j - n_samples, model, node_dict,
#                                                     leaf_count, n_samples, data, verbose, left, right)
#
#     if verbose:
#         print(k, i, j, left, right)
#     temp = map(lambda ii: data[ii], left)
#     left_pos = np.mean(list(temp), axis=0)
#     temp = map(lambda ii: data[ii], right)
#     right_pos = np.mean(list(temp), axis=0)
#
#     # this assumes that agg_cluster used euclidean distances
#     dist = metrics.pairwise_distances([left_pos, right_pos], metric='euclidean')[0, 1]
#
#     all_children = [x for y in [left, right] for x in y]
#     pos = np.mean(list(map(lambda ii: data[ii], all_children)), axis=0)
#
#     # store the results to speed up any additional or recursive evaluations
#     node_dict[k] = {'top_child': [i, j], 'children': all_children, 'pos': pos, 'dist': dist,
#                     'node_i': k + n_samples}
#     return all_children, node_dict

# def recursive_atr


def get_children(tree, leaves_are_self_children=False):
    # this is a recursive function
    expanded_tree = {}
    for node in range(max(tree.keys())):
        if node <= len(tree):
            if leaves_are_self_children:
                expanded_tree[node] = [node]
            else:
                expanded_tree[node] = []

        else:
            # expanded_tree[node] = list_children_single_node(node, tree)
            expanded_tree[node] = list_children_single_node(node, tree, leaves_are_self_children)

    return expanded_tree


def list_children_single_node(node, tree, leaves_are_self_children=False, only_leaves_are_children=True):
    # children = []
    if node <= len(tree):
        if leaves_are_self_children:
            children = [node]
        else:
            children = []

    else:
        children = list(tree[node])

        # Check each child, and add their children to the list
        for child in children:
            if child <= len(tree):
                pass
            else:
                children += list_children_single_node(child, tree, only_leaves_are_children=True)
    if only_leaves_are_children:
        # print(sorted(np.unique(i for i in children if i <= len(tree))))
        # print()
        return [i for i in sorted(np.unique(children))if i <= len(tree)]
    else:
        return sorted(np.unique(children))


def centroid_distances(node_a, node_b, tree, data, axis=0, distance=cusca.mydist):
    if axis == 0:
        data = np.transpose(data)
        # print("no changes", data.shape)
    elif axis == 1:
        # data = np.transpose(data)
        # print("Transpose", data.shape)
        pass
    else:
        exit("Variable 'data' does not have that many axises (╯°□°)╯︵ ┻━┻")

    # data = np.transpose(data)

    # number_of_leaves = len(tree)
    children_of_a = list_children_single_node(node_a, tree=tree, leaves_are_self_children=True)
    children_of_b = list_children_single_node(node_b, tree=tree, leaves_are_self_children=True)
    # print(data[children_of_a])
    # centroid_of_a = np.mean(data[children_of_a], axis=0)
    # print(data[children_of_b])
    # centroid_of_b = np.mean(data[children_of_b], axis=0)

    # print(distance(centroid_of_a, centroid_of_b))
    # print(axis)
    # print(data.shape)
    # print(children_of_a, children_of_b)
    # print(data)
    # print(data[children_of_a])
    # print(children_of_b)

    distances_list = []

    if distance in [cusca.mydist, euclidian_similarity]:
        for pair in itertools.product(data[children_of_a][:], data[children_of_b][:]):
            # print(pair[0], pair[1])
            distances_list.append(distance(pair[0], pair[1]))
        # return 1 / sum(distances_list)
        # return np.average([1/dist for dist in distances_list])
        # return 1 / np.average(distances_list)
        # return max(distances_list)
        return np.average(distances_list)

    elif distance in [cusca.custom_pearson]:
        for pair in itertools.product(data[children_of_a], data[children_of_b]):
            distances_list.append(abs(distance(pair[0], pair[1])))
        # print(distances_list)
        return np.prod(distances_list)
        # return np.average(distances_list)

    else:
        Warning('Using a custom distance.')
        for pair in itertools.product(data[children_of_a], data[children_of_b]):
            distances_list.append(distance(pair[0], pair[1]))
        return sum(distances_list)


def euclidian_similarity(x, y):
    dist = cusca.mydist(x, y)
    # return 1/(1+dist)
    return 1/(np.exp(dist))


def better_dendodist(children, distance, tree, data, axis):
    distances_list = []
    for pair in children:
        # print(pair[0], pair[1], data.shape)
        # print(distance)
        distances_list.append(centroid_distances(pair[0], pair[1], tree, data, axis, distance=euclidian_similarity))

    return distances_list
