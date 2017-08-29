from subprocess import call
import os
import numpy as np
from hc_functions import *
import pickle
import cuzcatlan as cusca
from sklearn.cluster import AgglomerativeClustering
from time import time

# -------------------------
# Begin parser tests


def test_gct_parsing():
    gct_name, __, __ = parse_inputs(['', '../data/test_dataset.gct'])
    assert gct_name == '../data/test_dataset.gct'


def test_distance_metric_parsing():
    __, distance_metric, __ = parse_inputs(['', '../data/test_dataset.gct', 'information_coefficient'])
    assert distance_metric == 'information_coefficient'


def test_default_distance_metric_parsing():
    __, distance_metric, __ = parse_inputs(['', '../data/test_dataset.gct'])
    assert distance_metric == 'euclidean'


def test_default_output_distance_parsing():
    __, __, output_distances = parse_inputs(['', '../data/test_dataset.gct', 'information_coefficient'])
    assert output_distances is False


def test_output_distance_parsing():
    __, __, output_distances = parse_inputs(['', '../data/test_dataset.gct', 'information_coefficient', 'True'])
    assert output_distances is True


def test_sloppy_output_distance_parsing():
    __, __, output_distances = parse_inputs(['', '../data/test_dataset.gct', 'information_coefficient', 'Falso!'])
    assert output_distances is True


def test_picky_output_distance_parsing():
    __, __, output_distances = parse_inputs(['', '../data/test_dataset.gct', 'information_coefficient', 'f'])
    assert output_distances is False


# End of parser tests
# -------------------------
# Begin output tests
# def test_data_parser_output_1():
#     data, __, __ = parse_data('../data/test_dataset.gct')
#     assert data == data

def test_data_parser_output_1():
    data, __, __ = parse_data('../data/test_dataset.gct')
    data_2compare = pickle.load(open('../data/data_2compare.p', 'rb'))
    assert np.allclose(data, data_2compare)

def test_data_parser_output_2():
    __, data_df, __ = parse_data('../data/test_dataset.gct')
    data_df_2compare = pickle.load(open('../data/data_df_2compare.p', 'rb'))
    assert data_df.equals(data_df_2compare)

def test_data_parser_output_3():
    true_labels = ['L00', 'L01', 'L02', 'L03', 'L04', 'L05', 'L06', 'L07', 'L08', 'L09', 'L10', 'L11', 'L12', 'L13',
                   'L14', 'L15', 'L16', 'L17', 'L18', 'L19', 'L20', 'M21', 'M22', 'M23', 'M24', 'M25', 'M26', 'M27',
                   'M28', 'M29', 'M30', 'M31', 'M32', 'M33', 'M34']
    __, __, plot_labels = parse_data('../data/test_dataset.gct')
    assert plot_labels == true_labels

def test_known_output():
    #  Creating some test labels
    new_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    data, __, __ = parse_data('../data/test_dataset.gct')
    test_model = AgglomerativeClustering(linkage='average', n_clusters=2, affinity=str2func['custom_euclidean'])
    test_model.fit(data)
    # We know that the euclidean distance gets one label wrong
    assert count_mislabels(test_model.labels_, new_labels) == 1
