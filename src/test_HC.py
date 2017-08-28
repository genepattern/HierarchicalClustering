from subprocess import call
import os
import numpy as np
from hc_functions import *


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

# def test_known_output():
    # Creating some test labels

