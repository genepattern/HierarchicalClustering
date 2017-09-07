"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
Modified by:
    Edwin F. Juárez
    ejuarez@ucsd.edu
    Mesirov Lab -- UCSD Medicine Department.
"""

RANDOM_SEED = 20121020

import sys

from .computational_cancer_biology import (association, gsea, inference,
                                           mutual_vulnerability)
from .support.file import (load_data_table, read_gct, read_gmt, read_gmts,
                           write_data_table, write_gct, write_rnk)
from .support.plot import (plot_clustermap, plot_distribution, plot_heatmap,
                           plot_nmf, plot_points, plot_violin_box_or_bar)
from .support.system import install_libraries

sys.setrecursionlimit(10000)

# Wont install libraries
install_libraries([
#     'rpy2',
    'biopython',
    'plotly',
])
