import os
import sys
from subprocess import call

WORKING_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
ROOT = os.path.join(WORKING_DIR, '..')
TASKLIB = os.path.join(ROOT, 'src/')
INPUT_FILE_DIR = os.path.join(ROOT, 'data/')

def do_this(comando):
    print("***************")
    sys.stdout.flush()
    print("About to make this command line call\n\t", comando)
    print("*")
    print("*")
    sys.stdout.flush()
    call(comando, shell=True)

os.chdir("usr/src")

# ##DEBUG##
# print("*****\nDebugging now\n*****")
# do_this("echo $LC_ALL")
# # do_this("pip install /usr/ej_temp")
# print("***************")
# sys.stdout.flush()
# import cuzcatlan as cusca
# # do_this('pip list | grep "cuzcatlan"')
# # do_this('pip list | grep "matplotlib"')
#
# print("*****\nEding debugging now\n*****")


# docker run \
# -v /Users/edjuaro/GoogleDrive/modules/HierarchicalClustering/src:/usr/src \
# -v /Users/edjuaro/GoogleDrive/cuzcatlan:/usr/ej_temp \
# genepattern/docker-python36 python usr/src/command_line.py



# gct_file = 'test_dataset_50x38.gct'
# gct_file = 'test_dataset.gct'
# gct_file = 'minimal_dataset.gct'
# gct_file = 'BRCA_minimal_60x19.gct'
# gct_file = 'OV_data_subset.gct'
# gct_file = 'BRCA_filtered.gct'
# gct_file = 'all_aml_train.gct'
# gct_file = 'all_aml_test.gct'
# gct_file = '/Users/edjuaro/GoogleDrive/modules/download_from_gdc/src/demo.gct'
# gct_file = 'https://datasets.genepattern.org/data/test_data/BRCA_minimal_60x19.gct'
# gct_file = 'https://datasets.genepattern.org/data/test_data/BRCA_large_20783x40.gct'

gct_file = 'https://datasets.genepattern.org/data/test_data/BRCA_minimal_60x19.gct'

# func = 'euclidean'
func = 'pearson'
# func = 'manhattan'
# func = 'uncentered_pearson'
# func = 'absolute_pearson'
# func = 'absolute_uncentered_pearson'
# func = 'spearman'
# func = 'kendall'
# func = 'euclidean'
# func = 'cosine'
# func = 'information_coefficient'

# gct_name, distance_metric, output_distances, row_distance_metric, clustering_method, output_base_name
# command = "python HierarchicalClustering.py "+INPUT_FILE_DIR+gct_file+" "+func+" True "+func
# command = "python HierarchicalClustering.py "+INPUT_FILE_DIR+gct_file+" "+func+" False "+func
# command = "python HierarchicalClustering.py "+INPUT_FILE_DIR+gct_file+" "+"No_column_clustering"+" False "+func
# command = "python HierarchicalClustering.py "+INPUT_FILE_DIR+gct_file+" "+func+" False 0 "+\
#           " m HC_out False False Mean Mean"
# command = "python HierarchicalClustering.py "+gct_file+" "+func+" False 0 "+\
#           " m HC_out False False Mean Mean"

command = "python "+"HierarchicalClustering.py "+gct_file+" "+func+" False 0 "+\
        " m HC_out False False Mean Mean"
print("About to make this command line call\n\t", command)
call(command, shell=True)


# docker run \
# -v /Users/edjuaro/GoogleDrive/modules/HierarchicalClustering/src:/usr/src \
# -v /Users/edjuaro/GoogleDrive/cuzcatlan:/usr/ej_temp \
# -v /Users/edjuaro/GoogleDrive/modules/HierarchicalClustering/src:/usr/data \
# genepattern/docker-python36 python usr/src/command_line.py
