import os
import sys
from subprocess import call

WORKING_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
ROOT = os.path.join(WORKING_DIR, '..')
TASKLIB = os.path.join(ROOT, 'src/')
INPUT_FILE_DIR = os.path.join(ROOT, 'data/')

gct_file = 'test_dataset.gct'
# func = 'euclidean'
# func = 'pearson'
# func = 'manhattan'
# func = 'uncentered_pearson'
# func = 'absolute_pearson'
# func = 'absolute_uncentered_pearson'
# func = 'spearman'
# func = 'kendall'
func = 'euclidean'
command = "python HierarchicalClustering.py "+INPUT_FILE_DIR+gct_file+" "+func+" True"
print("About to make this command line call\n\t", command)
call(command, shell=True)
