import os
import sys
from subprocess import call

WORKING_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
ROOT = os.path.join(WORKING_DIR, '..')
TASKLIB = os.path.join(ROOT, 'src/')
INPUT_FILE_DIR = os.path.join(ROOT, 'data/')

gct_file = 'test_dataset.gct'
func = 'euclidean'
command = "python HierarchicalClustering.py "+INPUT_FILE_DIR+gct_file+" "+func
print("About to make this command line call\n\t", command)
call(command, shell=True)
