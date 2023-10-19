import numpy as np
import matplotlib.pyplot as plt
import sys 
import csv


sys.path.append('..')
from helpers import *
from costs import *
from implementations import *

#%load_ext autoreload
#%autoreload 2

# LOAD DATA

x_train, x_test, y_train, train_ids, test_ids = load_csv_data('dataset_to_release')


#PROCESS DATA




