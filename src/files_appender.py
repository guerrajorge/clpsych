import pandas as pd
import csv
import re
import os
from pathlib import Path
import sys
import numpy as np
from datetime import datetime

project_name = 'clpsych'
project_path = Path(os.getcwd()).parent

if sys.platform == "win32":
    data_path = 'D:\Dataset\{0}\dataset'.format(project_name)
elif sys.platform == 'darwin':
    data_path = '/Volumes/Dataset/{0}/dataset'.format(project_name)
else:
    data_path = Path(project_path, 'dataset')

utils_path = str(Path(project_path, 'utils'))
# including the project folder and the utils folder
if utils_path not in ''.join(sys.path):
    sys.path.extend([str(project_path), utils_path])

print('project path = {0}'.format(project_path))
print('data path = {0}'.format(data_path))
print('')
print('sys.path = {0}'.format(sys.path))

rs_path = Path(data_path, 'submissions')
# get all the json files and their stem
need_processing = [element.replace('_norm.csv', '') for element in os.listdir(rs_path) if 'norm.csv' in element]

for file in need_processing:
    print('file = {0}'.format(file))
    file1name = file + '.csv'
    file1 = Path(rs_path, file1name)
    print('\t reading - {0}'.format(file1)) 
    data1 = pd.read_csv(file1)
    
    file2name = file + '_norm.csv'
    file2 = Path(rs_path, file2name)
    print('\t reading - {0}'.format(file2)) 
    data2 = pd.read_csv(file2)
    
    data2['subreddit'] = data1['subreddit']
    print('storing results in {0}'.format(file2name))
    data2.to_csv(file2, index=False)