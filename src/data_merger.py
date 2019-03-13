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

data_norm_files = Path(rs_path, 'dataset_normalized_files.txt')

files_dataset = list()
if data_norm_files.exists():
    # list of file that have been added tothe dataset_normalized file
    with data_norm_files.open('r') as in_file:
        for row in in_file:
            files_dataset.append(row.replace('\n', ''))

dataset = ''
first = True

for file in rs_path.iterdir():
    if '_norm.csv' in file.name and file.name not in files_dataset:
        print('file = {0}'.format(file.stem))
        print('\t reading - {0}'.format( datetime.now()))
        data = pd.read_csv(file)
        data.dropna(axis=0, inplace=True)
        if first:
            dataset = data.copy()
            first = False
        else:
            dataset = dataset.append(data, ignore_index=True)
        files_dataset.append(file.name)

print('storing dataset')        
dataset.to_csv(Path(rs_path, 'dataset_normalized.csv'), index=False)

# list of file that have been added tothe dataset_normalized file
with data_norm_files.open('a') as out_file:
    for file_name in files_dataset:
        out_file.write('{0}\n'.format(file_name))     