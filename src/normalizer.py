import pandas as pd
import csv
import re
import os
from pathlib import Path
import sys
import numpy as np
from datetime import datetime

# function to normalize contents
def clean_str(string):
     # float occurs when the string is emtpy
    if not pd.isnull(string):
        print('\t\t len={0}'.format(len(string)))
        """
        Tokenization/string cleaning for datasets.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = string.replace('  ','')
        string = re.sub(r'((http|https|ftp|ftps)\:\/\/[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(\/\S*)?)', '_URL_', string)
        string = re.sub(r'[\w\.-]+@[\w\.-]+', '_EMAIL_', string)
        string = re.sub(r'address = ([0-9]{1,3}[\.]){3}[0-9]{1,3}', '_IP_', string)
        string = string.replace('[deleted]', '')
        return string.strip()

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
processed_files = [element.replace('norm.csv', '') 
                   for element in os.listdir(rs_path) if 'norm.csv' in element]

need_to_process = ['RS_2016-02',
 'RS_2016-03',
 'RS_2016-04',
 'RS_2016-05']

for file in rs_path.iterdir():
    if file.suffix == '.csv' and file.stem not in processed_files and not file.is_dir() and file.stem in need_to_process:
        print('file = {0}'.format(file.stem))
        print('\t reading - {0}'.format( datetime.now()))
        data = pd.read_csv(file)
        total = data.shape[0]
        print('\t processing - {0}'.format(datetime.now()))
        final = list()
        for row_ix, row in enumerate(data['title'] + data['selftext']):
            print('\t\t {0}/{1}'.format(row_ix, total))
            final.append(clean_str(row))
        print('\t storing - {0}'.format(datetime.now()))
        n_data = pd.DataFrame({'title_body': final})
        new_filename = file.stem + '_norm.csv'
        n_data.to_csv(Path(rs_path, new_filename), index=False)
        print('\t finished {0}'.format(datetime.now()))