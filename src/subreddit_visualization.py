from urllib import request
from bs4 import BeautifulSoup
import os
from pathlib import Path
import sys
import pandas as pd
import ndjson
import json
import bz2
from io import StringIO
import re

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


# decompreses the bz2 files
comments_path = Path(data_path, 'submissions')
# get all the json files and their stem
processed_files = [element.replace('.csv', '') for element in os.listdir(comments_path) if '.csv' in element]

for file in comments_path.iterdir():
    if (file.suffix == '.bz2' or file.suffix == '') and file.stem not in processed_files and not file.is_dir() and re.search('2[0-9]{3}', str(file))[0] == '2010': 
        try:
            print('processing {0}'.format(file))
            # open file
            if file.suffix == '.bz2':
                zipfile = bz2.BZ2File(file)
                # get the decompressed data
                data = zipfile.read()
                # convert to string
                s = str(data,'utf-8')
                ndata = StringIO(s)
            elif file.suffix == '':
                ndata = file
            # convert json to dataframe
            df = pd.read_json(ndata, lines=True)
            # keep relevant columns
            df = df[['subreddit', 'subreddit_id', 'selftext', 'author', 'title', 'created_utc']].copy()
            filename = file.stem + '.csv'
            new_file = Path(comments_path, filename)
            # store in file
            df.to_csv(new_file, index=False)
            processed_files.append(file.stem)
            print('stored {0}'.format(new_file))
        except:
            print('error')
            error_files.append(file.stem)