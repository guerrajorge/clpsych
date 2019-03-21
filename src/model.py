import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from pycorenlp import StanfordCoreNLP
from keras import models
from keras import layers
from keras import regularizers
from utils.data_helpers import build_vocab
from utils.data_helpers import load_data_and_labels
from utils.data_helpers import pad_sentences

project_name = 'clpsych'
project_path = Path(os.getcwd()).parent

if sys.platform == "win32":
    data_path = 'D:\Dataset\{0}\dataset'.format(project_name)
    model_path = Path(project_path, 'models')
    src_path = src_path = Path(project_path, 'src')

elif sys.platform == 'darwin':
    data_path = '/Volumes/Dataset/{0}/dataset'.format(project_name)
    model_path = '/Volumes/Dataset/{0}/models'.format(project_name)
    src_path = '/Volumes/Dataset/{0}/src'.format(project_name)

else:
    data_path = Path(project_path, 'dataset')
    model_path = Path(project_path, 'models')
    src_path = Path(project_path, 'src')

utils_path = str(Path(project_path, 'utils'))
# including the project folder and the utils folder
if utils_path not in ''.join(sys.path):
    sys.path.extend([str(project_path), str(utils_path), str(src_path)])

print('project path = {0}'.format(project_path))
print('data path = {0}'.format(data_path))
print('model path = {0}'.format(model_path))
print('utils path = {0}'.format(utils_path))
print('sys.path = {0}'.format(sys.path))

n = 1
filename = 'risk_title_body_{0}.csv'.format(n)
suicide_data = pd.read_csv(Path(data_path, filename))

nlp = StanfordCoreNLP('http://localhost:9000')

sentiment_list = list()
sentiment_df = pd.DataFrame(columns=['text', 'sentiments', 'sentiment_dist', 'y'])

total_len = suicide_data.shape[0]

print('filename = {0}'.format(filename))

for ix, post in suicide_data.iterrows():
    print('processing {0}/{1}'.format(ix, total_len))
    res = nlp.annotate(post['title_body'],
                       properties={'timeout': 60000000, 'annotators': 'sentiment', 'outputFormat': 'json'})
    sentiment = list()
    sentiment_dist = list()
    for s in res["sentences"]:
        # print("{0}: {1}: {2} {3}".format( s["index"], " ".join([t["word"] for t in s["tokens"]]),
        # s["sentimentValue"], s["sentiment"]))
        sentiment.append(s["sentiment"])
        sentiment_dist.append(s['sentimentDistribution'])
    sentiment_df = sentiment_df.append(
        {'text': post['title_body'], 'sentiments': sentiment, 'sent dist': sentiment_dist,
         'y': post['risk_label']}, ignore_index=True)

filename = 'risk_title_body_{0}_results.csv'.format(n)
sentiment_df.to_csv(Path(data_path, filename), index=False)
