import threading
from pycorenlp import StanfordCoreNLP
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import csv


def post_sentiment(post, index, nlp):
    """
    for provided post, with one or multiple sentence, returns a sentiment distribution of the post by all the different
    sentiments found
    :param post: reddit post
    :param index: current post index
    :param nlp: nlp library server object
    :return:
    """
    # res = nlp.annotate(post,
    #                    properties={'timeout': 60000000,
    #                                'annotators': 'sentiment',
    #                                'outputFormat': 'json'})
    # # store the distribution of the 5 possible sentiments
    # sentiment_dist = list()
    #
    # # obtain sentiment distribution for each sentence in the post
    # for s in res["sentences"]:
    #     sentiment_dist.append(s['sentimentDistribution'])
    # # find average of the distributions by sentiment
    # dist_sum = np.array(sentiment_dist).sum(axis=0)
    # sentiment_dist = dist_sum / dist_sum.sum()
    # # data point with index and averaged distribution
    # row = np.append(index, sentiment_dist)
    # return row
    res = nlp.annotate(post,
                       properties={'timeout': 60000000,
                                   'annotators': 'ssplit',
                                   'outputFormat': 'json'})
    return len(res['sentences'])


def project_paths(project_name):
    """
    obtain the necessary directory paths
    :param project_name: name of the project
    :return: project_path, data_loc, model_loc
    """
    project_loc = Path(os.getcwd()).parent

    if sys.platform == "win32":
        data_loc = 'D:\Dataset\{0}\dataset'.format(project_name)
        model_loc = Path(project_loc, 'models')

    elif sys.platform == 'darwin':
        data_loc = '/Volumes/Dataset/{0}/dataset'.format(project_name)
        model_loc = '/Volumes/Dataset/{0}/models'.format(project_name)

    else:
        data_loc = Path(project_loc, 'dataset')
        model_loc = Path(project_loc, 'models')

    utils_path = str(Path(project_loc, 'utils'))
    # including the project folder and the utils folder
    if utils_path not in ''.join(sys.path):
        sys.path.extend([str(project_loc), str(utils_path)])

    return project_loc, data_loc, model_loc


if __name__ == "__main__":

    project_path, data_path, model_path = project_paths(project_name='clpsych')

    train_test = 'test'

    n = '1'

    # if train_test == 'train':
    #
    #     input_filename = 'risk_title_body_{0}.csv'.format(n)
    #     # this file will contains the risk, title, body and sentiment associated with each post
    #     out_filename = 'risk_tbs_{0}.csv'.format(n)
    #     out_filename_error = 'risk_tbs_{0}_error.csv'.format(n)
    #     out_file = Path(data_path, out_filename)
    #     out_file_error = Path(data_path, out_filename_error)
    #
    #     print('reading data {0}'.format(input_filename))
    #     suicide_data = pd.read_csv(Path(data_path, input_filename))
    #     suicide_data.columns = ['index', 'risk_label', 'title_body']
    #
    # else:
    #     data_path = Path(data_path, 'clpsych19_test_data')
    #     input_filename = 'clpsych19_test_title_body_{0}.csv'.format(n)
    #     out_filename = 'clpsych19_test_title_body_sentiments_sentence_{0}.csv'.format(n)
    #     out_filename_error = 'clpsych19_test_title_body_sentiments_error_sentence_{0}.csv'.format(n)
    #     out_file = Path(data_path, out_filename)
    #     out_file_error = Path(data_path, out_filename_error)
    #
    #     print('reading data {0}'.format(input_filename))
    #     suicide_data = pd.read_csv(Path(data_path, input_filename))

    data_path = Path(data_path, 'clpsych19_test_data')
    input_filename = 'clpsych19_test_title_body.csv'.format(n)
    out_filename = 'clpsych19_test_title_body_sentiments_sentence.csv'.format(n)
    out_filename_error = 'clpsych19_test_title_body_sentiments_error_sentence.csv'.format(n)
    out_file = Path(data_path, out_filename)
    out_file_error = Path(data_path, out_filename_error)

    print('reading data {0}'.format(input_filename))
    suicide_data = pd.read_csv(Path(data_path, input_filename))

    # -1 for the first document which starts at 0
    last_index = -1
    # obtain the last line processed without error
    if out_file.exists():
        with out_file.open('r') as f:
            for row in reversed(list(csv.reader(f))):
                if len(row) != 0:
                    last_index = float(row[0])
                    break

    # initializing nlp object
    nlp_object = StanfordCoreNLP('http://localhost:900{0}'.format(n))

    total_len = suicide_data.shape[0]

    e_file = out_file_error.open('a+')
    # this out file does not have headers
    with out_file.open('a+') as o_file:
        writer = csv.writer(o_file, delimiter=',')
        writer_error = csv.writer(e_file, delimiter=',')
        for data_ix, data in suicide_data.iterrows():
            # make sure only new data points are being processed
            if train_test == 'train':
                c_last = data['user_id']
            else:
                c_last = data_ix

            if data_ix > last_index:
                print('processing {0}/{1}, file ix = {2}'.format(data_ix, total_len, c_last))
                try:
                    res = post_sentiment(post=data['title_body'], index=c_last, nlp=nlp_object)
                    tmp_list = [data['user_id']]
                    tmp_list.extend([data['post_id']])
                    tmp_list.extend([res])
                    # tmp_list.extend('0')
                    writer.writerow(tmp_list)
                    o_file.flush()
                except:
                    print('error {0}'.format(c_last))
                    writer_error.writerow([c_last])
                    e_file.flush()
                    tmp_list = [data['user_id']]
                    tmp_list.extend([data['post_id']])
                    tmp_list.extend([np.nan])
                    # tmp_list.extend('1')
                    writer.writerow(tmp_list)
                    o_file.flush()
                    continue

    e_file.close()
