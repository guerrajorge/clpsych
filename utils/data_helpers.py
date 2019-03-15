import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# from sentence_splitter import SentenceSplitter, split_text_into_sentences
# splitter = SentenceSplitter(language='en')
# from util import TextCleaner, InputReader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path



def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
#     string = re.sub(r"\[.*\]", "Name", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(file_path):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    dataset = pd.read_csv(Path(file_path, 'post_risklabel.csv'))
    post = dataset['post']
    post = [str(s).strip() for s in post]
    user_risk_label = dataset['risk_label']
    # Split by words
    x_text = [clean_str(sent) for sent in post]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    risk_labels = list()
    for label in user_risk_label:
        if label == 'a':
            risk_labels.append([1, 0, 0, 0])
        elif label == 'b':
            risk_labels.append([0, 1, 0, 0])
        elif label == 'c':
            risk_labels.append([0, 0, 1, 0])
        elif label == 'd':
            risk_labels.append([0, 0, 0, 1])
    y = np.asarray(risk_labels)
    return [x_text, y]



def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences, sequence_length


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    sentences_padded_list = list()
    for sentence in sentences:
        sentence_list = list()
        for word in sentence:
            if word in vocabulary:
                sentence_list.append(vocabulary[word])
            else:
                sentence_list.append(np.random.uniform(-0.01, 0.01))
        sentences_padded_list.append(sentence_list)
    x = np.array(sentences_padded_list)
    y = np.array(labels)
    return [x, y]



def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def get_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    annotated_data = pd.read_csv("./dataset/post_risklabel.csv")
    annotation_class = annotated_data['risk_label']
    # Generate labels
    class_labels = list()
    for label in annotation_class:
        if label == 'a':
            class_labels.append(0)
        elif label == 'b':
            class_labels.append(1)
        elif label == 'c':
            class_labels.append(2)
        elif label == 'd':
            class_labels.append(3)
    y = np.asarray(class_labels)
    return y


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.autoscale()
    # plt.tight_layout()



def multiclass_roc_auc_score(truth, pred, average="weighted"):

    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return roc_auc_score(truth, pred, average=average)