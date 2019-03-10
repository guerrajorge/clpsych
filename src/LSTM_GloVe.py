from keras.layers import Input, Dense, Embedding, LSTM, Dropout
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from data_helpers import load_data_and_labels
from data_helpers import build_vocab
from data_helpers import pad_sentences
from data_helpers import build_input_data
from data_helpers import create_embedding_matrix
from data_helpers import get_labels
from data_helpers import plot_confusion_matrix
from data_helpers import multiclass_roc_auc_score
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


print('Loading annotated social text data')
x_text, y_class = load_data_and_labels()
y = get_labels()

sentences_padded, sequence_length = pad_sentences(x_text)

# global variebles
embedding_dim = 200
num_filters = 512
drop = 0.5
epochs = 1
batch_size = 100

# define 10-fold cross validation test harness
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cvscores = []
auc_scores = []


print('10 fold CV starting')
for train, test in kfold.split(sentences_padded, y_class):
    # split train & test set
    print('spliting train and test set')
    X_train = list()
    X_test = list()
    for index in train:
        X_train.append(sentences_padded[index])
    for index in test:
        X_test.append(sentences_padded[index])
    y_train = y_class[train]
    y_test = y_class[test]

    # building vocabulary on train set
    print('building vocabulary on train set')
    vocabulary, vocabulary_inv = build_vocab(X_train)

    # Maps sentences to vectors based on vocabulary
    print('Mapping sentences to vectors based on vocabulary')
    X_train, y_train = build_input_data(X_train, y_train, vocabulary)
    # print(X_train.shape)
    X_test, y_test = build_input_data(X_test, y_test, vocabulary)
    # all x and y for predicting
    x, y_class = build_input_data(sentences_padded, y_class, vocabulary)
    # print(X_test.shape)
    vocabulary_size = len(vocabulary_inv)

    # building embedding matrix using GloVe word embeddings
    print('building embedding matrix using GloVe word embeddings')
    embedding_matrix = create_embedding_matrix('./dataset/myGloVe200d.txt', vocabulary, embedding_dim)

    # this returns a tensor
    print("Creating Model...")
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=sequence_length)(inputs)

    lstm = LSTM(num_filters, kernel_initializer='normal', activation='relu')(embedding)

    dropout = Dropout(drop)(lstm)
    output = Dense(units=4, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

# checkpoint = ModelCheckpoint('./model/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    print("Training Model...")
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)  # starts training

    # evaluate the model
    print("Evaluate Model...")
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

    print('Saving Model...')
    model_name = 'LSTM_GloVe_' + str(len(cvscores))
    model.save('./model/'+ model_name + '.hdf5')  # creates a HDF5 file 'my_model.h5'

    print('Saving vocabulary to .json file')
    with open('./vocabulary/' + model_name + '.json', 'w') as fp:
        json.dump(vocabulary, fp)

    print('Predicting categories...')
    y_pred = model.predict(x)
    y_classes = y_pred.argmax(axis=-1)
    auc_score = multiclass_roc_auc_score(y, y_classes, average="weighted")
    print("%s: %.2f%%" % ('Average AUC', auc_score * 100))
    auc_scores.append(auc_score * 100)
    df_y_classes = pd.DataFrame(y_classes)
    df_y = pd.DataFrame(y)
    result = pd.concat([df_y, df_y_classes], axis=1)
    result.columns = ['true_class', 'predict_class']
    result.to_csv('./results/' + model_name + 'result.csv', encoding='utf-8', index = False)

    print('Generating confusion matrix...')
    conf_mat = confusion_matrix(y, y_classes)

    print('Plotting results...')
    fig, ax = plt.subplots(figsize=(10, 10))
    labels = ['Economic', 'Education', 'Health Care', 'Housing', 'Interaction with the legal system',
              'Occupational', 'Other', 'social environment', 'Spiritural Life',
              'Support circumstances and networks', 'Transportaion']

    plot_confusion_matrix(conf_mat, classes=labels, normalize=True,
                          title='Normalized confusion matrix')
    plt.gcf().subplots_adjust(bottom=0.15)

    print('Saving plots...')
    fig.savefig('./figure/' + model_name + 'result.png')
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
print("auc" + "%.2f%% (+/- %.2f%%)" % (np.mean(auc_scores), np.std(auc_scores)))