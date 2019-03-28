import pandas as pd
from pathlib import Path
import sys
import os
import numpy as np
from numpy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import multi_gpu_model
from sklearn.impute import SimpleImputer


def class_report(y_true, y_pred, y_score=None, average='macro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true {0} is not the same shape as y_pred {1}".format(
            y_true.shape,
            y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    # Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels)

    avg = list(precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average=average))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred-cnt'] = pred_cnt
    class_report_df['pred-cnt'].iloc[-1] = total

    if not (y_score is None):
        # false positive rate
        fpr = dict()
        # true positive rate
        tpr = dict()
        roc_auc = dict()
        for label_ix, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_ix])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"],
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df


def f1(y_true, y_prediction):

    y_prediction = k.round(y_prediction)
    tp = k.sum(k.cast(y_true * y_prediction, 'float'), axis=0)
    fp = k.sum(k.cast((1 - y_true) * y_prediction, 'float'), axis=0)
    fn = k.sum(k.cast(y_true * (1 - y_prediction), 'float'), axis=0)

    p = tp / (tp + fp + k.epsilon())
    r = tp / (tp + fn + k.epsilon())

    f1_metric = 2 * p * r / (p + r + k.epsilon())
    f1_metric = tf.where(tf.is_nan(f1_metric), tf.zeros_like(f1_metric), f1_metric)

    return k.mean(f1_metric)


def f1_loss(y_true, y_prediction):

    tp = k.sum(k.cast(y_true * y_prediction, 'float'), axis=0)
    fp = k.sum(k.cast((1 - y_true) * y_prediction, 'float'), axis=0)
    fn = k.sum(k.cast(y_true * (1 - y_prediction), 'float'), axis=0)

    p = tp / (tp + fp + k.epsilon())
    r = tp / (tp + fn + k.epsilon())

    f1_val = 2 * p * r / (p + r + k.epsilon())
    f1_val = tf.where(tf.is_nan(f1_val), tf.zeros_like(f1_val), f1_val)

    return 1 - k.mean(f1_val)


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


def run_model(data, classifier, k_fold):

    data = data.fillna(-1).copy()

    random_state = 7
    report_with_auc_df = ''

    # keep non control user ids
    k_fold = k_fold[k_fold['user_id'] > 0]

    print('5 fold CV starting')

    for fold_ix in range(1, 6):

        print('\nFold = {0}'.format(fold_ix))

        train_ix = k_fold[(k_fold['fold'] == fold_ix) & (k_fold['train_text'] == 'training')]['user_id']
        test_ix = k_fold[(k_fold['fold'] == fold_ix) & (k_fold['train_text'] == 'test')]['user_id']

        x_train = data[data.index.isin(train_ix)].copy()
        x_test = data[data.index.isin(test_ix)].copy()

        y_train = x_train['target']
        x_train.drop(['target'], axis=1, inplace=True)

        y_test = x_test['target']
        x_test.drop(['target'], axis=1, inplace=True)

        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(x_train)
        x_train = imp_mean.transform(x_train)

        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(x_test)
        x_test = imp_mean.transform(x_test)

        x_train = (x_train - x_train.mean()) / (x_train.max() - x_train.min())
        x_test = (x_test - x_test.mean()) / (x_test.max() - x_test.min())

        if classifier == 'nn':

            available_gpu = k.tensorflow_backend._get_available_gpus()
            available_gpus = len(available_gpu)

            # basic neural network
            model = Sequential()
            model.add(Dense(5000, input_dim=np.shape(x_train)[1], activation='relu'))
            model.add(Dense(2500, activation='relu'))
            model.add(Dense(2000, activation='relu'))
            model.add(Dense(1000, activation='relu'))
            model.add(Dense(500, activation='relu'))
            model.add(Dense(250, activation='relu'))
            model.add(Dense(100, activation='relu'))
            model.add(Dense(50, activation='relu'))
            model.add(Dense(len(y_train.unique()), activation='sigmoid'))

            model = multi_gpu_model(model, gpus=available_gpus)

            # treating every instance of class 1 as 50 instances of class 0
            class_weight = {1: 1, 2: 10, 3: 10, 4: 1}

            # Compile model
            model.compile(optimizer='adam', loss=f1_loss, metrics=[f1])
            model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1,
                      class_weight=class_weight)

            y_pred_train = np.argmax(model.predict(x_train), axis=1)
            y_pred_test = np.argmax(model.predict(x_test), axis=1)
            y_score = model.predict(x_test)

        elif classifier == 'logit':

            model = LogisticRegression(class_weight='balanced', n_jobs=-1,
                                       multi_class='auto', solver='lbfgs',
                                       tol=0.00001, C=10, max_iter=1000, verbose=True,
                                       random_state=random_state)
            model.fit(x_train, y_train)
            y_pred_test = model.predict(x_test)
            y_score = model.predict_proba(x_test)

        report_with_auc = class_report(
            y_true=y_test,
            y_pred=y_pred_test,
            y_score=y_score,
            average='macro')

        cv_column = [fold_ix]
        cv_column.extend([''] * (report_with_auc.index.shape[0] - 1))
        report_with_auc['Fold'] = cv_column
        report_with_auc['Risk-Factor'] = report_with_auc.index
        report_with_auc = report_with_auc.set_index(['Fold', 'Risk-Factor'])

        if fold_ix == 1:
            report_with_auc_df = report_with_auc.copy()
        else:
            report_with_auc_df = report_with_auc_df.append(report_with_auc.copy())

        return report_with_auc_df


if __name__ == "__main__":

    project_path, data_path, model_path = project_paths(project_name='clpsych')

    filename = Path(data_path, 'static_features_pandas_v2.pkl')
    dataset = pd.read_pickle(filename)

    # define 5-fold cross validation test harness
    folds = pd.read_csv(Path(data_path, 'clpsych19_public_crossvalidation_splits.csv'), header=None,
                        names=['fold', 'train_text', 'user_id'])

    report = run_model(data=dataset, classifier='logit', k_fold=folds)

    report.to_csv(Path(data_path, 'report_logit.csv'))
