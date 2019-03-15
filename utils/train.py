from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import metrics
from datetime import datetime
import logging

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)


# def compile_model(network, nb_classes, input_shape):
def compile_model(network, input_dim):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network
        input_dim (int): number of features
    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            # model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
            model.add(Dense(25, activation=activation, input_dim=input_dim))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(1, activation='sigmoid'))

    # model.compile(loss='categorical_crossentropy', optimizer=optimizer,
    #               metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metrics.mae, metrics.categorical_accuracy])

    return model


def train_and_score(network, x_train, x_test, y_train, y_test):
    """
    Train the model, return test loss
    :param network: the parameters of the network
    :param x_train: training dataset
    :param x_test: testing dataset
    :param y_train: training labels
    :param y_test: testing labels
    :return:
    """

    input_dim = x_train.shape[1]

    model = compile_model(network, input_dim)
    logging.getLogger('regular.time').info('training model')

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    # serialize model to JSON
    time_name = datetime.now().strftime('%Y%m%d%H%M%S')
    file_name = 'nn_' + time_name + '.json'
    model_output_dir = 'models/' + file_name
    weight_file_name = 'nn_weights_' + time_name + '.h5'
    weight_model_output_dir = 'models/' + weight_file_name

    model_json = model.to_json()
    with open(model_output_dir, 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weight_model_output_dir)

    msg = 'Saved model to disk time = {0}'.format(time_name)
    logging.getLogger('regular.time').info(msg=msg)

    score = model.evaluate(x_test, y_test, verbose=0)

    # just want to return the name without the file format to keep syntax standards
    return time_name, model, score[1]  # 1 is accuracy. 0 is loss.
