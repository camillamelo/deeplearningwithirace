#import urllib

import gc
import optuna
import tensorflow as tf
import keras
import pandas
from tensorflow.python.client import device_lib
from keras.backend import clear_session
from keras.datasets import cifar10
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.models import Sequential
from datetime import datetime
from keras.callbacks import EarlyStopping
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileOpenLock

CLASSES = 10


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()
    tf.keras.backend.clear_session()

    #Create model
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 8)
    n_dense_layers = trial.suggest_int("n_dense_layers", 1, 3)
    print(" Convolutional Layers: {}".format(n_conv_layers))
    print(" Dense Layers: {}".format(n_dense_layers))

    model = Sequential()
    #Camadas Convolucionais
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    max_pooling = trial.suggest_categorical("maxpooling", [True, False])
    batchnormalization = trial.suggest_categorical("batchnormalization", [True, False])
    n_pooling = 0
    if max_pooling:
        model.add(MaxPooling2D())
        n_pooling = 1
    if batchnormalization:
        model.add(BatchNormalization())
    
    for i in range(n_conv_layers-1):
        filters_ind = trial.suggest_int("filtersind{}".format(i), 5, 8) #expoente do número de filtros
        model.add(
            Conv2D(
                filters=2 ** filters_ind,
                kernel_size=trial.suggest_categorical("kernel_size{}".format(i), [3, 5, 7, 9, 11]),
                activation="relu",
                padding='same'
            )
        )
        if n_pooling < 4: #maximo 4 max_poolings pois divide a entrada pela metade e 32 = 2 ^ 4
            max_pooling = trial.suggest_categorical("maxpooling{}".format(i), [True, False])
            if max_pooling:
                model.add(MaxPooling2D())
                n_pooling = n_pooling + 1
        batchnormalization = trial.suggest_categorical("batchnormalization{}".format(i), [True, False])
        if batchnormalization:
            model.add(BatchNormalization())
    dropout = trial.suggest_float("dropout", 0, 0.5, step=0.01)
    model.add(Dropout(dropout))
    model.add(Flatten())
 
    #Camadas Densas
    
    for i in range(n_dense_layers-1):
        num_hidden = trial.suggest_int("num_hidden_neurons{}".format(i), 4, 128, log=True)
        model.add(Dense(num_hidden, activation='tanh'))
        model.add(Dropout(dropout))

    model.add(Dense(CLASSES, activation="softmax"))
    #model.summary()

    # Get learning rate
    learnindex=trial.suggest_int("learnind", 0, 5)
    learning_rate = 10**(-learnindex)
    print(" Learning rate: {}".format(learning_rate))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])


    batch_index=trial.suggest_int("batchind", 4, 10)
    batch_size=2**batch_index
    print(" Batch Size: {}".format(batch_size))

    #Use Data Generator to avoid OOM errors
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator()
    train_generator = data_generator.flow(x_train, y_train, batch_size=batch_size)
    
    epochs=trial.suggest_int("epochs", 10, 300)
    print(" Epochs: {}".format(epochs))
    #Early Stopping
    if epochs > 100:
        patience = int(epochs * 0.05)
    else:
        patience = 5
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

    model.fit(
        train_generator,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        epochs=epochs,
        verbose=False,
        callbacks=[es]
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(x_valid, y_valid, verbose=0)
    
    #Release GPU memory
    del model
    del train_generator
    del data_generator
    gc.collect()

    return score[0]


if __name__ == "__main__":
    #check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                #tf.config.set_logical_device_configuration(gpu,
                #                                           [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    #carregar dados
    (x_train, y_train), (x_valid, y_valid) = cifar10.load_data()
    img_x, img_y = x_train.shape[1], x_train.shape[2]
    # Normalizar os dados
    x_train = x_train.astype('float32') / 255
    x_valid = x_valid.astype('float32') / 255

    # Converter rótulos de classe para vetores binários
    y_train = keras.utils.to_categorical(y_train, CLASSES)
    y_valid = keras.utils.to_categorical(y_valid, CLASSES)
    input_shape = (img_x, img_y, 3)

    #storage = "sqlite:///db.sqlite3"
    #Use journal storage for paralel processing
    storage = JournalStorage(JournalFileStorage("optuna-journal.log"))
    #get date and time
    #now = datetime.now()
    #date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    study_name = "Experiment2"
    study_file = study_name + ".csv"
    study = optuna.create_study(storage=storage, direction="minimize", study_name=study_name, load_if_exists = True)
    study.optimize(objective, n_trials=100, n_jobs=3, show_progress_bar=True)
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df_study = study.trials_dataframe()
    df_study.to_csv(study_file)