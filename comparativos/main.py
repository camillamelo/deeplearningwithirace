# -*- coding: utf-8 -*-
import os
import argparse
import tensorflow as tf
import keras
import time
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
#from matplotlib import pyplot as plt
from keras.datasets import cifar10
from keras.backend import clear_session
from keras.backend import clear_session
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.callbacks import EarlyStopping

exp_name='exp2'

#Neural Network
def define_model(convLayers, maxPooling, batchN, numFilters, sizeFilters, denseLayers, denseNeurons, dropout):

  model = Sequential()
  #camadas convolucionais
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
  if maxPooling[0]:
     model.add(MaxPooling2D())
  if batchN[0]:
     model.add(BatchNormalization())
  for i in range(convLayers-1):
     model.add(
        Conv2D(
           filters=numFilters[i],
           kernel_size=sizeFilters[i],
           activation="relu",
           padding='same'
        )
      )
     if maxPooling[i+1]: #0 é o da 1a camada
        model.add(MaxPooling2D())
     if batchN[i+1]: #0 é o da 1a camada
        model.add(BatchNormalization())
  model.add(Dropout(dropout))
  model.add(Flatten())
  #camadas densas
  for d in range(denseLayers-1):
     model.add(Dense(denseNeurons[d], activation='tanh'))
     model.add(Dropout(dropout))

  model.add(Dense(n_classes, activation="softmax"))
  return model


def evaluate_model(epochs, batch_size, learning_rate, 
                   convLayers, maxPooling, batchN, numFilters, sizeFilters, denseLayers, denseNeurons, dropout, 
                   Xtrain, Ytrain, Xtest, Ytest):
    clear_session()
    tf.keras.backend.clear_session()
    start_time = time.time()

    model = define_model(convLayers, maxPooling, batchN, numFilters, sizeFilters, denseLayers, denseNeurons, dropout)
    #model.summary()
    print('Taxa de aprendizado: ', learning_rate, 'Num de épocas: ', epochs, 'Tamanho do lote: ', batch_size, 'Dropout: ', dropout)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])
    
    #Use Data Generator to avoid OOM errors
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator()
    train_generator = data_generator.flow(Xtrain, Ytrain, batch_size=batch_size)
    #Early Stopping
    if epochs > 100:
       patience = int(epochs * 0.05)
    else:
       patience = 5
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

    model.fit(
        train_generator,
        validation_data=(Xtest, Ytest),
        shuffle=True,
        epochs=epochs,
        verbose=0,
        callbacks=[es]
    )
 
    end_time = time.time()

    total_time = end_time - start_time #tempo total em segundos
    output = model.evaluate(Xtest,Ytest)
  
    del model
    del train_generator
    del data_generator
    gc.collect()

    return output, total_time

#Loading Dataset
n_classes = 10
num_folds = 5

(x_train, y_train), (x_valid, y_valid) = cifar10.load_data()

x_train = x_train.astype('float32') / 255
x_valid = x_valid.astype('float32') / 255

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_valid = tf.keras.utils.to_categorical(y_valid, n_classes)

# Merge inputs and targets
inputs = np.concatenate((x_train, x_valid), axis=0)
targets = np.concatenate((y_train, y_valid), axis=0)
# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

if __name__ == '__main__':
    
    #check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                #tf.config.set_logical_device_configuration(gpu,
                #                                           [tf.config.LogicalDeviceConfiguration(memory_limit=3500)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
	
    history = pd.DataFrame(columns=('ID', 'fold', 'exp_mean_loss', 'loss', 'accuracy', 'time'))
    df_results = pd.read_csv(exp_name+" geral.csv",sep=";",decimal=",")
    for index, linha in df_results.iterrows():
       #learning rate = 10 ** -li
       learnrate = 10**(-linha["learningIndex"])
       #batch size = 2 ** bti
       batchsize = 2**(linha["batchIndex"])
       #multiple layers params
      
       convLayers = linha["camadasConvolucionais"]
       maxPooling = []
       batchNorm = []
       numFilters = []
       sizeFilters = []
    
       for l in range(convLayers):
          maxPooling.append(linha['maxpooling{}'.format(l+1)]  == 1)
          batchNorm.append(linha['batchNormalization{}'.format(l+1)] == 1)
          if l > 0:
             numFilters.append(2 ** linha['numFiltrosInd{}'.format(l+1)])
             sizeFilters.append(int(linha['tamanhoFiltros{}'.format(l+1)]))

       denseLayers = linha["camadasDensas"]
       denseNeurons = []
       for d in range(denseLayers-1):
          denseNeurons.append(linha['neuroniosDensos{}'.format(d+1)])

       fold_no = 1
       for train, test in kfold.split(inputs, targets):
          
          output, total_time = evaluate_model(linha["numEpocas"],batchsize,learnrate,
                   convLayers, maxPooling, batchNorm, numFilters, sizeFilters, denseLayers, denseNeurons, linha["dropout"],
                   inputs[train], targets[train], inputs[test], targets[test])
          df_aux = pd.DataFrame({
             'ID': [linha['ID']], 
             'fold': fold_no,
             'exp_mean_loss': [linha['Media']],
             'loss': [output[0]],
             'accuracy': [output[1]],
             'time': [total_time]
             })
          print(df_aux)
          history = pd.concat([history, df_aux], ignore_index=True)
          # Increase fold number
          fold_no = fold_no + 1

    print(history)
    history.to_csv(exp_name+" final.csv")
	