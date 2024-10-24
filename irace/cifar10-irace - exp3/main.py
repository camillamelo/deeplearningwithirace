# -*- coding: utf-8 -*-
import os
import argparse
import tensorflow as tf
import keras
import time
import gc
import random
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.datasets import cifar10
from keras.backend import clear_session
from keras.backend import clear_session
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.callbacks import EarlyStopping


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
                   datfile, confid, Xtrain, Ytrain, Xtest, Ytest):
    clear_session()
    tf.keras.backend.clear_session()
    start_time = time.time()

    model = define_model(convLayers, maxPooling, batchN, numFilters, sizeFilters, denseLayers, denseNeurons, dropout)
    model.summary()
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
        #verbose=True,
        callbacks=[es]
    )
 
    end_time = time.time()

    total_time = end_time - start_time #tempo total em segundos
    output = model.evaluate(Xtest,Ytest)[0]
  
    with open('irace/'+datfile, 'w') as f:
       f.write(str(output))
       f.write("\n")
       f.write(str(total_time))
  
    del model
    del train_generator
    del data_generator
    gc.collect()

#Loading Dataset
n_classes = 10
num_folds = 10

(x_train, y_train), (x_valid, y_valid) = cifar10.load_data()

x_train = x_train.astype('float32') / 255
x_valid = x_valid.astype('float32') / 255

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_valid = tf.keras.utils.to_categorical(y_valid, n_classes)

# Merge inputs and targets
inputs = np.concatenate((x_train, x_valid), axis=0)
targets = np.concatenate((y_train, y_valid), axis=0)
# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=False)
folds = list(kfold.split(inputs, targets))

if __name__ == '__main__':
    #feature_group=0 #0-6
    #check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                #tf.config.set_logical_device_configuration(gpu,
                #                                           [tf.config.LogicalDeviceConfiguration(memory_limit=6700)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
	
    if os.getcwd().endswith('irace') or os.getcwd().endswith('irace/'):
        os.chdir('..')
    ap = argparse.ArgumentParser()
    ap.add_argument('--nc', dest='nc', type=int, required=True, help='camadasConvolucionais')

    ap.add_argument('--mp1', dest='mp1', type=str, required=True, help='Max Pooling Camada 1')
    ap.add_argument('--mp2', dest='mp2', type=str, required=False, help='Max Pooling Camada 2')
    ap.add_argument('--mp3', dest='mp3', type=str, required=False, help='Max Pooling Camada 3')
    ap.add_argument('--mp4', dest='mp4', type=str, required=False, help='Max Pooling Camada 4')
    ap.add_argument('--mp5', dest='mp5', type=str, required=False, help='Max Pooling Camada 5')
    ap.add_argument('--mp6', dest='mp6', type=str, required=False, help='Max Pooling Camada 6')
    ap.add_argument('--mp7', dest='mp7', type=str, required=False, help='Max Pooling Camada 7')
    ap.add_argument('--mp8', dest='mp8', type=str, required=False, help='Max Pooling Camada 8')

    ap.add_argument('--bn1', dest='bn1', type=str, required=True, help='batch normalization Camada 1')
    ap.add_argument('--bn2', dest='bn2', type=str, required=False, help='batch normalization Camada 2')
    ap.add_argument('--bn3', dest='bn3', type=str, required=False, help='batch normalization Camada 3')
    ap.add_argument('--bn4', dest='bn4', type=str, required=False, help='batch normalization Camada 4')
    ap.add_argument('--bn5', dest='bn5', type=str, required=False, help='batch normalization Camada 5')
    ap.add_argument('--bn6', dest='bn6', type=str, required=False, help='batch normalization Camada 6')
    ap.add_argument('--bn7', dest='bn7', type=str, required=False, help='batch normalization Camada 7')
    ap.add_argument('--bn8', dest='bn8', type=str, required=False, help='batch normalization Camada 8')

    ap.add_argument('--nf2', dest='nf2', type=int, required=False, help='indice numero filtros camada 2')
    ap.add_argument('--nf3', dest='nf3', type=int, required=False, help='indice numero filtros camada 3')
    ap.add_argument('--nf4', dest='nf4', type=int, required=False, help='indice numero filtros camada 4')
    ap.add_argument('--nf5', dest='nf5', type=int, required=False, help='indice numero filtros camada 5')
    ap.add_argument('--nf6', dest='nf6', type=int, required=False, help='indice numero filtros camada 6')
    ap.add_argument('--nf7', dest='nf7', type=int, required=False, help='indice numero filtros camada 7')
    ap.add_argument('--nf8', dest='nf8', type=int, required=False, help='indice numero filtros camada 8')
    
    ap.add_argument('--tf2', dest='tf2', type=int, required=False, help='tamanho filtros camada 2')
    ap.add_argument('--tf3', dest='tf3', type=int, required=False, help='tamanho filtros camada 3')
    ap.add_argument('--tf4', dest='tf4', type=int, required=False, help='tamanho filtros camada 4')
    ap.add_argument('--tf5', dest='tf5', type=int, required=False, help='tamanho filtros camada 5')
    ap.add_argument('--tf6', dest='tf6', type=int, required=False, help='tamanho filtros camada 6')
    ap.add_argument('--tf7', dest='tf7', type=int, required=False, help='tamanho filtros camada 7')
    ap.add_argument('--tf8', dest='tf8', type=int, required=False, help='tamanho filtros camada 8')

    ap.add_argument('--nd', dest='nd', type=int, required=True, help='camadasDensas')

    ap.add_argument('--nd1', dest='nd1', type=int, required=False, help='neurônios na camada densa 1')
    ap.add_argument('--nd2', dest='nd2', type=int, required=False, help='neurônios na camada densa 2')

    ap.add_argument('--dr', dest='dr', type=float, required=True, help='dropout rate')
    ap.add_argument('--li', dest='li', type=float, required=True, help='learning index')
    ap.add_argument('--bti', dest='bti', type=int, required=True, help='batch_index')
    ap.add_argument('--ne', dest='ne', type=int, required=True, help='max_epochs')

    ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')
    ap.add_argument('--config-id', dest='confid', type=str, required=False, help='config_id')
    ap.add_argument('--seed', dest='seed', type=int, required=True, help='Random seed')
    ap.add_argument('-i', dest='i', type=int, required=True, help='Instância (k-fold)')
        
    args,remaining_args=ap.parse_known_args()

    random.seed(args.seed)
    #learning rate = 10 ** -li
    learnrate = 10**(-int(args.li))
    #batch size = 2 ** bti
    batchsize = 2**(int(args.bti))
    #multiple layers params
    args_dict=vars(args)

    convLayers = args.nc
    maxPooling = []
    batchNorm = []
    numFilters = []
    sizeFilters = []
    
    for l in range(convLayers):
      maxPooling.append(args_dict['mp{}'.format(l+1)]  in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade'))
      batchNorm.append(args_dict['bn{}'.format(l+1)]  in ('true', '1', 't', 'y', 'yes', 'sim', 'verdade'))
      if l > 0:
        numFilters.append(2 ** args_dict['nf{}'.format(l+1)])
        sizeFilters.append(args_dict['tf{}'.format(l+1)])

    denseLayers = args.nd
    denseNeurons = []
    for d in range(denseLayers-1):
       denseNeurons.append(args_dict['nd{}'.format(d+1)])

    fold = folds[args.i]
    print("Fold: ", args.i)

    evaluate_model(args.ne,batchsize,learnrate,
                   convLayers, maxPooling, batchNorm, numFilters, sizeFilters, denseLayers, denseNeurons, args.dr,
                   args.datfile,args.confid, inputs[fold[0]], targets[fold[0]], inputs[fold[1]], targets[fold[1]])
	